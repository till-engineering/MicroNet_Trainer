"""Abschluss-Evaluation auf test: ein oder mehrere Modelle (neu, Referenz) nebeneinander.

Gruppen: gesamt und je Unterordner von test (z. B. eigen / extern), jeweils getrennt nach Bildern mit
Poren und Negativbildern. Jedes Modell mit seiner eigenen Schwelle (model.yaml).

Geschrieben nach <out>/:
    evaluation.xlsx   Summary (Modelle nebeneinander), Summary CI (95-%-Bootstrap), Images, Size classes,
                      Error origin, Curves (Pixel-Precision / -Recall / F1 je Schwelle)
    plots/            pr_curve.png, f1_threshold.png, porosity.png, recall_by_size.png
    overlays/         schlechteste Bilder des ersten Modells: Original | grau richtig, orange falsch, blau übersehen
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from . import compat, data, imageio, metrics, report

CLASS_COLORS = {0: (233, 236, 239), 1: (255, 255, 255), 2: (73, 80, 87), 3: (230, 159, 0), 4: (0, 114, 178)}
BINS = 1000


def _groups(record) -> list[str]:
    return ["gesamt"] + ([record.subset] if record.subset else [])


def bootstrap(values: np.ndarray, weights: np.ndarray | None, repeats: int, rng) -> tuple[float, float]:
    """95-%-Intervall des Mittelwerts (weights None) oder der Quote sum(values) / sum(weights)."""
    n = len(values)
    if n < 2 or repeats <= 0:
        return float("nan"), float("nan")
    index = rng.integers(n, size=(repeats, n))
    if weights is None:
        stats = values[index].mean(axis=1)
    else:
        stats = values[index].sum(axis=1) / np.maximum(weights[index].sum(axis=1), 1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize(images: pd.DataFrame, lambdas, aggregate: str, repeats: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    out = []

    def add(model, group, kind, metric, value, ci=(float("nan"), float("nan"))):
        out.append({"model": model, "group": group, "type": kind, "metric": metric, "value": value,
                    "ci_low": ci[0], "ci_high": ci[1]})

    for model in images["model"].unique():
        for group in sorted(set(g for gs in images["groups"] for g in gs), key=lambda g: (g != "gesamt", g)):
            sel = images[(images["model"] == model) & images["groups"].apply(lambda gs: group in gs)]
            if sel.empty:
                continue
            pores, negatives = sel[sel["gt_pore_px"] > 0], sel[sel["gt_pore_px"] == 0]
            for lam in lambdas:
                value = metrics.objective(sel[list(metrics.COUNT_KEYS)].to_dict("records"), lam, aggregate)
                add(model, group, "alle", f"Zielmaß λ={lam:g}", value["score"])
            if len(pores):
                tp, fp, fn = pores["tp_px"].sum(), pores["fp_px"].sum(), pores["fn_px"].sum()
                m, g_n, d_n = pores["matched_pores"].sum(), pores["gt_pores"].sum(), pores["det_pores"].sum()
                add(model, group, "Poren", "Bilder", len(pores))
                add(model, group, "Poren", "Pixel-F1 (Mittel je Bild)", pores["pixel_f1"].mean(),
                    bootstrap(pores["pixel_f1"].to_numpy(), None, repeats, rng))
                add(model, group, "Poren", "Pixel-Precision (gesamt)", metrics.ratio(tp, tp + fp))
                add(model, group, "Poren", "Pixel-Recall (gesamt)", metrics.ratio(tp, tp + fn))
                add(model, group, "Poren", "Pixel-F1 (gesamt)", metrics.ratio(2 * tp, 2 * tp + fp + fn))
                add(model, group, "Poren", "Objekt-F1 (Mittel je Bild)", pores["object_f1"].mean(),
                    bootstrap(pores["object_f1"].to_numpy(), None, repeats, rng))
                add(model, group, "Poren", "Objekt-Precision (gesamt)", metrics.ratio(m, d_n))
                add(model, group, "Poren", "Objekt-Recall (gesamt)", metrics.ratio(m, g_n))
                loose = (pores["object_recall_loose"].fillna(0) * pores["gt_pores"]).sum()
                add(model, group, "Poren", "Objekt-Recall IoU>=0.3 (gesamt)", metrics.ratio(loose, g_n))
                add(model, group, "Poren", "Porositätsfehler |pp| (Mittel)", pores["porosity_error_pp"].abs().mean(),
                    bootstrap(pores["porosity_error_pp"].abs().to_numpy(), None, repeats, rng))
                for key, name in (("gt_pores", "GT-Poren"), ("det_pores", "erkannte Poren"),
                                  ("missed_pores", "übersehene Poren"), ("phantom_pores", "Phantom-Poren"),
                                  ("split_pores", "zerteilte Poren"), ("merged_pores", "verschmolzene Poren")):
                    add(model, group, "Poren", name, int(pores[key].sum()))
            if len(negatives):
                ph = negatives["phantom_pores"].to_numpy(np.float64)
                px = negatives["image_px"].to_numpy(np.float64) / 1e6
                add(model, group, "negativ", "Bilder", len(negatives))
                add(model, group, "negativ", "Phantome / Mpx", ph.sum() / px.sum(), bootstrap(ph, px, repeats, rng))
                add(model, group, "negativ", "FP-Pixelanteil % (Mittel)", negatives["fp_px_share_percent"].mean())
                add(model, group, "negativ", "Bilder ohne Phantom", int(np.sum(ph == 0)))
    return pd.DataFrame(out)


def size_table(gt: pd.DataFrame, det: pd.DataFrame, pairs: pd.DataFrame, edges: list[float]) -> pd.DataFrame:
    labels = [f"{lo:g}-{hi:g} px" for lo, hi in zip(edges, edges[1:])] + [f">{edges[-1]:g} px"]
    bins = list(edges) + [np.inf]
    rows = []
    for (model, group), part in gt.groupby(["model", "group"]):
        d = det[(det["model"] == model) & (det["group"] == group)]
        p = pairs[(pairs["model"] == model) & (pairs["group"] == group)]
        gc = pd.cut(part["diameter_px"], bins, right=False, labels=labels)
        dc = pd.cut(d["diameter_px"], bins, right=False, labels=labels)
        pc = pd.cut(p["gt_diameter_px"], bins, right=False, labels=labels)
        for label in labels:
            g_sel, d_sel, p_sel = part[gc == label], d[dc == label], p[pc == label]
            rows.append({"model": model, "group": group, "diameter": label, "gt_pores": len(g_sel),
                         "recall": g_sel["matched"].mean() if len(g_sel) else np.nan,
                         "recall_iou0.3": g_sel["matched_loose"].mean() if len(g_sel) else np.nan,
                         "det_pores": len(d_sel), "precision": d_sel["matched"].mean() if len(d_sel) else np.nan,
                         "area_error_median_percent": 100 * p_sel["area_error"].median() if len(p_sel) else np.nan})
    return pd.DataFrame(rows)


def error_table(images: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["fp_pore_rim_px", "fp_matrix_px", "fn_pore_rim_px", "fn_interior_px"]
    for model in images["model"].unique():
        for group in sorted(set(g for gs in images["groups"] for g in gs), key=lambda g: (g != "gesamt", g)):
            sel = images[(images["model"] == model) & images["groups"].apply(lambda gs: group in gs)]
            sums = {k: int(sel[k].sum()) for k in keys}
            total = sum(sums.values())
            row = {"model": model, "group": group, **sums}
            row.update({f"{k}_percent": 100 * metrics.ratio(v, total) for k, v in sums.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def curves(hist: dict) -> pd.DataFrame:
    rows = []
    for (model, group), (pore, other) in hist.items():
        tp, fp = np.cumsum(pore[::-1])[::-1], np.cumsum(other[::-1])[::-1]
        total = pore.sum()
        for k in range(0, BINS + 1, 10):
            precision = metrics.ratio(tp[k], tp[k] + fp[k])
            recall = metrics.ratio(tp[k], total)
            rows.append({"model": model, "group": group, "threshold": k / BINS, "pixel_precision": precision,
                         "pixel_recall": recall, "pixel_f1": metrics.ratio(2 * tp[k], tp[k] + fp[k] + total)})
    return pd.DataFrame(rows)


def average_precision(pore: np.ndarray, other: np.ndarray) -> float:
    tp, fp = np.cumsum(pore[::-1]), np.cumsum(other[::-1])       # Schwelle von hoch nach tief
    total = pore.sum()
    if not total:
        return float("nan")
    recall = tp / total
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


# ---------------------------------------------------------------- Bilder

def _plots(folder: Path, images, sizes, curve_table, thresholds: dict) -> None:
    plt = report._plt()
    folder.mkdir(parents=True, exist_ok=True)
    total = curve_table[curve_table["group"] == "gesamt"]
    for name, x, y, xlabel, ylabel in (("pr_curve", "pixel_recall", "pixel_precision", "Recall", "Precision"),
                                       ("f1_threshold", "threshold", "pixel_f1", "Schwelle", "Pixel-F1")):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for i, (model, part) in enumerate(total.groupby("model", sort=False)):
            ax.plot(part[x], part[y], color=report.COLORS[i % 7], lw=2, label=model)
            if name == "f1_threshold":
                ax.axvline(thresholds[model], color=report.COLORS[i % 7], lw=1, ls="--")
        ax.set(xlabel=xlabel, ylabel=ylabel, title="Pixel, Bilder mit Poren (test)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=110)
        plt.close(fig)

    pores = images[images["gt_pore_px"] > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, (model, part) in enumerate(pores.groupby("model", sort=False)):
        ax.scatter(part["gt_porosity_percent"], part["det_porosity_percent"], s=14, color=report.COLORS[i % 7],
                   label=model, alpha=0.8)
    top = max(pores["gt_porosity_percent"].max(), pores["det_porosity_percent"].max(), 0.1) if len(pores) else 1
    ax.plot([0, top], [0, top], color="#999999", lw=1)
    ax.set(xlabel="Porosität GT [%]", ylabel="Porosität Modell [%]", title="Porosität je Bild (test)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "porosity.png", dpi=110)
    plt.close(fig)

    part = sizes[sizes["group"] == "gesamt"]
    if len(part):
        fig, ax = plt.subplots(figsize=(7, 4))
        models = list(part["model"].unique())
        labels = list(part["diameter"].unique())
        width = 0.8 / len(models)
        for i, model in enumerate(models):
            values = part[part["model"] == model].set_index("diameter").reindex(labels)["recall"]
            ax.bar(np.arange(len(labels)) + i * width, values, width, color=report.COLORS[i % 7], label=model)
        ax.set_xticks(np.arange(len(labels)) + 0.4 - width / 2, labels)
        ax.set(xlabel="Äquivalentdurchmesser", ylabel="Objekt-Recall", title="Gefundene Poren nach Größe (test)",
               ylim=(0, 1))
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / "recall_by_size.png", dpi=110)
        plt.close(fig)


def overlay(path: Path, original: np.ndarray, classes: np.ndarray, title: str, max_px: int) -> None:
    rgb = original if original.dtype == np.uint8 else (original // 257).astype(np.uint8)
    colored = np.zeros((*classes.shape, 3), np.uint8)
    for value, color in CLASS_COLORS.items():
        colored[classes == value] = color
    height, width = classes.shape
    factor = min(1.0, max_px / max(height, width))
    if factor < 1:
        size = (max(1, round(width * factor)), max(1, round(height * factor)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        colored = cv2.resize(colored, size, interpolation=cv2.INTER_NEAREST)
    gap = np.full((rgb.shape[0], 12, 3), 255, np.uint8)
    body = np.hstack([rgb, gap, colored])
    header = np.full((60, body.shape[1], 3), 255, np.uint8)
    for i, line in enumerate(title.split("\n")[:2]):
        cv2.putText(header, line, (8, 24 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    imageio.write_png(path, np.vstack([header, body]))


# ---------------------------------------------------------------- Hauptfunktion

def evaluate_models(run, models: list[tuple[str, Path]], out: Path) -> pd.DataFrame:
    s = run.settings
    cfg = s["evaluation"]
    out.mkdir(parents=True, exist_ok=True)
    if not run.test:
        run.console.log("Evaluation: keine Testbilder")
        return pd.DataFrame()
    rows, gts, dets, prs, hist, thresholds = [], [], [], [], {}, {}
    for label, folder in models:
        recipe, net = compat.load(folder, run.device)
        threshold = float(recipe["threshold"])
        thresholds[label] = threshold
        run.status = f"Evaluation · {label}"
        run.console.trial_start("Test", len(run.test))
        for i, record in enumerate(run.test, 1):
            image = np.array(data.load_image(record))
            gt = data.load_mask(record)
            prob = compat.predict(net, recipe, image, run.device, run.val_batch)
            values, gt_pores, det_pores, pairs = metrics.evaluate(prob >= threshold, gt, cfg)
            groups = _groups(record)
            rows.append({"model": label, "image": record.name, "subset": record.subset or "-",
                         "type": "negativ" if record.negative else "Poren", "groups": groups,
                         "threshold": threshold, **values})
            if not record.negative:
                bins = np.minimum((prob * BINS).astype(np.int64), BINS)
                pore = np.bincount(bins[gt], minlength=BINS + 1)
                other = np.bincount(bins[~gt], minlength=BINS + 1)
                for group in groups:
                    p, o = hist.get((label, group), (0, 0))
                    hist[(label, group)] = (p + pore, o + other)
            for group in groups:
                for table, target in ((gt_pores, gts), (det_pores, dets), (pairs, prs)):
                    if len(table):
                        target.append(table.assign(model=label, group=group, image=record.name))
            run.console.trial_update(i, f"{label}: {i}/{len(run.test)} Bilder")
            run.tick()
        del net
        if run.cuda:
            torch.cuda.empty_cache()

    images = pd.DataFrame(rows)
    empty_gt = pd.DataFrame(columns=["diameter_px", "matched", "matched_loose", "model", "group", "image"])
    empty_det = pd.DataFrame(columns=["diameter_px", "area_px", "matched", "real", "model", "group", "image"])
    empty_pairs = pd.DataFrame(columns=["gt_diameter_px", "area_error", "model", "group", "image"])
    gt_all = pd.concat(gts, ignore_index=True) if gts else empty_gt
    det_all = pd.concat(dets, ignore_index=True) if dets else empty_det
    pairs_all = pd.concat(prs, ignore_index=True) if prs else empty_pairs

    summary_long = summarize(images, run.lambdas, s["objective"]["aggregate"], int(cfg["bootstrap"]))
    for (label, group), (pore, other) in hist.items():
        summary_long = pd.concat([summary_long, pd.DataFrame([{
            "model": label, "group": group, "type": "Poren", "metric": "Pixel-AP (PR-Kurve)",
            "value": average_precision(pore, other), "ci_low": np.nan, "ci_high": np.nan}])], ignore_index=True)
    order = {m: i for i, m in enumerate(summary_long["metric"].unique())}
    summary = (summary_long.pivot_table(index=["group", "type", "metric"], columns="model", values="value",
                                        aggfunc="first", sort=False)
               .reset_index())
    summary["_g"] = summary["group"] != "gesamt"
    summary["_t"] = summary["type"].map({"alle": 0, "Poren": 1, "negativ": 2})
    summary["_m"] = summary["metric"].map(order)
    summary = summary.sort_values(["_g", "group", "_t", "_m"]).drop(columns=["_g", "_t", "_m"])
    summary = summary[["group", "type", "metric"] + [m for m, _ in models]]
    sizes = size_table(gt_all, det_all, pairs_all, cfg["size_classes_px"])
    errors = error_table(images)
    curve_table = curves(hist)
    images_out = images.drop(columns=["groups"])
    report.write_excel(out / "evaluation.xlsx", {"Summary": summary, "Summary CI": summary_long, "Images": images_out,
                                                 "Size classes": sizes, "Error origin": errors, "Curves": curve_table})
    _plots(out / "plots", images, sizes, curve_table, thresholds)
    _overlays(run, models[0], images, out / "overlays", cfg)

    for group in [g for g in summary["group"].unique()]:
        part = summary[summary["group"] == group].set_index("metric")
        lines = [f"Test [{group}]:"]
        for metric in ("Pixel-F1 (Mittel je Bild)", "Objekt-F1 (Mittel je Bild)", "Phantome / Mpx"):
            if metric in part.index:
                values = "   ".join(f"{m} {part.loc[metric, m]:.3f}" for m, _ in models
                                   if pd.notna(part.loc[metric, m]))
                lines.append(f"  {metric:<28} {values}")
        run.console.log("\n".join(lines))
    return summary


def _overlays(run, model: tuple[str, Path], images: pd.DataFrame, folder: Path, cfg: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    label, path = model
    mine = images[images["model"] == label]
    n = int(cfg["worst_images"])
    worst = [(r, "Objekt-F1") for r in mine[mine["gt_pore_px"] > 0].nsmallest(n, "object_f1")["image"]]
    worst += [(r, "Phantome") for r in mine[mine["gt_pore_px"] == 0].nlargest(n, "phantom_pores")["image"]]
    if not worst:
        return
    by_name = {r.name: r for r in run.test}
    recipe, net = compat.load(path, run.device)
    threshold = float(recipe["threshold"])
    table = mine.set_index("image")
    for rank, (name, reason) in enumerate(worst, 1):
        record = by_name[name]
        image = np.array(data.load_image(record))
        gt = data.load_mask(record)
        det = compat.predict(net, recipe, image, run.device, run.val_batch) >= threshold
        v = table.loc[name]
        title = (f"{name}   schlechtestes nach {reason} ({label})\n"
                 f"Pixel-F1 {v['pixel_f1']:.3f}  Objekt-F1 {v['object_f1']:.3f}  Phantome {int(v['phantom_pores'])}  "
                 f"Porosität GT {v['gt_porosity_percent']:.3f} %  Modell {v['det_porosity_percent']:.3f} %")
        safe = name.replace("/", "_")
        overlay(folder / f"{rank:02d}_{safe}.png", image, metrics.classes(det, gt), title, int(cfg["overlay_max_px"]))
    del net
    if run.cuda:
        torch.cuda.empty_cache()
