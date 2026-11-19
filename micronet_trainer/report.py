"""Tabellen und Bilder des Laufs: Verlauf je Trial, trials.xlsx, progress.png, Beispiel-Patches, Excel."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import imageio

COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999999"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    tmp = path.with_name(path.stem + "_tmp.xlsx")
    with pd.ExcelWriter(tmp, engine="xlsxwriter") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.sheets[name[:31]]
            for i, column in enumerate(table.columns):
                width = max([len(str(column))] + [len(f"{v:.4g}" if isinstance(v, float) else str(v))
                                                  for v in table[column].head(200)])
                sheet.set_column(i, i, min(max(width + 2, 8), 60))
            sheet.freeze_panes(1, 0)
    try:
        tmp.replace(path)
    except PermissionError:             # in Excel geöffnet: daneben schreiben
        tmp.replace(path.with_name(path.stem + "_neu.xlsx"))


# ---------------------------------------------------------------- je Trial

def trial_history(folder: Path, history: list[dict], lambdas: list[float]) -> None:
    table = pd.DataFrame(history)
    table.to_csv(folder / "history.csv", index=False)
    plt = _plt()
    fig, (ax_loss, ax_val) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax_loss.plot(table["epoch"], table["train_loss"], color=COLORS[0], lw=2)
    ax_loss.set(xlabel="Epoche", ylabel="Trainings-Loss", title="Loss (BCE + Dice)")
    for i, lam in enumerate(lambdas):
        key = f"score_lambda{lam:g}"
        if key in table:
            valid = table.dropna(subset=[key])
            ax_val.plot(valid["epoch"], valid[key], color=COLORS[i % len(COLORS)], lw=2, marker="o", ms=3,
                        label=f"Ziel λ={lam:g}")
    key = f"phantoms_per_mpx_lambda{lambdas[0]:g}"
    if key in table:
        ax2 = ax_val.twinx()
        valid = table.dropna(subset=[key])
        ax2.plot(valid["epoch"], valid[key], color="#999999", lw=1, ls="--", label="Phantome/Mpx")
        ax2.set_ylabel("Phantome / Mpx")
    ax_val.set(xlabel="Epoche", ylabel="Zielmaß (val)", title="Validierung")
    if ax_val.lines:
        ax_val.legend(loc="lower right", fontsize=8)
    for ax in (ax_loss, ax_val):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(folder / "history.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- Studie

def trials_table(study, lambdas: list[float]) -> pd.DataFrame:
    rows = []
    for trial in study.trials:
        a = trial.user_attrs
        row = {"trial": trial.number, "state": trial.state.name, "status": a.get("status", ""),
               "minutes": a.get("minutes"), "epochs": a.get("epochs"), "resumed_from": a.get("resume_from")}
        row.update({f"p_{k}": v for k, v in trial.params.items()})
        for lam in lambdas:
            tag = f"lambda{lam:g}"
            info = a.get(tag) or {}
            for key in ("score", "threshold", "epoch", "pixel_f1", "object_f1", "phantoms_per_mpx"):
                row[f"{key}_{tag}"] = info.get(key)
        row["error"] = a.get("error", "")
        rows.append(row)
    return pd.DataFrame(rows)


def progress_plot(path: Path, table: pd.DataFrame, lambdas: list[float]) -> None:
    key = f"score_lambda{lambdas[0]:g}"
    if table.empty or key not in table:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for state, color, label in (("COMPLETE", COLORS[0], "fertig"), ("PRUNED", "#BBBBBB", "abgebrochen (Pruning)")):
        part = table[(table["state"] == state) & table[key].notna()]
        ax.scatter(part["trial"], part[key], s=22, color=color, label=label, zorder=3)
    valid = table[table[key].notna()].sort_values("trial")
    if len(valid):
        ax.step(valid["trial"], valid[key].cummax(), where="post", color=COLORS[1], lw=2, label="bester bisher")
    ax.set(xlabel="Trial", ylabel=f"Zielmaß λ={lambdas[0]:g} (val)", title="Verlauf der Studie")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def importance(study) -> pd.DataFrame:
    try:
        import optuna
        values = optuna.importance.get_param_importances(study)
        return pd.DataFrame({"parameter": list(values), "importance": list(values.values())})
    except Exception:       # noqa: BLE001 – zu wenige Trials o. ä.
        return pd.DataFrame(columns=["parameter", "importance"])


# ---------------------------------------------------------------- Beispiel-Patches

def augment_examples(path: Path, stream, count: int = 12, columns: int = 4) -> None:
    """Patches, wie das Training sie sieht (zurückgerechnet aus der Normierung), Maskenrand magenta."""
    rng = np.random.default_rng(7)
    opened: dict = {}
    mean = np.asarray(stream.norm["mean"], np.float32)
    std = np.asarray(stream.norm["std"], np.float32)
    tiles = []
    for _ in range(count):
        image, mask = stream.patch(rng, opened)
        unit = image.numpy().transpose(1, 2, 0) * std + mean
        rgb = (np.clip(unit, 0, 1) * 255 + 0.5).astype(np.uint8)
        m = mask.numpy()[0].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, (255, 0, 255), 1)
        tiles.append(rgb)
    size = tiles[0].shape[0]
    rows = -(-len(tiles) // columns)
    sheet = np.full((rows * (size + 6) + 6, columns * (size + 6) + 6, 3), 215, np.uint8)
    for i, tile in enumerate(tiles):
        y, x = 6 + (i // columns) * (size + 6), 6 + (i % columns) * (size + 6)
        sheet[y:y + size, x:x + size] = tile
    imageio.write_png(path, sheet)
