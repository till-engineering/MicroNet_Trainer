"""Metriken: Vergleich GT-Maske vs. Vorhersage eines Bilds – Definitionen wie Pore_Detection
(dataset_eval/metrics.py), aber nur das Modell: das ganze Bild ist Probe (keine ROI, kein Maßstab, keine Filter).

    Pixel      tp / fp / fn, Precision, Recall, F1 = 2tp / (2tp + fp + fn), IoU; FP am Porenrand oder in
               der Matrix, FN am Rand oder im Inneren (rim_px)
    Objekte    GT-Poren und erkannte Poren = Zusammenhangskomponenten (connectivity); Treffer ab IoU
               match_iou (>= 0.5: höchstens ein Partner je Pore); Objekt-F1 = 2 Treffer / (GT + erkannt);
               übersehen (keine Überlappung), Phantom (erkannt ohne Überlappung), zerteilt, verschmolzen
    Negativ    Bild ohne GT-Pore: jede erkannte Pore ist ein Phantom; Phantome je Megapixel

quick() liefert nur die Zählwerte für das Zielmaß (schnell, viele Schwellen); evaluate() alles.
"""

import math

import cv2
import numpy as np
import pandas as pd

COUNT_KEYS = ("image_px", "gt_pore_px", "tp_px", "fp_px", "fn_px", "gt_pores", "det_pores",
              "matched_pores", "phantom_pores")


def ratio(a, b) -> float:
    return a / b if b else float("nan")


def disk(radius_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))


def diameter(area_px) -> np.ndarray:
    return 2 * np.sqrt(np.asarray(area_px, dtype=np.float64) / math.pi)


def components(mask: np.ndarray, connectivity: int):
    """(Anzahl, Labelbild int32, Fläche je Label inkl. 0)."""
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity, ltype=cv2.CV_32S)
    return n - 1, labels, np.bincount(labels.ravel(), minlength=n)


def _pairs(gt_labels, gt_area, det_labels, det_area, n_det):
    """Paare (GT-Id, erkannte Id) mit gemeinsamen Pixeln und IoU."""
    both = (gt_labels > 0) & (det_labels > 0)
    base = n_det + 1
    code = gt_labels[both].astype(np.int64) * base + det_labels[both]
    pairs, shared = np.unique(code, return_counts=True)
    g, p = pairs // base, pairs % base
    return g, p, shared, shared / (gt_area[g] + det_area[p] - shared)


def quick(det: np.ndarray, gt_info, cfg: dict) -> dict:
    """Zählwerte für das Zielmaß. gt_info = components(gt) (einmal je Bild, für alle Schwellen)."""
    n_gt, gt_labels, gt_area = gt_info
    n_det, det_labels, det_area = components(det, cfg["connectivity"])
    gt_px, det_px = int(gt_area[1:].sum()), int(det_area[1:].sum())
    tp = matched = 0
    phantoms = n_det
    if n_gt and n_det:
        g, p, shared, iou = _pairs(gt_labels, gt_area, det_labels, det_area, n_det)
        tp = int(shared.sum())
        matched = int(np.count_nonzero(np.bincount(g[iou >= cfg["match_iou"]], minlength=n_gt + 1)[1:]))
        phantoms = n_det - len(np.unique(p))
    return {"image_px": int(det.size), "gt_pore_px": gt_px, "tp_px": tp, "fp_px": det_px - tp, "fn_px": gt_px - tp,
            "gt_pores": n_gt, "det_pores": n_det, "matched_pores": matched, "phantom_pores": phantoms}


def pixel_f1(row: dict) -> float:
    total = 2 * row["tp_px"] + row["fp_px"] + row["fn_px"]
    return 2 * row["tp_px"] / total if total else float("nan")


def object_f1(row: dict) -> float:
    total = row["gt_pores"] + row["det_pores"]
    return 2 * row["matched_pores"] / total if total else float("nan")


def objective(rows: list[dict], lam: float, aggregate: str) -> dict:
    """Zielmaß: 0.5 Pixel-F1 + 0.5 Objekt-F1 (Bilder mit Poren, Mittel oder 10-%-Quantil)
    - lam x Phantome je Mpx (Negativbilder)."""
    pores = [r for r in rows if r["gt_pore_px"] > 0]
    negatives = [r for r in rows if r["gt_pore_px"] == 0]

    def agg(values):
        if not values:
            return 0.0
        return float(np.mean(values)) if aggregate == "mean" else float(np.quantile(values, 0.1))

    pix, obj = agg([pixel_f1(r) for r in pores]), agg([object_f1(r) for r in pores])
    f1 = 0.5 * pix + 0.5 * obj
    area = sum(r["image_px"] for r in negatives)
    phantoms = sum(r["phantom_pores"] for r in negatives) / area * 1e6 if area else 0.0
    return {"score": f1 - lam * phantoms, "f1": f1, "pixel_f1": pix, "object_f1": obj, "phantoms_per_mpx": phantoms}


def evaluate(det: np.ndarray, gt: np.ndarray, cfg: dict):
    """Alle Werte eines Bilds: (Werte, GT-Poren, erkannte Poren, Paare) – wie Pore_Detection metrics.evaluate."""
    det, gt = det.astype(bool), gt.astype(bool)
    v: dict = {}
    image_px = det.size
    gt_px, det_px = int(np.count_nonzero(gt)), int(np.count_nonzero(det))
    v["image_px"] = image_px
    v["gt_pore_px"] = gt_px
    v["gt_porosity_percent"] = 100 * gt_px / image_px
    v["det_porosity_percent"] = 100 * det_px / image_px
    v["porosity_error_pp"] = v["det_porosity_percent"] - v["gt_porosity_percent"]
    v["porosity_error_rel_percent"] = 100 * ratio(v["porosity_error_pp"], v["gt_porosity_percent"])

    # --- Pixel
    tp = int(np.count_nonzero(det & gt))
    fp, fn = det_px - tp, gt_px - tp
    v.update(tp_px=tp, fp_px=fp, fn_px=fn, pixel_precision=ratio(tp, tp + fp), pixel_recall=ratio(tp, tp + fn),
             pixel_iou=ratio(tp, tp + fp + fn))
    v["pixel_f1"] = pixel_f1(v)
    v["fp_px_share_percent"] = 100 * fp / image_px
    rim = cfg["rim_px"]
    near_gt = cv2.dilate(gt.astype(np.uint8), disk(rim)) > 0
    v["fp_pore_rim_px"] = int(np.count_nonzero(det & ~gt & near_gt))
    v["fp_matrix_px"] = fp - v["fp_pore_rim_px"]
    gt_core = cv2.erode(gt.astype(np.uint8), disk(rim)) > 0
    v["fn_pore_rim_px"] = int(np.count_nonzero(gt & ~det & ~gt_core))
    v["fn_interior_px"] = fn - v["fn_pore_rim_px"]

    # --- Objekte
    n_gt, gt_labels, gt_area = components(gt, cfg["connectivity"])
    n_det, det_labels, det_area = components(det, cfg["connectivity"])
    matched_gt = np.zeros(n_gt + 1, bool)
    loose_gt = np.zeros(n_gt + 1, bool)
    matched_det = np.zeros(n_det + 1, bool)
    touched_gt = np.zeros(n_gt + 1, bool)
    touched_det = np.zeros(n_det + 1, bool)
    pieces = np.zeros(n_gt + 1, np.int64)
    merged_into = np.zeros(n_det + 1, np.int64)
    pair_gt = pair_area_error = np.zeros(0)
    if n_gt and n_det:
        g, p, shared, iou = _pairs(gt_labels, gt_area, det_labels, det_area, n_det)
        good = iou >= cfg["match_iou"]
        matched_gt[g[good]] = True
        matched_det[p[good]] = True
        loose_gt[g[iou >= cfg["small_match_iou"]]] = True
        touched_gt[g] = True
        touched_det[p] = True
        # Hauptpartner: je erkannter Pore die GT-Pore mit der größten Überlappung und umgekehrt
        order = np.argsort(shared)
        major_gt_of_det = np.zeros(n_det + 1, np.int64)
        major_gt_of_det[p[order]] = g[order]
        major_det_of_gt = np.zeros(n_gt + 1, np.int64)
        major_det_of_gt[g[order]] = p[order]
        cover_of_gt = np.zeros(n_gt + 1)
        np.maximum.at(cover_of_gt, g, shared / gt_area[g])
        pieces = np.bincount(major_gt_of_det[1:], minlength=n_gt + 1)
        pieces[0] = 0
        merged_into = np.bincount(major_det_of_gt[1:][cover_of_gt[1:] >= 0.5], minlength=n_det + 1)
        merged_into[0] = 0
        pair_gt = gt_area[g[good]]
        pair_area_error = det_area[p[good]] / gt_area[g[good]] - 1
    on_gt = np.bincount(det_labels[gt], minlength=n_det + 1)
    real = on_gt / np.maximum(det_area, 1) >= 0.5      # erkannte Pore liegt überwiegend auf GT-Pore

    matched = int(matched_gt[1:].sum())
    v.update(gt_pores=n_gt, det_pores=n_det, matched_pores=matched,
             object_precision=ratio(matched, n_det), object_recall=ratio(matched, n_gt),
             object_recall_loose=ratio(int(loose_gt[1:].sum()), n_gt),
             missed_pores=int(n_gt - touched_gt[1:].sum()),
             phantom_pores=int(n_det - touched_det[1:].sum()),
             split_pores=int(np.count_nonzero(pieces >= 2)),
             merged_pores=int(np.count_nonzero(merged_into >= 2)),
             false_objects=int(np.count_nonzero(~real[1:])))
    v["object_f1"] = object_f1(v)
    v["phantoms_per_mpx"] = v["phantom_pores"] / image_px * 1e6

    gt_pores = pd.DataFrame({"diameter_px": diameter(gt_area[1:]), "matched": matched_gt[1:],
                             "matched_loose": loose_gt[1:]})
    det_pores = pd.DataFrame({"diameter_px": diameter(det_area[1:]), "area_px": det_area[1:],
                              "matched": matched_det[1:], "real": real[1:]})
    pairs = pd.DataFrame({"gt_diameter_px": diameter(pair_gt), "area_error": pair_area_error})
    return v, gt_pores, det_pores, pairs


def classes(det: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Klasse je Pixel fürs Overlay: 1 Matrix, 2 Pore richtig, 3 falsch erkannt, 4 übersehen."""
    result = np.ones(gt.shape, np.uint8)
    result[det & gt] = 2
    result[det & ~gt] = 3
    result[gt & ~det] = 4
    return result
