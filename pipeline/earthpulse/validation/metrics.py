"""Validation metrics for continuous (displacement/velocity) and categorical (change maps) outputs."""
from __future__ import annotations

import numpy as np


def continuous(est: np.ndarray, ref: np.ndarray) -> dict[str, float | int]:
    m = np.isfinite(est) & np.isfinite(ref)
    e, r = est[m], ref[m]
    if e.size == 0:
        return {"n": 0}
    d = e - r
    corr = float(np.corrcoef(e, r)[0, 1]) if e.size > 2 and e.std() > 0 and r.std() > 0 else float("nan")
    return {"n": int(e.size), "mae": float(np.mean(np.abs(d))), "rmse": float(np.sqrt(np.mean(d ** 2))),
            "bias": float(np.mean(d)), "median_abs_error": float(np.median(np.abs(d))), "pearson_r": corr}


def categorical(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float | int]:
    m = np.ones(pred.shape, bool) if valid is None else valid.astype(bool)
    p, r = pred.astype(bool)[m], ref.astype(bool)[m]
    tp, fp = int((p & r).sum()), int((p & ~r).sum())
    fn, tn = int((~p & r).sum()), int((~p & ~r).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp and prec + rec) else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    n = tp + fp + fn + tn
    po = (tp + tn) / n if n else float("nan")
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2 if n else float("nan")
    kappa = (po - pe) / (1 - pe) if n and pe != 1 else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1, "iou": iou,
            "dice": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan"), "overall_accuracy": po,
            "kappa": kappa}


def area_and_boundary(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray, pixel_area_m2: np.ndarray,
                      pixel_size_m: float) -> dict[str, float | None]:
    """Area difference and boundary agreement between a detected change map and a reference map.
    Boundary error = distance from each detected boundary pixel to the nearest reference boundary (and vice versa)."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    p = pred.astype(bool) & valid
    r = ref.astype(bool) & valid
    a_p = float(np.sum(pixel_area_m2[p]) / 1e6)
    a_r = float(np.sum(pixel_area_m2[r]) / 1e6)
    out: dict[str, float | None] = {"area_detected_km2": round(a_p, 3), "area_reference_km2": round(a_r, 3),
                                    "area_difference_km2": round(a_p - a_r, 3),
                                    "area_difference_pct": round(100 * (a_p - a_r) / a_r, 1) if a_r > 0 else None}
    bp = p & ~binary_erosion(p)
    br = r & ~binary_erosion(r)
    if bp.any() and br.any():
        d_pr = distance_transform_edt(~br)[bp] * pixel_size_m
        d_rp = distance_transform_edt(~bp)[br] * pixel_size_m
        both = np.concatenate([d_pr, d_rp])
        out.update({"boundary_mean_distance_m": round(float(both.mean()), 1),
                    "boundary_median_distance_m": round(float(np.median(both)), 1),
                    "boundary_hausdorff95_m": round(float(np.percentile(both, 95)), 1)})
    return out
