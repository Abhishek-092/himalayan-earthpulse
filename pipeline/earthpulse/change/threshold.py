"""Bi-temporal change indices and automatic thresholds."""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu


def log_ratio_db(i1: np.ndarray, i2: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """10*log10(I2/I1) on linear intensities (dB change; negative = darker on date 2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.maximum(i2, floor) / np.maximum(i1, floor))


def difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(b, float) - np.asarray(a, float)


def otsu(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    return float(threshold_otsu(v)) if v.size > 10 else float("nan")


def kittler_illingworth(values: np.ndarray, bins: int = 256) -> float:
    """Minimum-error thresholding (Kittler & Illingworth 1986) - robust for unequal class sizes in SAR change maps."""
    v = values[np.isfinite(values)]
    if v.size < 10:
        return float("nan")
    hist, edges = np.histogram(v, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h = hist.astype(float) / hist.sum()
    best_t, best_j = centers[bins // 2], np.inf
    cum_p = np.cumsum(h)
    cum_m = np.cumsum(h * centers)
    cum_s = np.cumsum(h * centers ** 2)
    tot_m, tot_s = cum_m[-1], cum_s[-1]
    for t in range(1, bins - 1):
        p1, p2 = cum_p[t], 1.0 - cum_p[t]
        if p1 < 1e-6 or p2 < 1e-6:
            continue
        m1 = cum_m[t] / p1
        m2 = (tot_m - cum_m[t]) / p2
        v1 = cum_s[t] / p1 - m1 * m1
        v2 = (tot_s - cum_s[t]) / p2 - m2 * m2
        if v1 <= 1e-12 or v2 <= 1e-12:
            continue
        j = 1 + 2 * (p1 * np.log(np.sqrt(v1)) + p2 * np.log(np.sqrt(v2))) - 2 * (p1 * np.log(p1) + p2 * np.log(p2))
        if j < best_j:
            best_j, best_t = j, centers[t]
    return float(best_t)


def robust_sigma_threshold(values: np.ndarray, k: float = 3.0) -> tuple[float, float]:
    """Median +/- k * 1.4826 * MAD (lower, upper) - a noise-calibrated symmetric threshold."""
    v = values[np.isfinite(values)]
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    return float(med - k * mad), float(med + k * mad)
