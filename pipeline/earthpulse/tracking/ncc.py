"""Dense feature (offset) tracking by normalised cross-correlation with sub-pixel peak refinement.

Used by CDM on co-registered, same-grid NISAR GCOV amplitude images (same track/frame -> identical map grid),
so measured pixel offsets are directly map displacements (east, north) in metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import match_template


@dataclass
class OffsetField:
    rows: np.ndarray  # (R,) chip centre rows
    cols: np.ndarray  # (C,) chip centre cols
    dx: np.ndarray  # (R, C) offset in pixels (+ = towards increasing column)
    dy: np.ndarray  # (R, C) offset in pixels (+ = towards increasing row)
    peak: np.ndarray  # (R, C) NCC peak value
    snr: np.ndarray  # (R, C) peak / mean |NCC| of the surface


def _subpixel(c: np.ndarray, r: int, k: int) -> tuple[float, float]:
    """Gaussian 3-point peak fit (falls back to parabolic) on the correlation surface."""
    def fit(m1: float, m0: float, p1: float) -> float:
        if min(m1, m0, p1) > 0:
            l1, l0, lp = np.log(m1), np.log(m0), np.log(p1)
            den = 2 * (l1 - 2 * l0 + lp)
            if abs(den) > 1e-12:
                return (l1 - lp) / den
        den = 2 * (m1 - 2 * m0 + p1)
        return (m1 - p1) / den if abs(den) > 1e-12 else 0.0
    H, W = c.shape
    sy = fit(c[r - 1, k], c[r, k], c[r + 1, k]) if 0 < r < H - 1 else 0.0
    sx = fit(c[r, k - 1], c[r, k], c[r, k + 1]) if 0 < k < W - 1 else 0.0
    return float(np.clip(sy, -0.5, 0.5)), float(np.clip(sx, -0.5, 0.5))


def track(ref: np.ndarray, sec: np.ndarray, chip: int = 32, search: int = 8, step: int = 8,
          valid_mask: np.ndarray | None = None, min_valid: float = 0.8) -> OffsetField:
    """Track `ref` chips inside `sec` search windows. Images should be log-amplitude (dB), speckle filtered."""
    ref = np.asarray(ref, float)
    sec = np.asarray(sec, float)
    H, W = ref.shape
    half = chip // 2
    margin = half + search
    rows = np.arange(margin, H - margin, step)
    cols = np.arange(margin, W - margin, step)
    R, C = len(rows), len(cols)
    dx = np.full((R, C), np.nan)
    dy = np.full((R, C), np.nan)
    pk = np.full((R, C), np.nan)
    snr = np.full((R, C), np.nan)
    fill_r = np.nanmedian(ref)
    fill_s = np.nanmedian(sec)
    for a, r in enumerate(rows):
        for b, c in enumerate(cols):
            t = ref[r - half:r + half, c - half:c + half]
            s = sec[r - half - search:r + half + search, c - half - search:c + half + search]
            vt = np.isfinite(t)
            if vt.mean() < min_valid or np.isfinite(s).mean() < min_valid:
                continue
            if valid_mask is not None and not valid_mask[r, c]:
                continue
            t = np.where(vt, t, fill_r)
            s = np.where(np.isfinite(s), s, fill_s)
            if t.std() < 1e-6:
                continue
            cc = match_template(s, t)  # (2*search+1, 2*search+1)
            i, k = np.unravel_index(np.argmax(cc), cc.shape)
            peak = cc[i, k]
            if i in (0, cc.shape[0] - 1) or k in (0, cc.shape[1] - 1):
                continue  # peak at search edge -> unreliable
            sy, sx = _subpixel(cc, i, k)
            dy[a, b] = i - search + sy
            dx[a, b] = k - search + sx
            pk[a, b] = peak
            snr[a, b] = peak / (np.mean(np.abs(cc)) + 1e-9)
    return OffsetField(rows, cols, dx, dy, pk, snr)


def filter_offsets(f: OffsetField, min_peak: float = 0.15, min_snr: float = 2.0, median_size: int = 3,
                   max_dev_px: float = 1.5) -> OffsetField:
    """Quality filtering: correlation thresholds + removal of outliers vs local median (vector consistency)."""
    from scipy.ndimage import generic_filter

    good = (f.peak >= min_peak) & (f.snr >= min_snr)
    dx = np.where(good, f.dx, np.nan)
    dy = np.where(good, f.dy, np.nan)
    mdx = generic_filter(dx, np.nanmedian, size=median_size, mode="nearest")
    mdy = generic_filter(dy, np.nanmedian, size=median_size, mode="nearest")
    dev = np.hypot(dx - mdx, dy - mdy)
    keep = np.isfinite(dev) & (dev <= max_dev_px)
    return OffsetField(f.rows, f.cols, np.where(keep, dx, np.nan), np.where(keep, dy, np.nan),
                       np.where(keep, f.peak, np.nan), np.where(keep, f.snr, np.nan))
