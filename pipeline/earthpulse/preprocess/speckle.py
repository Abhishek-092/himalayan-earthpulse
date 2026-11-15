"""Speckle handling for detected (intensity) SAR imagery.

* ``lee``           - classic Lee MMSE filter (local statistics).
* ``refined_lee``   - Lee (1981) refined filter with 8 edge-aligned non-square windows (7x7), the standard
                      edge-preserving choice for forest boundaries (BDM) and crevasse texture (CDM).
* ``quegan``        - Quegan & Yu (2001) multitemporal filter: uses the whole time stack to suppress speckle
                      while preserving per-date radiometry (good for change detection on stacks).
* ``estimate_enl``  - equivalent number of looks from homogeneous areas (needed by the omnibus test).

All filters work on *linear power* (not dB) and preserve NaN as no-data.
Interferometric phase must NOT be filtered with these (phase filtering is a separate step in GDM).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _nan_uniform(img: np.ndarray, size: int | tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Local mean and valid fraction ignoring NaNs."""
    valid = np.isfinite(img)
    filled = np.where(valid, img, 0.0)
    s = ndimage.uniform_filter(filled, size=size, mode="reflect")
    c = ndimage.uniform_filter(valid.astype(float), size=size, mode="reflect")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan), c


def lee(img: np.ndarray, size: int = 5, enl: float = 1.0) -> np.ndarray:
    """Lee MMSE speckle filter (multiplicative noise model, noise variance coefficient 1/ENL)."""
    img = np.asarray(img, dtype=float)
    mean, _ = _nan_uniform(img, size)
    sq, _ = _nan_uniform(img * img, size)
    var = np.maximum(sq - mean * mean, 0.0)
    cu2 = 1.0 / enl
    with np.errstate(invalid="ignore", divide="ignore"):
        ci2 = var / (mean * mean)
        w = np.clip(1.0 - cu2 / ci2, 0.0, 1.0)
    w = np.nan_to_num(w)
    out = mean + w * (np.nan_to_num(img) - mean)
    out[~np.isfinite(img)] = np.nan
    return out


# Eight 7x7 directional masks for the refined Lee filter (edge-aligned half-windows).
def _refined_lee_masks() -> list[np.ndarray]:
    m = np.zeros((7, 7), bool)
    m[:, 3:] = True  # right half incl. centre column (vertical edge)
    diag = np.triu(np.ones((7, 7), bool))  # upper-right triangle incl. diagonal
    base = [m, diag]
    masks = []
    for b in base:
        for k in range(4):
            masks.append(np.rot90(b, k))
    # Order: E, N, W, S (vertical/horizontal half windows) then NE, NW, SW, SE (diagonal)
    return masks


_RL_MASKS = _refined_lee_masks()

# 3x3 gradient templates on 3x3 grid of 3x3-block means, one per direction (vertical, horizontal, two diags).
_GRAD = [
    np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float),  # vertical edge (E-W gradient)
    np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], float),  # horizontal edge (N-S gradient)
    np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], float),  # diagonal (NE)
    np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], float),  # anti-diagonal (NW)
]


def refined_lee(img: np.ndarray, enl: float = 1.0) -> np.ndarray:
    """Refined Lee filter (7x7) - selects, per pixel, the edge-aligned sub-window most homogeneous with the centre.

    Implementation:
      1. 3x3 block means on a 3x3 grid (spacing 2 px) around each pixel -> 4 directional gradients.
      2. Strongest gradient picks the edge orientation; the sign picks which side of the edge the centre lies on.
      3. Lee MMSE weight computed from statistics of the selected 7x7 half/diagonal window.
    """
    img = np.asarray(img, dtype=float)
    valid = np.isfinite(img)
    x = np.where(valid, img, np.nanmedian(img) if valid.any() else 0.0)

    # 3x3 block means (each centred two pixels apart -> covers 7x7)
    bm = ndimage.uniform_filter(x, size=3, mode="reflect")
    offs = (-2, 0, 2)
    blocks = np.stack([[np.roll(np.roll(bm, -dy, 0), -dx, 1) for dx in offs] for dy in offs])  # (3,3,H,W)
    grads = np.stack([np.abs(np.einsum("ij,ijhw->hw", g, blocks)) for g in _GRAD])  # (4,H,W)
    signs = np.stack([np.einsum("ij,ijhw->hw", g, blocks) for g in _GRAD])
    direction = np.argmax(grads, axis=0)  # 0..3
    sign = np.take_along_axis(signs, direction[None], 0)[0]

    # Choose one of 8 masks: for each orientation, the half whose mean is closest to the centre block.
    # Mask indexing in _RL_MASKS: rot90 k of base[0] -> 0..3 (E, N, W, S); base[1] -> 4..7.
    # orientation 0 (vertical edge): E (0) if sign>0 else W (2); 1 (horizontal): N (1) / S (3);
    # 2 (diag): NE-type (4) / SW-type (6); 3 (anti-diag): NW-type (5) / SE-type (7).
    lut_pos = np.array([0, 1, 4, 5])
    lut_neg = np.array([2, 3, 6, 7])
    choice = np.where(sign > 0, lut_pos[direction], lut_neg[direction])

    out = np.empty_like(x)
    cu2 = 1.0 / enl
    for k, mk in enumerate(_RL_MASKS):
        sel = choice == k
        if not sel.any():
            continue
        w = mk.astype(float) / mk.sum()
        mean = ndimage.correlate(x, w, mode="reflect")
        sq = ndimage.correlate(x * x, w, mode="reflect")
        var = np.maximum(sq - mean * mean, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            # MMSE weight, Lee (1981): b = var_x / var_y with var_x = (var_y - mean^2 cu2)/(1+cu2)
            var_x = (var - mean * mean * cu2) / (1.0 + cu2)
            b = np.clip(np.where(var > 0, var_x / var, 0.0), 0.0, 1.0)
        res = mean + b * (x - mean)
        out[sel] = res[sel]
    out[~valid] = np.nan
    return out


def quegan(stack: np.ndarray, size: int = 7) -> np.ndarray:
    """Quegan & Yu multitemporal speckle filter on a (T,H,W) linear-intensity stack.

    J_k = E[I_k]/T * sum_i I_i / E[I_i]  — local means E[.] by boxcar; preserves each date's local mean.
    """
    stack = np.asarray(stack, dtype=float)
    means = np.stack([_nan_uniform(s, size)[0] for s in stack])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = stack / means
    valid_count = np.sum(np.isfinite(ratio), axis=0)
    avg_ratio = np.nansum(ratio, axis=0) / np.maximum(valid_count, 1)
    out = means * avg_ratio[None]
    out[~np.isfinite(stack)] = np.nan
    return out


def estimate_enl(img: np.ndarray, window: int = 9, top_fraction: float = 0.05) -> float:
    """Estimate ENL = mean^2/var from the most homogeneous windows of a linear intensity image."""
    img = np.asarray(img, dtype=float)
    mean, cnt = _nan_uniform(img, window)
    sq, _ = _nan_uniform(img * img, window)
    var = np.maximum(sq - mean * mean, 1e-30)
    enl_map = mean * mean / var
    good = np.isfinite(enl_map) & (cnt > 0.99)
    if good.sum() < 10:
        return 1.0
    vals = enl_map[good]
    # Homogeneous areas have the highest ENL; use a high percentile but avoid extreme outliers.
    thr = np.quantile(vals, 1.0 - top_fraction)
    return float(np.median(vals[vals >= thr]))


def to_db(x: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10.0 * np.log10(np.maximum(x, floor))


def from_db(x: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(x) / 10.0)
