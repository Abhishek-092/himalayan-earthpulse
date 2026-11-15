"""Fast sliding-window GLCM texture features (exact, vectorised).

Instead of computing a co-occurrence matrix per window (slow), each GLCM statistic is written as a local average
of a function of the quantised pixel pair (q(x), q(x+d)), so it reduces to box filters:

  contrast      = E[(i-j)^2]          homogeneity = E[1/(1+(i-j)^2)]
  dissimilarity = E[|i-j|]            energy      = sum_ij P_ij^2       entropy = -sum_ij P_ij log P_ij
  correlation   = cov(i,j)/(sd_i sd_j)

Energy/entropy need the local joint histogram: one box filter per (i,j) bin (L^2 filters; L=8 -> 64).
The GLCM is made symmetric and averaged over the four standard directions (0, 45, 90, 135 deg).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

_DIRECTIONS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))


def quantize(img: np.ndarray, levels: int = 8, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Quantise to integer levels using robust 2-98 percentile stretch (input usually in dB)."""
    v = img[np.isfinite(img)]
    lo = np.percentile(v, 2) if lo is None else lo
    hi = np.percentile(v, 98) if hi is None else hi
    q = np.floor((np.clip(img, lo, hi) - lo) / max(hi - lo, 1e-9) * levels)
    q = np.clip(q, 0, levels - 1)
    q[~np.isfinite(img)] = -1
    return q.astype(np.int16)


def glcm_features(img_db: np.ndarray, window: int = 7, levels: int = 8, distance: int = 1,
                  features: tuple[str, ...] = ("contrast", "homogeneity", "entropy", "energy", "dissimilarity",
                                               "correlation")) -> dict[str, np.ndarray]:
    q = quantize(img_db, levels)
    acc = {f: np.zeros(q.shape) for f in features}
    need_hist = any(f in features for f in ("entropy", "energy"))
    for dy, dx in _DIRECTIONS:
        dy, dx = dy * distance, dx * distance
        q2 = np.roll(np.roll(q, -dy, 0), -dx, 1)
        valid = (q >= 0) & (q2 >= 0)
        vf = valid.astype(float)
        cnt = ndimage.uniform_filter(vf, window) + 1e-12
        i = np.where(valid, q, 0).astype(float)
        j = np.where(valid, q2, 0).astype(float)

        def local_mean(x: np.ndarray) -> np.ndarray:
            return ndimage.uniform_filter(np.where(valid, x, 0.0), window) / cnt

        d = i - j
        if "contrast" in features:
            acc["contrast"] += local_mean(d * d)
        if "dissimilarity" in features:
            acc["dissimilarity"] += local_mean(np.abs(d))
        if "homogeneity" in features:
            acc["homogeneity"] += local_mean(1.0 / (1.0 + d * d))
        if "correlation" in features:
            # symmetric GLCM: marginals identical -> use pooled mean/var of i and j
            mi = 0.5 * (local_mean(i) + local_mean(j))
            vi = 0.5 * (local_mean(i * i) + local_mean(j * j)) - mi * mi
            cij = local_mean(i * j) - mi * mi
            with np.errstate(invalid="ignore", divide="ignore"):
                acc["correlation"] += np.where(vi > 1e-9, cij / vi, 1.0)
        if need_hist:
            ent = np.zeros(q.shape)
            eng = np.zeros(q.shape)
            # symmetric pair histogram: count unordered pairs (a,b) with a<=b once, doubling off-diagonals
            lo_ = np.minimum(i, j).astype(int)
            hi_ = np.maximum(i, j).astype(int)
            for a in range(levels):
                for b in range(a, levels):
                    ind = valid & (lo_ == a) & (hi_ == b)
                    if not ind.any():
                        continue
                    pab = ndimage.uniform_filter(ind.astype(float), window) / cnt
                    if a == b:
                        eng += pab * pab
                        ent -= np.where(pab > 0, pab * np.log(pab), 0.0)
                    else:
                        half = 0.5 * pab  # split between P_ab and P_ba
                        eng += 2 * half * half
                        ent -= 2 * np.where(half > 0, half * np.log(half), 0.0)
            if "entropy" in features:
                acc["entropy"] += ent
            if "energy" in features:
                acc["energy"] += eng
    out = {f: acc[f] / len(_DIRECTIONS) for f in features}
    nan = ~np.isfinite(img_db)
    for f in out:
        out[f][nan] = np.nan
    return out
