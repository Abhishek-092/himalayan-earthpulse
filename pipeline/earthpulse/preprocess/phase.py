"""Goldstein-Werner adaptive interferometric phase filter (Goldstein & Werner 1998).

The NISAR GUNW products report wrappedInterferogramFiltering = 'no_filter', so we apply it ourselves to the wrapped
residual phase before re-unwrapping (models/gdm.py), as recommended for GDM in the project's preprocessing plan.
Overlapping patches are FFT-ed, the spectrum is weighted by its smoothed magnitude^alpha and transformed back; patches
are blended with a triangular window. Masked pixels carry zero amplitude so they do not leak into the spectrum.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def goldstein(ifg: np.ndarray, alpha: float = 0.5, patch: int = 16, step: int = 4) -> np.ndarray:
    z = np.nan_to_num(np.asarray(ifg, complex))
    H, W = z.shape
    ph, pw = H + patch, W + patch
    zp = np.zeros((ph, pw), complex)
    zp[:H, :W] = z
    out = np.zeros_like(zp)
    wsum = np.zeros((ph, pw))
    win1 = 1 - np.abs(np.linspace(-1, 1, patch))
    win = np.outer(win1, win1) + 1e-6
    for r in range(0, H, step):
        for c in range(0, W, step):
            blk = zp[r:r + patch, c:c + patch]
            if not np.any(blk):
                continue
            S = np.fft.fft2(blk)
            A = uniform_filter(np.abs(S), 3)
            A = A / (A.max() + 1e-12)
            filt = np.fft.ifft2(S * A ** alpha)
            out[r:r + patch, c:c + patch] += filt * win
            wsum[r:r + patch, c:c + patch] += win
    res = out[:H, :W] / np.maximum(wsum[:H, :W], 1e-12)
    return np.where(np.abs(z) > 0, res, 0)
