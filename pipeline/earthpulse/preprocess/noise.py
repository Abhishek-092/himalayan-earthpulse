"""Thermal-noise handling for GCOV backscatter.

The released GCOV products report noiseCorrectionApplied = False and provide the noise-equivalent backscatter (NESZ)
per polarisation on a coarse grid. We (1) interpolate NESZ to the data grid, (2) subtract it in linear power
(clipped at 10% of NESZ so values stay positive), and (3) flag pixels within 3 dB of the noise floor as unreliable -
critical for HV over dark, smooth or shadowed surfaces where the radar echo is mostly instrument noise.
Note: NESZ is given in the product's linear backscatter scale; the small sigma/gamma normalisation difference is ignored.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def nesz_to_grid(nesz: np.ndarray, nx: list[float], ny: list[float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    nx_a, ny_a = np.asarray(nx, float), np.asarray(ny, float)
    yo = np.argsort(ny_a)
    f = RegularGridInterpolator((ny_a[yo], nx_a), np.asarray(nesz, float)[yo, :], bounds_error=False, fill_value=None)
    X, Y = np.meshgrid(x, y)
    return f(np.stack([Y, X], -1)).astype(np.float32)


def noise_correct(intensity: np.ndarray, nesz: np.ndarray) -> np.ndarray:
    out = np.asarray(intensity, float) - nesz
    return np.where(np.isfinite(intensity), np.maximum(out, 0.1 * nesz), np.nan)


def noise_floor_mask(intensity: np.ndarray, nesz: np.ndarray, margin_db: float = 3.0) -> np.ndarray:
    """True where the measured intensity is within `margin_db` of the noise floor (signal not trustworthy)."""
    return np.isfinite(intensity) & (intensity < nesz * 10 ** (margin_db / 10))
