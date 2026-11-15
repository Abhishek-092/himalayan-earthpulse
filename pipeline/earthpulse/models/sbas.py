"""SBAS time-series inversion of an interferogram network (Berardino et al. 2002), per pixel, with
Tikhonov temporal smoothing so disconnected networks / gaps remain solvable, and full uncertainty propagation.

Unknowns: incremental displacements between consecutive acquisition dates (N-1).
Each pair (i, j) observes sum of increments i..j-1. Per-pixel weights (1/sigma^2, 0 for masked) are applied by
batched normal equations:  (A^T W A + lam^2 L^T L) x = A^T W d.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from earthpulse.config import NISAR_L_WAVELENGTH_M


def phase_to_los_m(phase_rad: np.ndarray, wavelength: float = NISAR_L_WAVELENGTH_M) -> np.ndarray:
    """Unwrapped phase -> LOS displacement (m). Positive = motion towards the satellite (NISAR/ISCE3 convention:
    d = -lambda/(4 pi) * phi)."""
    return -wavelength / (4.0 * np.pi) * phase_rad


def coherence_phase_sigma(coh: np.ndarray, looks: float) -> np.ndarray:
    """Cramer-Rao bound of interferometric phase std (rad) for multilooked coherence (Rodriguez & Martin 1992)."""
    c = np.clip(coh, 1e-3, 0.999)
    return np.sqrt((1 - c ** 2) / (2 * looks * c ** 2))


@dataclass
class SBASResult:
    dates: np.ndarray  # (N,) datetime64[D]
    displacement: np.ndarray  # (N, H, W) cumulative, first date = 0
    sigma: np.ndarray  # (N, H, W) 1-sigma of cumulative displacement
    residual_rms: np.ndarray  # (H, W) weighted network misclosure
    n_pairs_used: np.ndarray  # (H, W)


def design_matrix(pairs: list[tuple[int, int]], n_dates: int) -> np.ndarray:
    A = np.zeros((len(pairs), n_dates - 1))
    for r, (i, j) in enumerate(pairs):
        lo, hi = min(i, j), max(i, j)
        A[r, lo:hi] = 1.0 if j > i else -1.0
    return A


def invert(dates: np.ndarray, pairs: list[tuple[int, int]], disp: np.ndarray, sigma: np.ndarray,
           smooth: float = 0.02, block: int = 40000) -> SBASResult:
    """dates (N,), pairs [(ref_idx, sec_idx)], disp/sigma (P, H, W) pair displacements & 1-sigma (NaN = masked).

    smooth: regularisation strength relative to the typical pair weight; penalises the *second difference of
    velocity* (i.e. favours smooth motion) only where data are weak — it vanishes where the network is redundant.
    """
    P = len(pairs)
    N = len(dates)
    H, W = disp.shape[1:]
    A = design_matrix(pairs, N)
    days = np.diff(dates.astype("datetime64[D]").astype(float))
    # regulariser on velocity differences between consecutive intervals
    L = np.zeros((max(N - 2, 0), N - 1))
    for k in range(N - 2):
        L[k, k] = 1.0 / days[k]
        L[k, k + 1] = -1.0 / days[k + 1]
    L *= np.median(days)
    D = disp.reshape(P, -1).T  # (M, P)
    S = sigma.reshape(P, -1).T
    valid = np.isfinite(D) & np.isfinite(S) & (S > 0)
    Wt = np.where(valid, 1.0 / np.where(valid, S, 1.0) ** 2, 0.0)
    Df = np.where(valid, D, 0.0)
    M = D.shape[0]
    cum = np.full((M, N), np.nan)
    cum_sig = np.full((M, N), np.nan)
    rms = np.full(M, np.nan)
    C = np.tril(np.ones((N - 1, N - 1)))  # increments -> cumulative (dates 1..N-1)
    LtL = L.T @ L
    for s in range(0, M, block):
        e = min(s + block, M)
        w = Wt[s:e]
        typical = np.median(w[w > 0]) if np.any(w > 0) else 1.0
        lam2 = (smooth ** 2) * typical
        ata = np.einsum("pi,mp,pj->mij", A, w, A) + lam2 * LtL[None] + 1e-12 * typical * np.eye(N - 1)[None]
        atd = np.einsum("pi,mp,mp->mi", A, w, Df[s:e])
        cov = np.linalg.inv(ata)
        x = np.einsum("mij,mj->mi", cov, atd)
        ccov = np.einsum("ij,mjk,lk->mil", C, cov, C)
        cum[s:e, 1:] = x @ C.T
        cum[s:e, 0] = 0.0
        cum_sig[s:e, 1:] = np.sqrt(np.maximum(np.einsum("mii->mi", ccov), 0))
        cum_sig[s:e, 0] = 0.0
        pred = x @ A.T
        r = (Df[s:e] - pred) * np.sqrt(w)
        npair = valid[s:e].sum(1)
        rms[s:e] = np.where(npair > 0, np.sqrt((r ** 2).sum(1) / np.maximum(npair, 1)), np.nan)
    npairs = valid.sum(1)
    # pixels with too few pairs (fewer than intervals/2) are unreliable
    bad = npairs < max(2, (N - 1) // 2)
    cum[bad] = np.nan
    cum_sig[bad] = np.nan
    return SBASResult(
        dates=dates.astype("datetime64[D]"),
        displacement=cum.T.reshape(N, H, W),
        sigma=cum_sig.T.reshape(N, H, W),
        residual_rms=rms.reshape(H, W),
        n_pairs_used=npairs.reshape(H, W),
    )


def decompose_asc_desc(v_asc: np.ndarray, v_desc: np.ndarray, los_asc: np.ndarray, los_desc: np.ndarray,
                       s_asc: np.ndarray | None = None, s_desc: np.ndarray | None = None):
    """Two LOS geometries -> east-west and vertical components (north neglected: near-polar orbits are
    insensitive to N-S motion; stated as a limitation).

    los_*: (3, H, W) or (3,) ground-to-satellite unit vectors (east, north, up).
    Returns (east, up, sigma_east, sigma_up); NaN where either geometry is missing.
    """
    la = np.asarray(los_asc, float)
    ld = np.asarray(los_desc, float)
    if la.ndim == 1:
        la = la[:, None, None] * np.ones((1,) + v_asc.shape)
    if ld.ndim == 1:
        ld = ld[:, None, None] * np.ones((1,) + v_desc.shape)
    a11, a12 = la[0], la[2]
    a21, a22 = ld[0], ld[2]
    det = a11 * a22 - a12 * a21
    with np.errstate(invalid="ignore", divide="ignore"):
        east = (a22 * v_asc - a12 * v_desc) / det
        up = (-a21 * v_asc + a11 * v_desc) / det
        if s_asc is not None and s_desc is not None:
            se = np.sqrt((a22 * s_asc) ** 2 + (a12 * s_desc) ** 2) / np.abs(det)
            su = np.sqrt((a21 * s_asc) ** 2 + (a11 * s_desc) ** 2) / np.abs(det)
        else:
            se = su = np.full_like(east, np.nan)
    bad = np.abs(det) < 0.1
    for arr in (east, up, se, su):
        arr[bad] = np.nan
    return east, up, se, su


def remove_ramp(field: np.ndarray, exclude: np.ndarray | None = None, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fit & remove a planar (order 1) or quadratic (2) ramp using valid pixels outside `exclude`."""
    H, W = field.shape
    yy, xx = np.mgrid[0:H, 0:W]
    xx = (xx - W / 2) / W
    yy = (yy - H / 2) / H
    cols = [np.ones_like(xx), xx, yy]
    if order >= 2:
        cols += [xx * xx, yy * yy, xx * yy]
    G = np.stack([c.ravel() for c in cols], 1)
    use = np.isfinite(field).ravel()
    if exclude is not None:
        use &= ~exclude.ravel()
    if use.sum() < 10 * G.shape[1]:
        return field, np.zeros_like(field)
    coef, *_ = np.linalg.lstsq(G[use], field.ravel()[use], rcond=None)
    ramp = (G @ coef).reshape(H, W)
    return field - ramp, ramp
