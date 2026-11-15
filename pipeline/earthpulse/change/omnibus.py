"""Multitemporal SAR change detection with the complex-Wishart omnibus likelihood-ratio test.

Conradsen, Nielsen & Skriver (2016), "Determining the points of change in time series of polarimetric SAR data",
IEEE TGRS 54(5). Also Canty, Nielsen, Conradsen & Skriver (2020), Remote Sensing 12(1) (dual-pol diagonal case).

Why this test: unlike a fixed dB threshold, it uses the known speckle statistics (ENL) to turn every detected change
into a *probability* (p-value), handles any number of dates, and pinpoints *when* each change happened. That gives
BDM change maps with a built-in, statistically meaningful confidence layer.

Inputs are per-date covariance data in linear power:
  * ``diag``: (T, p, H, W) intensities of p polarisation channels (e.g. HH, HV) -> diagonal (dual-pol) model
  * ``full``: (T, p, p, H, W) complex covariance matrices (quad-pol GCOV with off-diagonal terms)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


def _logdet(c: np.ndarray, diagonal: bool) -> np.ndarray:
    """log|C| over leading (p) or (p,p) axes; C shaped (p, ...) for diagonal or (p, p, ...) for full."""
    if diagonal:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sum(np.log(c), axis=0)
    m = np.moveaxis(np.moveaxis(c, 0, -1), 0, -1)  # (..., p, p)
    sign, ld = np.linalg.slogdet(m)
    return np.where(sign > 0, ld, np.nan)


def _p_from_stat(z: np.ndarray, f: float, omega2: float) -> np.ndarray:
    """P(-2 rho lnQ <= z) under H0 with Box's second-order correction; returns 1 - that (= p-value)."""
    cdf = chi2.cdf(z, f) + omega2 * (chi2.cdf(z, f + 4) - chi2.cdf(z, f))
    return np.clip(1.0 - cdf, 0.0, 1.0)


def omnibus_pvalue(data: np.ndarray, enl: float, diagonal: bool = True) -> np.ndarray:
    """p-value of H0 'no change at all in the whole series' for each pixel."""
    k = data.shape[0]
    p = data.shape[1]
    n = float(enl)
    sum_ld = sum(_logdet(data[i], diagonal) for i in range(k))
    ld_mean = _logdet(data.mean(axis=0), diagonal)
    lnq = n * (sum_ld - k * ld_mean)
    if diagonal:
        # product of p independent single-pol tests (p=1 formulas, f and omega2 add over channels)
        f = p * (k - 1)
        rho = 1.0 - (k / n - 1.0 / (n * k)) / (6.0 * (k - 1))
        omega2 = p * (-(k - 1) / 4.0 * (1.0 - 1.0 / rho) ** 2)
    else:
        f = (k - 1) * p * p
        rho = 1.0 - (2 * p * p - 1) / (6.0 * (k - 1) * p) * (k / n - 1.0 / (n * k))
        omega2 = (p * p * (p * p - 1) / (24.0 * rho * rho) * (k / n ** 2 - 1.0 / (n * k) ** 2)
                  - p * p * (k - 1) / 4.0 * (1.0 - 1.0 / rho) ** 2)
    return _p_from_stat(-2.0 * rho * lnq, f, omega2)


def _rj_stat(sum_prev: np.ndarray, x_j: np.ndarray, j: int, n: float, p: int, diagonal: bool) -> np.ndarray:
    """ln R_j for testing X_j against the (assumed homogeneous) previous j-1 matrices. Inputs are *sums of looks*."""
    a_prev = sum_prev
    a_j = sum_prev + x_j
    lr = n * (p * (j * np.log(j) - (j - 1) * np.log(j - 1))
              + (j - 1) * _logdet(a_prev, diagonal) + _logdet(x_j, diagonal) - j * _logdet(a_j, diagonal))
    return lr


def _rj_pvalue(lnr: np.ndarray, j: int, n: float, p: int, diagonal: bool) -> np.ndarray:
    if diagonal:
        f = p
        rho = 1.0 - (1.0 + 1.0 / (j * (j - 1))) / (6.0 * n)
        omega2 = p * (-0.25 * (1.0 - 1.0 / rho) ** 2)
    else:
        f = p * p
        rho = 1.0 - (2 * p * p - 1) / (6.0 * p) * (1.0 + 1.0 / (j * (j - 1))) / n
        omega2 = (-p * p / 4.0 * (1.0 - 1.0 / rho) ** 2
                  + p * p * (p * p - 1) / (24.0 * rho * rho) * (1.0 + (2 * j - 1) / (j * j * (j - 1) ** 2)) / (n * n))
    return _p_from_stat(-2.0 * rho * lnr, f, omega2)


@dataclass
class ChangeResult:
    p_nochange: np.ndarray  # omnibus p-value (H0: no change in entire series)
    n_changes: np.ndarray  # number of detected change points
    first_change: np.ndarray  # index of first change date (-1 if none)
    last_change: np.ndarray  # index of last change date (-1 if none)
    change_matrix: np.ndarray  # (T, H, W) bool: change detected at date index t (vs. the preceding segment)
    direction: np.ndarray  # (T, H, W) int8: +1 brighter, -1 darker (in the reference channel), 0 no change
    min_pvalue: np.ndarray  # smallest R_j p-value along the series (strength of strongest change)


def sequential_change(data: np.ndarray, enl: float, alpha: float = 0.01, diagonal: bool = True,
                      ref_channel: int = -1) -> ChangeResult:
    """Detect change points per pixel with the sequential R_j test (Conradsen et al. 2016, Sec. III-B).

    Starting from the first date, each new date j is tested against the mean of the current homogeneous
    segment. If p(R_j) < alpha, a change is declared at j and a new segment starts at j.
    The omnibus p-value is returned too; pixels whose omnibus test is not significant are reported unchanged
    (this controls the family-wise false-alarm rate across the series).
    """
    data = np.asarray(data)
    k, p = data.shape[0], data.shape[1]
    n = float(enl)
    spatial = data.shape[2:] if diagonal else data.shape[3:]
    x = data * n  # sums of looks
    p_omni = omnibus_pvalue(data, enl, diagonal)
    significant = p_omni < alpha

    seg_sum = x[0].copy()
    seg_len = np.ones(spatial, dtype=np.int32)
    change = np.zeros((k,) + spatial, bool)
    direction = np.zeros((k,) + spatial, np.int8)
    min_p = np.ones(spatial)
    for j in range(1, k):
        jj = seg_len + 1  # position of the new date within the current segment (>=2)
        lnr = np.full(spatial, np.nan)
        pv = np.ones(spatial)
        for m in np.unique(jj):
            sel = jj == m
            if diagonal:
                lnr_m = _rj_stat(seg_sum[:, sel], x[j][:, sel], int(m), n, p, True)
            else:
                lnr_m = _rj_stat(seg_sum[:, :, sel], x[j][:, :, sel], int(m), n, p, False)
            lnr[sel] = lnr_m
            pv[sel] = _rj_pvalue(lnr_m, int(m), n, p, diagonal)
        pv = np.where(np.isfinite(pv), pv, 1.0)
        is_change = (pv < alpha) & significant
        min_p = np.minimum(min_p, np.where(significant, pv, 1.0))
        change[j] = is_change
        # direction from the reference channel intensity vs segment mean
        if diagonal:
            seg_mean_ref = seg_sum[ref_channel] / (seg_len * n)
            cur_ref = data[j, ref_channel]
        else:
            seg_mean_ref = np.real(seg_sum[ref_channel, ref_channel]) / (seg_len * n)
            cur_ref = np.real(data[j, ref_channel, ref_channel])
        direction[j] = np.where(is_change, np.sign(cur_ref - seg_mean_ref), 0).astype(np.int8)
        # update segments
        if diagonal:
            seg_sum = np.where(is_change[None], x[j], seg_sum + x[j])
        else:
            seg_sum = np.where(is_change[None, None], x[j], seg_sum + x[j])
        seg_len = np.where(is_change, 1, seg_len + 1)

    n_changes = change.sum(axis=0).astype(np.int16)
    first = np.where(n_changes > 0, np.argmax(change, axis=0), -1)
    last = np.where(n_changes > 0, (k - 1) - np.argmax(change[::-1], axis=0), -1)
    return ChangeResult(p_omni, n_changes, first, last, change, direction, min_p)
