"""Shared Temporal Analysis Framework (used identically by BDM, GDM and CDM).

One engine computes the 15 temporal parameters from the concept document for ANY parameter time series
(backscatter, displacement, velocity, ...), per pixel or per region, with uncertainty:

  baseline, current, net_change, relative_change, rate (+stderr), acceleration (+stderr), seasonal amplitude/phase,
  seasonal_anomaly, persistence, recurrence, variability, onset, duration, recovery_rate, confidence,
  and a behaviour class (stable / continuous / accelerating / decelerating / seasonal / episodic / recovering).
  Spatial extent is computed by :mod:`earthpulse.geo` from the per-pixel change flags returned here.

Design rules (scientific honesty):
  * A change is "detected" only if it exceeds k x noise, where noise comes from supplied per-observation sigma
    or, if absent, from the robust scatter of the baseline period.
  * Seasonal terms are fitted only when the series spans >= ``min_seasonal_span_yr``; otherwise reported NaN.
  * Every output is NaN where data are insufficient - never silently filled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DAYS_PER_YEAR = 365.25


def to_decimal_years(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times)
    if np.issubdtype(t.dtype, np.datetime64):
        days = (t.astype("datetime64[s]").astype("int64")) / 86400.0
        return 1970.0 + days / DAYS_PER_YEAR
    return t.astype(float)


@dataclass
class TemporalConfig:
    baseline_n: int = 3  # first n valid observations form the baseline (if baseline_end not given)
    baseline_end: float | None = None  # decimal year; observations <= this are baseline
    current_n: int = 1  # mean of last n valid observations
    k_sigma: float = 2.0  # change detection threshold in noise units
    min_persist: int = 2  # consecutive flagged obs needed to declare onset
    min_seasonal_span_yr: float = 1.5  # <1.5 yr: annual cycle and acceleration are not separable
    min_obs: int = 4


@dataclass
class TemporalResult:
    params: dict[str, np.ndarray]
    flags: np.ndarray  # (..., T) bool: |y - baseline| > k sigma
    fitted: np.ndarray  # (..., T) trend + seasonal model
    meta: dict[str, Any] = field(default_factory=dict)


def _wls(design: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched weighted least squares. design (T,m); y,w (N,T). Returns coef (N,m), cov (N,m,m)."""
    aw = design[None, :, :] * w[:, :, None]  # (N,T,m)
    ata = np.einsum("ntm,tk->nmk", aw, design)
    aty = np.einsum("ntm,nt->nm", aw, y)
    ridge = 1e-9 * np.eye(design.shape[1])
    cov = np.linalg.pinv(ata + ridge)
    coef = np.einsum("nmk,nk->nm", cov, aty)
    return coef, cov


def mann_kendall(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Mann-Kendall trend test (no tie correction). y (N,T) with NaNs. Returns (S, two-sided p)."""
    from scipy.stats import norm

    order = np.argsort(t)
    y = y[:, order]
    n_t = y.shape[1]
    s = np.zeros(y.shape[0])
    for i in range(n_t - 1):
        d = y[:, i + 1:] - y[:, i:i + 1]
        s += np.nansum(np.sign(d), axis=1)
    n = np.sum(np.isfinite(y), axis=1).astype(float)
    var = n * (n - 1) * (2 * n + 5) / 18.0
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(s > 0, (s - 1) / np.sqrt(var), np.where(s < 0, (s + 1) / np.sqrt(var), 0.0))
    p = 2 * (1 - norm.cdf(np.abs(z)))
    p[n < 4] = np.nan
    return s, p


def analyze(times: np.ndarray, y: np.ndarray, sigma: np.ndarray | None = None,
            cfg: TemporalConfig | None = None, reference: np.ndarray | float | None = None) -> TemporalResult:
    """Compute the shared temporal parameters.

    times: (T,) datetime64 or decimal years. y: (..., T). sigma: same shape as y or broadcastable, optional.
    reference: optional reference state (e.g. long-term mean); defaults to the baseline mean.
    """
    cfg = cfg or TemporalConfig()
    t = to_decimal_years(times)
    order = np.argsort(t)
    t = t[order]
    y = np.asarray(y, float)[..., order]
    shape = y.shape[:-1]
    T = y.shape[-1]
    Y = y.reshape(-1, T)
    N = Y.shape[0]
    S = None
    if sigma is not None:
        S = np.broadcast_to(np.asarray(sigma, float)[..., order] if np.ndim(sigma) else sigma, y.shape).reshape(-1, T)
    if S is not None:
        # floor sigmas: a zero sigma (e.g. the reference date of a displacement series) would get infinite weight
        pos = np.where(np.isfinite(S) & (S > 0), S, np.nan)
        floor = 0.1 * np.nanmedian(pos, axis=1, keepdims=True)
        S = np.where(np.isfinite(S), np.maximum(S, np.nan_to_num(floor, nan=1e-9)), S)
    valid = np.isfinite(Y)
    nvalid = valid.sum(axis=1)
    tc = t - t.mean()
    span = t[-1] - t[0] if T > 1 else 0.0

    # --- baseline & current ----------------------------------------------------------------
    if cfg.baseline_end is not None:
        bmask = np.broadcast_to(t <= cfg.baseline_end, Y.shape) & valid
    else:
        rank = np.cumsum(valid, axis=1)
        bmask = valid & (rank <= cfg.baseline_n)
    with np.errstate(invalid="ignore"):
        baseline = np.where(bmask.sum(1) > 0, np.nansum(np.where(bmask, Y, 0), 1) / np.maximum(bmask.sum(1), 1), np.nan)
    rank_rev = np.cumsum(valid[:, ::-1], axis=1)[:, ::-1]
    cmask = valid & (rank_rev <= cfg.current_n)
    current = np.where(cmask.sum(1) > 0, np.nansum(np.where(cmask, Y, 0), 1) / np.maximum(cmask.sum(1), 1), np.nan)
    ref = baseline if reference is None else np.broadcast_to(np.asarray(reference, float), shape).reshape(-1)
    net = current - ref
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(np.abs(ref) > 1e-12, 100.0 * net / np.abs(ref), np.nan)

    # --- noise level ---------------------------------------------------------------------------
    if S is not None:
        noise = np.sqrt(np.nanmean(np.where(valid, S ** 2, np.nan), axis=1))
    else:
        dev = np.where(bmask, Y - baseline[:, None], np.nan)
        noise = 1.4826 * np.nanmedian(np.abs(dev - np.nanmedian(dev, 1, keepdims=True)), axis=1)
        # baseline too short: fall back to first-difference scatter of the whole series
        diffs = np.diff(np.where(valid, Y, np.nan), axis=1)
        fd = 1.4826 * np.nanmedian(np.abs(diffs - np.nanmedian(diffs, 1, keepdims=True)), axis=1) / np.sqrt(2)
        noise = np.where((bmask.sum(1) >= 3) & (noise > 0), noise, fd)
    noise = np.where(np.isfinite(noise) & (noise > 0), noise, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        net_sigma = noise * np.sqrt(1.0 / np.maximum(bmask.sum(1), 1) + 1.0 / max(cfg.current_n, 1))

    # --- trend / acceleration / seasonal fit ------------------------------------------------------
    use_season = span >= cfg.min_seasonal_span_yr
    cols = [np.ones(T), tc, 0.5 * tc ** 2]
    if use_season:
        cols += [np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)]
    A = np.stack(cols, axis=1)
    if S is not None:
        W = np.where(valid, 1.0 / np.maximum(S, 1e-12) ** 2, 0.0)
    else:
        W = valid.astype(float) / np.where(np.isfinite(noise), noise, 1.0)[:, None] ** 2
    Yf = np.where(valid, Y, 0.0)
    coef, cov = _wls(A, Yf, W)
    # linear-only model for rate (more robust than the quadratic's linear term at short spans)
    A_lin = np.stack([np.ones(T), tc] + (cols[3:] if use_season else []), axis=1)
    coef_l, cov_l = _wls(A_lin, Yf, W)
    enough = nvalid >= cfg.min_obs
    rate = np.where(enough, coef_l[:, 1], np.nan)
    rate_se = np.where(enough, np.sqrt(np.maximum(cov_l[:, 1, 1], 0)), np.nan)
    acc = np.where(nvalid >= cfg.min_obs + 1, coef[:, 2], np.nan)
    acc_se = np.where(nvalid >= cfg.min_obs + 1, np.sqrt(np.maximum(cov[:, 2, 2], 0)), np.nan)
    fitted = coef @ A.T
    if use_season:
        amp = np.hypot(coef[:, 3], coef[:, 4])
        phase_doy = (np.degrees(np.arctan2(coef[:, 4], coef[:, 3])) % 360) / 360 * DAYS_PER_YEAR  # day of peak
        seasonal_now = coef[:, 3] * np.cos(2 * np.pi * t[-1]) + coef[:, 4] * np.sin(2 * np.pi * t[-1])
        trend_only = coef[:, :3] @ A[:, :3].T
        last_idx = np.where(valid.any(1), T - 1 - np.argmax(valid[:, ::-1], axis=1), 0)
        resid_last = Y[np.arange(N), last_idx] - trend_only[np.arange(N), last_idx] - seasonal_now
        seasonal_anomaly = np.where(enough, resid_last, np.nan)
        amp = np.where(enough, amp, np.nan)
        phase_doy = np.where(enough, phase_doy, np.nan)
    else:
        amp = np.full(N, np.nan)
        phase_doy = np.full(N, np.nan)
        seasonal_anomaly = np.full(N, np.nan)

    resid = np.where(valid, Y - fitted, np.nan)
    variability = np.nanstd(np.where(valid, Y, np.nan), axis=1)
    resid_sd = np.nanstd(resid, axis=1)

    # --- change flags, onset, persistence, recurrence, duration --------------------------------
    with np.errstate(invalid="ignore"):
        flags = valid & (np.abs(Y - ref[:, None]) > cfg.k_sigma * noise[:, None])
    # onset = first index where `min_persist` consecutive valid obs are flagged
    run = np.zeros_like(Y)
    for i in range(T):
        prev = run[:, i - 1] if i > 0 else 0
        run[:, i] = np.where(flags[:, i], prev + 1, np.where(valid[:, i], 0, prev))
    persistent = run >= cfg.min_persist
    has_onset = persistent.any(axis=1)
    onset_idx = np.where(has_onset, np.argmax(persistent, axis=1) - (cfg.min_persist - 1), -1)
    onset_idx = np.maximum(onset_idx, -1)
    onset_t = np.where(has_onset, t[np.clip(onset_idx, 0, T - 1)], np.nan)
    last_flag_idx = np.where(flags.any(1), T - 1 - np.argmax(flags[:, ::-1], axis=1), -1)
    duration_days = np.where(has_onset, (t[np.clip(last_flag_idx, 0, T - 1)] - onset_t) * DAYS_PER_YEAR, np.nan)
    after = (np.arange(T)[None, :] >= onset_idx[:, None]) & valid & has_onset[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        persistence = np.where(has_onset, (flags & after).sum(1) / np.maximum(after.sum(1), 1), 0.0)
    # recurrence: number of separate flagged episodes (rising edges over valid obs)
    # count only persistent episodes (runs of >= min_persist flagged obs) so isolated noise spikes are ignored
    ends_run = persistent & ~np.concatenate([persistent[:, 1:], np.zeros((N, 1), bool)], axis=1)
    recurrence = ends_run.sum(axis=1)

    # --- recovery rate: after peak deviation, fraction of deviation recovered per year -------------
    dev = np.where(valid, Y - ref[:, None], np.nan)
    absdev = np.where(valid, np.abs(dev), -np.inf)
    pk = np.argmax(absdev, axis=1)
    peak_dev = dev[np.arange(N), pk]
    last_valid = np.where(valid.any(1), T - 1 - np.argmax(valid[:, ::-1], axis=1), 0)
    end_dev = dev[np.arange(N), last_valid]
    dt_rec = t[last_valid] - t[pk]
    with np.errstate(invalid="ignore", divide="ignore"):
        recovered_frac = np.where(np.abs(peak_dev) > 0, 1.0 - end_dev / peak_dev, np.nan)
        recovery_rate = np.where((dt_rec > 0) & has_onset, recovered_frac / dt_rec, np.nan)

    # --- Mann-Kendall significance -------------------------------------------------------------
    _, mk_p = mann_kendall(t, np.where(valid, Y, np.nan))

    # --- confidence (0..1): detection strength x data sufficiency --------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.abs(net) / net_sigma
    strength = 1.0 - np.exp(-0.5 * np.nan_to_num(snr) ** 2 / 4.0)  # ~0.4 at 2 sigma, ~0.86 at 4 sigma
    sufficiency = np.clip(nvalid / max(cfg.min_obs * 2, 1), 0, 1)
    confidence = np.where(np.isfinite(noise), strength * sufficiency, np.nan)

    # --- behaviour class ----------------------------------------------------------------------
    cls = np.zeros(N, np.int8)  # 0 stable
    sig_trend = np.isfinite(rate) & (np.abs(rate) > 2 * rate_se) & (mk_p < 0.05)
    sig_acc = np.isfinite(acc) & (np.abs(acc) > 2 * acc_se)
    seasonal_dom = np.isfinite(amp) & (amp > 2 * np.maximum(resid_sd, 1e-12)) & ~sig_trend
    strong_peak = np.abs(peak_dev) > (cfg.k_sigma + 1.0) * noise
    recovering = has_onset & strong_peak & np.isfinite(recovered_frac) & (recovered_frac > 0.5) & (dt_rec > 0)
    episodic = (recurrence >= 2) & ~sig_trend
    cls[sig_trend] = 1  # continuous
    cls[sig_trend & sig_acc & (np.sign(acc) == np.sign(rate))] = 2  # accelerating
    cls[sig_trend & sig_acc & (np.sign(acc) != np.sign(rate))] = 3  # decelerating / stabilising
    cls[seasonal_dom] = 4  # seasonal
    cls[episodic] = 5  # episodic / recurring
    cls[recovering & ~sig_trend] = 6  # recovering
    cls[nvalid < cfg.min_obs] = -1  # insufficient data

    params = {
        "baseline": baseline, "current": current, "net_change": net, "net_change_sigma": net_sigma,
        "relative_change_pct": rel, "rate_per_yr": rate, "rate_se": rate_se,
        "acceleration_per_yr2": acc, "acceleration_se": acc_se,
        "seasonal_amplitude": amp, "seasonal_peak_doy": phase_doy, "seasonal_anomaly": seasonal_anomaly,
        "variability_sd": variability, "residual_sd": resid_sd, "noise": noise,
        "onset_year": onset_t, "duration_days": duration_days, "persistence": persistence,
        "recurrence": recurrence.astype(float), "recovered_fraction": recovered_frac,
        "recovery_rate_per_yr": recovery_rate, "mann_kendall_p": mk_p, "confidence": confidence,
        "behaviour_class": cls.astype(float), "n_obs": nvalid.astype(float),
    }
    params = {k: v.reshape(shape) for k, v in params.items()}
    return TemporalResult(params=params, flags=flags.reshape(y.shape), fitted=fitted.reshape(y.shape),
                          meta={"times_decimal_year": t.tolist(), "seasonal_fitted": bool(use_season),
                                "span_years": float(span), "config": cfg.__dict__,
                                "note": None if use_season else "series shorter than seasonal threshold: acceleration may include unresolved seasonal signal"})


BEHAVIOUR_LABELS = {
    -1: "insufficient data", 0: "stable", 1: "continuous trend", 2: "accelerating", 3: "decelerating / stabilising",
    4: "seasonal", 5: "episodic / recurring", 6: "recovering",
}


def summarize_series(times: np.ndarray, y: np.ndarray, sigma: np.ndarray | None = None,
                     cfg: TemporalConfig | None = None) -> dict[str, Any]:
    """Run :func:`analyze` on a single series and return JSON-friendly scalars (for regional summaries)."""
    res = analyze(times, np.asarray(y, float)[None, :], None if sigma is None else np.asarray(sigma)[None, :], cfg)
    out: dict[str, Any] = {}
    for k, v in res.params.items():
        val = float(v[0])
        out[k] = None if not np.isfinite(val) else round(val, 6)
    out["behaviour"] = BEHAVIOUR_LABELS.get(int(res.params["behaviour_class"][0]), "unknown")
    out["seasonal_fitted"] = res.meta["seasonal_fitted"]
    out["span_years"] = round(res.meta["span_years"], 3)
    return out
