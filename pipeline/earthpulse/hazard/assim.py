"""DEHFE - Dynamic Earth Hazard Forecasting Engine: a linear Kalman filter that keeps a daily estimate of the ground
state between NISAR passes.

State x = [displacement d (m), velocity v (m/day)].
  d_t = d_{t-1} + v_{t-1} + beta * rain_t   (rain-driven extra motion)
  v_t = v_{t-1}                             (+ process noise q_v)
Observation (only on NISAR pass days): y = d + noise(sigma_obs).
Hyper-parameters (q_d, q_v, beta) are chosen by maximising the innovation log-likelihood.
Validation: one-step-ahead hindcast of every pass from data strictly before it, vs a persistence forecast.
"""
from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np


def run(days: np.ndarray, rain: np.ndarray, obs_days: np.ndarray, obs: np.ndarray, obs_sig: np.ndarray,
        q_d: float, q_v: float, beta: float, x0=None, P0=None, stop_after: np.datetime64 | None = None):
    """Filter over daily `days`. Returns mean (T,2), var (T,2), loglik, and the one-step predictions at obs days."""
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q_d ** 2, q_v ** 2])
    x = np.zeros(2) if x0 is None else np.array(x0, float)
    P = np.diag([1e-6, (0.02 / 365) ** 2]) if P0 is None else np.array(P0, float)  # prior velocity 0 +- 20 mm/yr
    idx = {d: i for i, d in enumerate(obs_days.astype("datetime64[D]"))}
    T = len(days)
    mean = np.zeros((T, 2))
    var = np.zeros((T, 2))
    ll = 0.0
    preds = []
    for t in range(T):
        x = F @ x + np.array([beta * np.nan_to_num(rain[t]), 0.0])
        P = F @ P @ F.T + Q
        d = days[t]
        if d in idx and (stop_after is None or d <= stop_after):
            k = idx[d]
            s2 = obs_sig[k] ** 2 + P[0, 0]
            innov = obs[k] - x[0]
            preds.append((d, x[0], np.sqrt(P[0, 0] + obs_sig[k] ** 2), obs[k]))
            ll += -0.5 * (np.log(2 * np.pi * s2) + innov ** 2 / s2)
            K = P[:, 0] / s2
            x = x + K * innov
            P = P - np.outer(K, P[0, :])
        mean[t] = x
        var[t] = np.diag(P)
    return mean, var, ll, preds


def fit(days, rain, obs_days, obs, obs_sig) -> dict[str, float]:
    best = None
    for q_d, q_v, beta in product([5e-5, 1e-4, 2e-4, 5e-4, 1e-3], [1e-8, 1e-7, 1e-6, 1e-5], [0.0, -1e-6, -5e-6, -2e-5]):
        _, _, ll, _ = run(days, rain, obs_days, obs, obs_sig, q_d, q_v, beta)
        if best is None or ll > best[0]:
            best = (ll, q_d, q_v, beta)
    return {"loglik": best[0], "q_d": best[1], "q_v": best[2], "beta_m_per_mm": best[3]}


def hindcast(days, rain, obs_days, obs, obs_sig, params=None) -> dict[str, Any]:
    """Predict each pass k (k >= 4) using only passes before it - including the hyper-parameters, which are re-fitted on
    passes 0..k-1 (rolling, no leakage). Compared with persistence (last observed value)."""
    rows = []
    for k in range(4, len(obs_days)):
        stop = obs_days[k - 1]
        sel_fit = days <= stop
        pk = fit(days[sel_fit], rain[sel_fit], obs_days[:k], obs[:k], obs_sig[:k])
        sel = days <= obs_days[k]
        mean, var, _, _ = run(days[sel], rain[sel], obs_days[:k], obs[:k], obs_sig[:k],
                              pk["q_d"], pk["q_v"], pk["beta_m_per_mm"], stop_after=stop)
        pred = mean[-1, 0]
        sd = np.sqrt(var[-1, 0] + obs_sig[k] ** 2)
        rows.append({"date": str(obs_days[k]), "obs": float(obs[k]), "kf": float(pred), "kf_sd": float(sd),
                     "persistence": float(obs[k - 1])})
    if not rows:
        return {"n": 0}
    e_kf = np.array([r["obs"] - r["kf"] for r in rows])
    e_p = np.array([r["obs"] - r["persistence"] for r in rows])
    z = np.array([abs(r["obs"] - r["kf"]) / r["kf_sd"] for r in rows])
    return {"n": len(rows), "rolling_refit": True, "rmse_kf_mm": float(np.sqrt(np.mean(e_kf ** 2)) * 1000),
            "rmse_persistence_mm": float(np.sqrt(np.mean(e_p ** 2)) * 1000),
            "coverage_68": float(np.mean(z <= 1.0)), "coverage_95": float(np.mean(z <= 1.96)), "rows": rows}
