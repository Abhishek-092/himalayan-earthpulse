"""SIPM - Slope Instability Prediction Model.

(1) Physics: infinite-slope factor of safety with Monte Carlo soil parameters -> P(FS < 1) per pixel and day.
    FS = [c' + (gamma*z - m*gamma_w*z) * cos^2(beta) * tan(phi')] / (gamma*z*sin(beta)*cos(beta))
    m (0..1) = relative wetness of the soil column, taken from observed/forecast ERA5-Land soil moisture (7-28 cm)
    rescaled between its dry (5th pct) and wet (99th pct) climatological values.
(2) Statistics: logistic regression on rainfall/soil-moisture percentiles predicting a hydro-meteorological hazard day
    (GloFAS discharge >= 2-year flood level within the next 3 days). Time-split validation (train <= 2010, test >= 2011)
    against climatology and a fixed rainfall threshold. Small and calibrated, so it is exported and run live in the browser.
(3) Combination: hazard index = P(FS<1) x relative trigger; window probability P = 1 - prod(1 - p_day).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GAMMA = 19.0  # soil unit weight kN/m3
GAMMA_W = 9.81


def factor_of_safety(slope_deg, c_kpa, phi_deg, z_m, m):
    b = np.radians(slope_deg)
    phi = np.radians(phi_deg)
    num = c_kpa + (GAMMA * z_m - m * GAMMA_W * z_m) * np.cos(b) ** 2 * np.tan(phi)
    den = GAMMA * z_m * np.sin(b) * np.cos(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 1e-9, num / den, np.inf)


@dataclass
class SoilPrior:
    c_kpa: tuple[float, float] = (3.0, 12.0)
    phi_deg: tuple[float, float] = (28.0, 38.0)
    z_m: tuple[float, float] = (1.0, 3.0)


def prob_failure(slope_deg: np.ndarray, wetness: float, n: int = 200, prior: SoilPrior = SoilPrior(), seed: int = 0,
                 c_add: np.ndarray | float = 0.0) -> np.ndarray:
    """P(FS < 1) per pixel for a given wetness, integrating over uncertain soil parameters."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(*prior.c_kpa, n)
    phi = rng.uniform(*prior.phi_deg, n)
    z = rng.uniform(*prior.z_m, n)
    s = np.asarray(slope_deg, float)
    flat = s.ravel()
    cadd = np.broadcast_to(np.asarray(c_add, float), s.shape).ravel()
    fails = np.zeros(flat.size)
    for i in range(n):
        fails += factor_of_safety(flat, c[i] + cadd, phi[i], z[i], wetness) < 1.0
    out = (fails / n).reshape(s.shape)
    out[~np.isfinite(s)] = np.nan
    return out


def wetness(sm: np.ndarray, sm_dry: float, sm_wet: float) -> np.ndarray:
    return np.clip((sm - sm_dry) / max(sm_wet - sm_dry, 1e-6), 0.0, 1.0)


# ------------------------------------------------------------------------ statistical trigger model
FEATURES = ("r3", "api", "sm", "melt", "t7")  # 3-day rain, antecedent index, soil moisture, snowmelt, 7-day temperature


def standardize(X: np.ndarray, mu=None, sd=None):
    mu = X.mean(0) if mu is None else mu
    sd = X.std(0) + 1e-9 if sd is None else sd
    return (X - mu) / sd, mu, sd


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 300) -> np.ndarray:
    """L2-regularised logistic regression via Newton-Raphson (no sklearn dependency at inference)."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    pos = max(y.mean(), 1e-6)
    sw = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))  # balanced class weights
    R = l2 * np.eye(Xb.shape[1])
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (sw * (p - y)) + R @ w
        H = (Xb * (sw * p * (1 - p))[:, None]).T @ Xb + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(w[0] + X @ w[1:])))


def calibrate_platt(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Map balanced-weight scores to real-frequency probabilities (Platt scaling on the logit)."""
    z = np.log(np.clip(score, 1e-9, 1 - 1e-9) / np.clip(1 - score, 1e-9, 1))
    A = np.column_stack([np.ones_like(z), z])
    w = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y)
        H = (A * (p * (1 - p))[:, None]).T @ A + 1e-6 * np.eye(2)
        st = np.linalg.solve(H, g)
        w -= st
        if np.abs(st).max() < 1e-9:
            break
    return float(w[0]), float(w[1])


def apply_platt(score: np.ndarray, a: float, b: float) -> np.ndarray:
    z = np.log(np.clip(score, 1e-9, 1 - 1e-9) / np.clip(1 - score, 1e-9, 1))
    return 1 / (1 + np.exp(-(a + b * z)))


def roc_auc(score: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(score)
    r = np.empty(len(score))
    r[o] = np.arange(1, len(score) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_curve(score: np.ndarray, y: np.ndarray, n: int = 50):
    th = np.quantile(score, np.linspace(0, 1, n))[::-1]
    tpr = [float(((score >= t) & (y == 1)).sum() / max(y.sum(), 1)) for t in th]
    fpr = [float(((score >= t) & (y == 0)).sum() / max((1 - y).sum(), 1)) for t in th]
    return fpr, tpr


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 8) -> list[dict[str, float]]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() >= 5:
            out.append({"p_mean": float(p[m].mean()), "obs_freq": float(y[m].mean()), "n": int(m.sum())})
    return out


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def window_prob(p_daily: np.ndarray, days: int) -> np.ndarray:
    """P(at least one hazard day in the next `days` days), assuming conditional independence between days."""
    out = np.zeros(len(p_daily))
    for i in range(len(p_daily)):
        seg = p_daily[i:i + days]
        out[i] = 1 - np.prod(1 - seg)
    return out


def summary_metrics(p: np.ndarray, y: np.ndarray, base_rate: float) -> dict[str, Any]:
    bs = brier(p, y)
    bs_clim = brier(np.full_like(p, base_rate), y)
    return {"n": int(len(y)), "events": int(y.sum()), "roc_auc": roc_auc(p, y), "brier": bs, "brier_climatology": bs_clim,
            "brier_skill_score": float(1 - bs / bs_clim) if bs_clim > 0 else None}


# ------------------------------------------------------------------ calibration done properly (no test leakage)
def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    e, n = 0.0, len(p)
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == bins - 1 else (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def calibration_line(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Logistic recalibration slope/intercept on the logit (slope 1, intercept 0 = perfectly calibrated)."""
    z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
    a, b = calibrate_platt(p, y)
    return {"intercept": a, "slope": b}


def fit_isotonic(score: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(score, y)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_isotonic(score: np.ndarray, table: dict[str, list[float]]) -> np.ndarray:
    return np.interp(score, table["x"], table["y"])


def _oof_scores(X: np.ndarray, y: np.ndarray, years: np.ndarray, l2: float) -> np.ndarray:
    """Leave-one-year-out (out-of-fold) raw scores inside the training period."""
    s = np.full(len(y), np.nan)
    for yy in np.unique(years):
        m = years == yy
        Xa, mu, sd = standardize(X[~m])
        w = fit_logistic(Xa, y[~m], l2=l2)
        s[m] = predict(w, (X[m] - mu) / sd)
    return s


def calibrated_model(X: np.ndarray, y: np.ndarray, years: np.ndarray, train_end: int, cal_end: int, l2: float = 5.0,
                     gbm: bool = True, mode: str = "oof") -> dict[str, Any]:
    """Calibrated logistic model with no test leakage. Test = years > cal_end (never touched).
    mode="oof"     : train on all years <= cal_end; calibrator fitted on leave-one-year-out predictions within them
    mode="holdout" : train on years <= train_end; calibrator fitted on years (train_end, cal_end]
    The deployed model is exactly the evaluated one."""
    te = years > cal_end
    if mode == "oof":
        tr = years <= cal_end
        ca = tr
    else:
        tr, ca = years <= train_end, (years > train_end) & (years <= cal_end)
    Xs_tr, mu, sd = standardize(X[tr])
    w = fit_logistic(Xs_tr, y[tr], l2=l2)
    raw = lambda m: predict(w, (X[m] - mu) / sd)
    s_ca = _oof_scores(X[tr], y[tr], years[tr], l2) if mode == "oof" else raw(ca)
    s_te = raw(te)
    pa, pb = calibrate_platt(s_ca, y[ca])
    iso = fit_isotonic(s_ca, y[ca])
    b_platt = brier(apply_platt(s_ca, pa, pb), y[ca])
    b_iso = brier(apply_isotonic(s_ca, iso), y[ca])
    method = "isotonic" if b_iso < b_platt else "platt"
    calib = lambda s: apply_isotonic(s, iso) if method == "isotonic" else apply_platt(s, pa, pb)
    p_te = calib(s_te)
    base = float(y[tr | ca].mean())
    # what in-sample Platt (the previous approach) gives on test, for the before/after comparison
    pa0, pb0 = calibrate_platt(predict(w, Xs_tr), y[tr])
    p_old = apply_platt(s_te, pa0, pb0)
    out: dict[str, Any] = {
        "mode": mode,
        "splits": {"train": f"<= {cal_end if mode == 'oof' else train_end}",
                   "calibrate": "leave-one-year-out within training years" if mode == "oof" else f"{train_end + 1}-{cal_end}",
                   "test": f"> {cal_end}",
                   "n_train": int(tr.sum()), "n_cal": int(ca.sum()), "n_test": int(te.sum()),
                   "events_train": int(y[tr].sum()), "events_cal": int(y[ca].sum()), "events_test": int(y[te].sum())},
        "calibrator": method, "brier_cal": {"platt": b_platt, "isotonic": b_iso},
        "test": {**summary_metrics(p_te, y[te], base), "roc_auc": roc_auc(s_te, y[te]),
                 "ece": ece(p_te, y[te]), "calibration": calibration_line(p_te, y[te]),
                 "mean_predicted": float(p_te.mean()), "observed_rate": float(y[te].mean())},
        "test_before_fix": {"brier_skill_score": summary_metrics(p_old, y[te], base)["brier_skill_score"], "ece": ece(p_old, y[te]),
                            "mean_predicted": float(p_old.mean())},
        "reliability": reliability(p_te, y[te], bins=8),
        "roc": dict(zip(("fpr", "tpr"), roc_curve(s_te, y[te]))),
        "note_auc": "ROC/AUC computed on raw scores (isotonic calibration creates ties that would understate ranking skill).",
        "deploy": {"weights": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "calibrator": method,
                   "platt": [pa, pb], "isotonic": iso},
        "p_test": p_te, "test_mask": te,
    }
    if gbm:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            mk = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                        l2_regularization=1.0, class_weight="balanced", random_state=0)
            g = mk().fit(X[tr], y[tr])
            if mode == "oof":
                gs_ca = np.full(int(tr.sum()), np.nan)
                ytr, yrs, Xtr = y[tr], years[tr], X[tr]
                for yy in np.unique(yrs):
                    mm = yrs == yy
                    gs_ca[mm] = mk().fit(Xtr[~mm], ytr[~mm]).predict_proba(Xtr[mm])[:, 1]
            else:
                gs_ca = g.predict_proba(X[ca])[:, 1]
            gs_te = g.predict_proba(X[te])[:, 1]
            giso = fit_isotonic(gs_ca, y[ca])
            gp = apply_isotonic(gs_te, giso)
            out["gradient_boosting"] = {**summary_metrics(gp, y[te], base), "roc_auc": roc_auc(gs_te, y[te]), "ece": ece(gp, y[te])}
        except Exception as e:  # pragma: no cover
            out["gradient_boosting"] = {"error": str(e)}
    return out
