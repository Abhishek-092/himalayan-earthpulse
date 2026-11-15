"""Phase 2 models recover known answers on synthetic data."""
import numpy as np

from earthpulse.hazard import assim, flood, slope


def test_infinite_slope_textbook():
    # dry cohesionless slope: FS = tan(phi)/tan(beta)
    fs = slope.factor_of_safety(30.0, 0.0, 35.0, 2.0, 0.0)
    assert abs(fs - np.tan(np.radians(35)) / np.tan(np.radians(30))) < 1e-9
    # saturation lowers FS; cohesion raises it
    assert slope.factor_of_safety(30, 0, 35, 2, 1.0) < fs < slope.factor_of_safety(30, 5, 35, 2, 0.0)
    p = slope.prob_failure(np.array([10.0, 45.0]), 1.0)
    assert p[0] < 0.05 and p[1] > 0.5


def test_logistic_and_metrics():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 2))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(-2 + 1.5 * X[:, 0])))).astype(float)
    w = slope.fit_logistic(X, y, l2=1.0)
    s = slope.predict(w, X)
    a, b = slope.calibrate_platt(s, y)
    p = slope.apply_platt(s, a, b)
    assert abs(p.mean() - y.mean()) < 0.02  # calibrated to real frequency
    assert slope.roc_auc(p, y) > 0.75
    assert slope.summary_metrics(p, y, y.mean())["brier_skill_score"] > 0.05
    assert np.allclose(slope.window_prob(np.array([0.5, 0.5, 0.0]), 2)[:2], [0.75, 0.5])


def test_kalman_recovers_velocity_and_is_calibrated():
    rng = np.random.default_rng(1)
    days = np.arange(np.datetime64("2025-01-01"), np.datetime64("2026-01-01"))
    v = -0.03 / 365.25
    obs_days = days[::12]
    truth = v * (obs_days - days[0]).astype(float)
    sig = np.full(len(obs_days), 0.003)
    obs = truth + rng.normal(0, 0.003, len(obs_days))
    mean, var, _, _ = assim.run(days, np.zeros(len(days)), obs_days, obs, sig, 1e-4, 1e-7, 0.0)
    assert abs(mean[-1, 1] * 365.25 - (-0.03)) < 0.012
    hc = assim.hindcast(days, np.zeros(len(days)), obs_days, obs, sig, {"q_d": 1e-4, "q_v": 1e-7, "beta_m_per_mm": 0.0})
    assert 0.4 < hc["coverage_68"] <= 1.0


def test_gumbel_return_levels():
    from scipy.stats import gumbel_r

    m = gumbel_r.rvs(loc=200, scale=70, size=4000, random_state=2)
    rl = flood.return_levels(m)
    assert abs(rl["Q2"] - gumbel_r.ppf(0.5, 200, 70)) < 6
    assert abs(rl["Q20"] - gumbel_r.ppf(0.95, 200, 70)) < 15
    assert flood.level_for(rl["Q5"] + 1, rl) == 2


def test_landslide_helpers():
    from earthpulse.hazard import landslide as ls

    assert ls.cell_of(30.56, 79.56) == (30.5, 79.5)
    c = ls._contingency(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert c["pod"] == 0.5 and c["far"] == 0.5 and abs(c["csi"] - 1 / 3) < 1e-9


def test_calibration_toolkit():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < p).astype(float)
    assert slope.ece(p, y) < 0.02  # calibrated by construction
    assert slope.ece(p * 0.3, y) > 0.2  # badly under-predicting
    iso = slope.fit_isotonic(p * 0.3, y)
    q = slope.apply_isotonic(p * 0.3, iso)
    assert slope.ece(q, y) < 0.02
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)  # isotonic preserves ranking
    # three-way split: calibrator never sees test years
    X = rng.normal(size=(6000, 2))
    yy = (rng.random(6000) < 1 / (1 + np.exp(-(-3 + 1.5 * X[:, 0])))).astype(float)
    years = np.repeat(np.arange(2000, 2012), 500)
    r = slope.calibrated_model(X, yy, years, 2005, 2008, gbm=False)
    assert r["splits"]["n_test"] == 1500 and r["test"]["ece"] < 0.05
