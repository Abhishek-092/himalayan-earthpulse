"""Each algorithm must recover a known synthetic truth."""
import numpy as np
import pytest

from earthpulse.change.omnibus import omnibus_pvalue, sequential_change
from earthpulse.change.threshold import kittler_illingworth, log_ratio_db
from earthpulse.epitn.network import EventLayer, build_network, cooccurrence
from earthpulse.features.texture import glcm_features
from earthpulse.geo.area import pixel_area_m2
from earthpulse.models.sbas import decompose_asc_desc, invert, phase_to_los_m, remove_ramp
from earthpulse.preprocess.speckle import estimate_enl, quegan, refined_lee
from earthpulse.temporal.framework import BEHAVIOUR_LABELS, analyze, summarize_series
from earthpulse.tracking.ncc import track
from earthpulse.validation.metrics import categorical, continuous

RNG = np.random.default_rng(42)


def speckle(mean, looks, rng=RNG):
    return mean * rng.gamma(looks, 1.0 / looks, size=np.shape(mean))


# ---------------- speckle -------------------------------------------------------------------
def test_refined_lee_reduces_speckle_and_keeps_mean_and_edge():
    truth = np.full((80, 80), 0.05)
    truth[:, 40:] = 0.2
    img = speckle(truth, 4)  # GCOV is multilooked (ENL ~4-20); refined Lee has a known small bias at 1 look
    f = refined_lee(img, enl=4)
    for sl in (np.s_[5:75, 5:35], np.s_[5:75, 45:75]):
        assert abs(f[sl].mean() / truth[sl].mean() - 1) < 0.1
        assert f[sl].std() < 0.6 * img[sl].std()
    # edge stays sharp: columns right next to the edge keep their side's level
    assert f[10:70, 37].mean() < 0.11 and f[10:70, 42].mean() > 0.13


def test_quegan_and_enl():
    truth = np.full((10, 60, 60), 0.1)
    stack = speckle(truth, 4)
    f = quegan(stack, 7)
    assert f[:, 10:50, 10:50].std() < 0.6 * stack[:, 10:50, 10:50].std()
    enl = estimate_enl(stack[0])
    assert 2.5 < enl < 6.5


# ---------------- omnibus change test -------------------------------------------------------
def test_omnibus_false_alarm_rate_and_detection():
    T, H, W, n = 8, 100, 100, 16
    mean = np.ones((T, 2, H, W)) * np.array([0.2, 0.05])[None, :, None, None]
    # change: HV drops by 70% (-5.2 dB, typical canopy loss at L-band HV) from date 5 in the left block
    mean[5:, 1, :30, :30] *= 0.3
    data = speckle(mean, n)
    alpha = 0.01
    res = sequential_change(data, enl=n, alpha=alpha, diagonal=True)
    nochange = np.ones((H, W), bool)
    nochange[:30, :30] = False
    far = (res.n_changes[nochange] > 0).mean()
    assert far < 3 * alpha
    det = res.change_matrix[5, :30, :30].mean()
    assert det > 0.8
    assert (res.direction[5, :30, :30][res.change_matrix[5, :30, :30]] == -1).all()
    # omnibus p-values are ~uniform under H0
    p = omnibus_pvalue(data[:, :, 50:, 50:], n)
    assert abs((p < 0.05).mean() - 0.05) < 0.02


# ---------------- thresholds ----------------------------------------------------------------
def test_kittler_illingworth_separates_classes():
    v = np.concatenate([RNG.normal(0, 1, 9000), RNG.normal(-6, 1, 1000)])
    t = kittler_illingworth(v)
    assert -4.5 < t < -1.5
    assert np.allclose(log_ratio_db(np.array([1.0]), np.array([0.1])), -10)


# ---------------- texture -------------------------------------------------------------------
def test_glcm_smooth_vs_rough():
    img = np.zeros((60, 60))
    img[:, 30:] = RNG.normal(0, 5, (60, 30))
    img[:, :30] = RNG.normal(0, 0.05, (60, 30)) + np.linspace(-5, 5, 30)[None, :]
    f = glcm_features(img, window=7, levels=8)
    assert f["contrast"][10:50, 40:55].mean() > 3 * f["contrast"][10:50, 5:20].mean()
    assert f["entropy"][10:50, 40:55].mean() > f["entropy"][10:50, 5:20].mean()
    assert f["homogeneity"][10:50, 5:20].mean() > f["homogeneity"][10:50, 40:55].mean()
    assert np.nanmax(f["energy"]) <= 1.0 + 1e-9


# ---------------- temporal framework --------------------------------------------------------
def _dates(n, step=12):
    return np.datetime64("2025-11-17") + np.arange(n) * np.timedelta64(step, "D")


def test_temporal_rate_acceleration_season():
    d = _dates(61)  # 2 years: needed to separate acceleration from the annual cycle
    t = (d - d[0]).astype(float) / 365.25
    y = 3.0 + -40.0 * t + 0.5 * (-30.0) * t ** 2 + 4.0 * np.cos(2 * np.pi * (t + 0.88))
    noise = 0.5
    obs = y + RNG.normal(0, noise, y.size)
    res = analyze(d, obs[None], sigma=np.full((1, obs.size), noise))
    p = res.params
    assert res.meta["seasonal_fitted"]
    assert abs(p["acceleration_per_yr2"][0] - (-30.0)) < 5
    assert p["rate_per_yr"][0] < -40
    assert abs(p["seasonal_amplitude"][0] - 4.0) < 1.0
    assert BEHAVIOUR_LABELS[int(p["behaviour_class"][0])] == "accelerating"


def test_temporal_onset_persistence_recovery():
    d = _dates(20)
    y = np.zeros(20)
    y[6:12] = -5.0  # disturbance at index 6
    y[12:] = np.linspace(-4, -0.5, 8)  # recovery
    obs = y + RNG.normal(0, 0.3, 20)
    s = summarize_series(d, obs, sigma=np.full(20, 0.3))
    onset_expected = 1970 + (d[6].astype("datetime64[s]").astype(int) / 86400) / 365.25
    assert abs(s["onset_year"] - onset_expected) < 0.04
    assert s["recovered_fraction"] > 0.6
    assert s["persistence"] > 0.5


def test_temporal_stable_no_false_change():
    d = _dates(25)
    obs = RNG.normal(0, 1.0, (500, 25))
    res = analyze(d, obs, sigma=np.ones((500, 25)))
    cls = res.params["behaviour_class"]
    assert (cls == 0).mean() > 0.85
    assert np.isnan(res.params["onset_year"]).mean() > 0.8


# ---------------- SBAS ----------------------------------------------------------------------
def test_sbas_recovers_velocity_with_gap_and_redundancy():
    N, H, W = 15, 6, 6
    dates = _dates(N)
    t = (dates - dates[0]).astype(float) / 365.25
    v_true = -0.08  # m/yr (8 cm/yr subsidence)
    truth = v_true * t
    pairs = [(i, i + 1) for i in range(N - 1) if i != 7] + [(i, i + 2) for i in range(0, N - 2, 2)]
    sig = 0.004
    disp = np.stack([np.full((H, W), truth[j] - truth[i]) + RNG.normal(0, sig, (H, W)) for i, j in pairs])
    res = invert(dates, pairs, disp, np.full_like(disp, sig))
    est = res.displacement[-1]
    assert abs(est.mean() - truth[-1]) < 0.006
    assert np.all(np.abs(est - truth[-1]) < 3.5 * res.sigma[-1])
    assert np.all(res.sigma[-1] > 0) and np.all(res.sigma[-1] < 0.02)
    assert np.all(res.sigma[-1] >= res.sigma[1])


def test_sbas_masked_pairs():
    N = 8
    dates = _dates(N)
    pairs = [(i, i + 1) for i in range(N - 1)] + [(i, i + 2) for i in range(N - 2)]
    disp = np.stack([np.full((2, 2), 0.01 * (j - i)) for i, j in pairs])
    disp[2, 0, 0] = np.nan  # one masked pair - redundancy keeps it solvable
    res = invert(dates, pairs, disp, np.full_like(disp, 0.002))
    assert np.allclose(res.displacement[-1], 0.07, atol=0.005)


def test_phase_conversion_and_decomposition():
    # one fringe (2 pi) at L-band = lambda/2 ~ 11.9 cm
    assert abs(abs(phase_to_los_m(np.array(2 * np.pi))) - 0.1192) < 0.001
    east, up = 0.02, -0.05
    los_a = np.array([-0.62, -0.12, 0.77])
    los_d = np.array([0.62, -0.12, 0.77])
    va = np.array([[los_a[0] * east + los_a[2] * up]])
    vd = np.array([[los_d[0] * east + los_d[2] * up]])
    e, u, _, _ = decompose_asc_desc(va, vd, los_a, los_d)
    assert abs(e[0, 0] - east) < 1e-6 and abs(u[0, 0] - up) < 1e-6


def test_remove_ramp():
    yy, xx = np.mgrid[0:50, 0:50]
    signal = np.zeros((50, 50))
    signal[20:30, 20:30] = -1.0
    field = signal + 0.01 * xx - 0.02 * yy + 0.3
    excl = signal != 0
    out, _ = remove_ramp(field, exclude=excl)
    assert np.abs(out[~excl]).max() < 1e-6


# ---------------- feature tracking ----------------------------------------------------------
def test_ncc_recovers_subpixel_shift():
    from scipy.ndimage import gaussian_filter, shift

    base = gaussian_filter(RNG.normal(0, 1, (160, 160)), 1.5)
    true_dy, true_dx = 2.3, -1.6
    moved = shift(base, (true_dy, true_dx), order=3, mode="reflect")
    f = track(base, moved, chip=32, search=6, step=16)
    assert abs(np.nanmedian(f.dx) - true_dx) < 0.15
    assert abs(np.nanmedian(f.dy) - true_dy) < 0.15
    assert np.nanmedian(f.peak) > 0.8


# ---------------- EPITN ---------------------------------------------------------------------
def test_epitn_detects_association_not_random():
    H = W = 60
    a = np.zeros((H, W), bool)
    a[10:25, 10:25] = True
    b_assoc = np.zeros((H, W), bool)
    b_assoc[12:27, 12:27] = True
    b_far = np.zeros((H, W), bool)
    b_far[40:55, 40:55] = True
    onset_a = np.where(a, 2026.3, np.nan)
    onset_b = np.where(b_assoc, 2026.5, np.nan)
    la = EventLayer("A", "bdm", a, onset_a)
    r1 = cooccurrence(la, EventLayer("B", "gdm", b_assoc, onset_b), n_perm=199)
    r2 = cooccurrence(la, EventLayer("C", "cdm", b_far), n_perm=199)
    assert r1["significant"] and r1["p_value"] < 0.05
    assert not r2["significant"]
    assert r1["lag"]["median_days"] > 0
    net = build_network([la, EventLayer("B", "gdm", b_assoc, onset_b)], n_perm=99)
    assert net["edges"][0]["source"] == "A" and net["edges"][0]["directed"]


# ---------------- metrics & area ------------------------------------------------------------
def test_metrics_and_area():
    c = continuous(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert abs(c["mae"] - 1 / 3) < 1e-9
    m = categorical(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert m["tp"] == 1 and abs(m["iou"] - 1 / 3) < 1e-9
    x = 300000 + np.arange(10) * 20.0
    y = 3380000 - np.arange(10) * 20.0
    a = pixel_area_m2(32644, x, y)
    assert np.allclose(a, 400, rtol=0.003)
    g = pixel_area_m2(4326, np.array([79.0, 79.001]), np.array([30.0, 30.001]))
    assert 9000 < g.mean() < 11000


# ---------------- radar geometry, noise and phase filtering ---------------------------------------------
def test_layover_shadow_geometry():
    from earthpulse.preprocess.geometry import layover_shadow

    # east-west profile; radar to the east (LOS points east), incidence 35 deg
    x = np.arange(200) * 30.0
    elev = np.tile(np.where(x < 3000, 0.0, np.where(x < 3300, (x - 3000) * 2.0, 600.0)), (10, 1))  # steep west-facing wall
    elev2 = np.tile(np.where(x < 3000, 600.0, np.where(x < 3300, 600 - (x - 3000) * 2.0, 0.0)), (10, 1))  # east-facing drop
    ones = np.ones_like(elev)
    a = layover_shadow(elev, 30.0, -30.0, 0.6 * ones, 0 * ones, 35 * ones)
    b = layover_shadow(elev2, 30.0, -30.0, 0.6 * ones, 0 * ones, 35 * ones)
    # wall rising towards the radar faces AWAY from it (63 deg > 55 deg grazing) -> shadow; flat ground fine
    assert a["shadow"][:, 101:109].mean() > 0.9 and not a["layover"][:, 20:80].any()
    # slope descending towards the radar faces it at 63 deg > 35 deg incidence -> layover
    assert b["layover"][:, 101:109].mean() > 0.9
    # cast shadow behind the wall (west of it, away from the radar) is not expected; flat ground east is clean
    assert not a["shadow"][:, 150:190].any()


def test_noise_floor():
    from earthpulse.preprocess.noise import noise_correct, noise_floor_mask

    nesz = np.full(4, 0.003)
    i = np.array([0.004, 0.006, 0.05, np.nan])
    assert noise_floor_mask(i, nesz).tolist() == [True, False, False, False]
    c = noise_correct(i, nesz)
    assert abs(c[2] - 0.047) < 1e-9 and c[0] > 0 and np.isnan(c[3])


def test_goldstein_reduces_phase_noise():
    from earthpulse.preprocess.phase import goldstein

    yy, xx = np.mgrid[0:64, 0:64]
    truth = 0.15 * xx + 0.05 * yy
    noisy = truth + RNG.normal(0, 0.8, truth.shape)
    f = goldstein(np.exp(1j * noisy), alpha=0.8)
    err_before = np.angle(np.exp(1j * (noisy - truth)))
    err_after = np.angle(f * np.exp(-1j * truth))
    assert np.std(err_after[8:-8, 8:-8]) < 0.7 * np.std(err_before[8:-8, 8:-8])


def test_area_and_boundary():
    from earthpulse.validation.metrics import area_and_boundary
    pred = np.zeros((40, 40), bool); pred[10:20, 10:20] = True
    ref = np.zeros((40, 40), bool); ref[10:20, 12:22] = True  # same size, shifted 2 px
    out = area_and_boundary(pred, ref, np.ones((40, 40), bool), np.full((40, 40), 100.0), 10.0)
    assert out["area_difference_km2"] == 0 and out["area_detected_km2"] == 0.01
    assert 0 < out["boundary_median_distance_m"] <= 20


def test_nearest_index_resampling():
    from earthpulse.preprocess.coherence import _nearest_index
    src = np.arange(0, 800, 80.0)  # 80 m GUNW grid
    dst = np.array([-500.0, 0, 39, 41, 790, 2000])
    idx = _nearest_index(src, dst)
    assert list(idx) == [-1, 0, 0, 1, 9, -1]
