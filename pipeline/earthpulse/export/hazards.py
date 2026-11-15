"""Phase 2 orchestration + export for one region: forcing -> SIPM (physics + statistics) -> DEHFE -> flood -> alerts.

Writes web/public/data/<region>/hazards/{manifest.json, model.json, pfs_*.png, *.bin}.
model.json holds everything the browser needs to recompute the trigger probability from a live Open-Meteo forecast.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from typing import Any

import numpy as np
from pyproj import Transformer

from earthpulse.config import WEB_DATA_DIR, WORK_DIR, Region
from earthpulse.export.web import LayerWriter, _clean
from earthpulse.forcing import events as ev
from earthpulse.forcing import openmeteo as om
from earthpulse.geo import ancillary
from earthpulse.hazard import alerts, assim, flood, slope

log = logging.getLogger(__name__)
TRAIN_END = np.datetime64("2010-12-31")


def _features(ix: dict[str, np.ndarray], sm: np.ndarray, tables: dict[str, list[float]]) -> np.ndarray:
    cols = []
    for f in slope.FEATURES:
        x = sm if f == "sm" else ix[f]
        cols.append(om.to_pct(x, tables[f]))
    return np.column_stack(cols)


def nisar_soil_moisture(region: Region, arc: dict[str, np.ndarray]) -> dict[str, Any] | None:
    """NISAR L3 SME2 soil moisture over the region, compared with ERA5-Land 0-7 cm on the same days."""
    from earthpulse.catalog.discover import parse_granule_name
    from earthpulse.catalog.stacks import region_granules
    from earthpulse.io.nisar import Subset, cache_path

    rows = []
    for g in region_granules(region):
        if g.product != "SME2":
            continue
        pth = cache_path(region.id, "SME2", g.granule_id)
        if not pth.exists():
            continue
        sub = Subset.load(pth)
        sm = sub.layers["soilMoisture"].astype(float)
        unc = sub.layers.get("soilMoistureUncertainty", np.zeros_like(sm)).astype(float)
        ok = np.isfinite(sm) & (sm >= 0) & (sm <= 0.7) & (unc < 0.1)
        if ok.mean() < 0.2:
            continue
        d = parse_granule_name(g.granule_id)["ref_time"][:10]
        rows.append((d, float(np.median(sm[ok])), float(np.median(unc[ok])), float(ok.mean()), g.direction, g.granule_id))
    if len(rows) < 5:
        return None
    rows.sort()
    era = dict(zip(arc["time"].astype(str), arc["soil_moisture_0_to_7cm_mean"]))
    nis = np.array([r[1] for r in rows])
    e5 = np.array([era.get(r[0], np.nan) for r in rows])
    m = np.isfinite(e5)
    from earthpulse.validation.metrics import continuous
    cmp = continuous(nis[m], e5[m]) if m.sum() >= 5 else {"n": int(m.sum())}
    return {"dates": [r[0] for r in rows], "nisar": [round(r[1], 4) for r in rows], "uncertainty": [round(r[2], 4) for r in rows],
            "valid_fraction": [round(r[3], 3) for r in rows], "direction": [r[4] for r in rows], "granules": [r[5] for r in rows],
            "era5_0_7cm": [None if not np.isfinite(v) else round(float(v), 4) for v in e5], "comparison": cmp,
            "note": "NISAR SME2 (L3, provisional, 200 m EASE-Grid) median over the region vs ERA5-Land top layer (~9 km). "
                    "Agreement in timing supports using ERA5-Land/forecast soil moisture as the slope-wetness forcing; "
                    "NISAR observes the state on pass days."}


def landslide_forecast(region: Region, lat: float, lon: float, fc, ixf, smf) -> dict[str, Any] | None:
    """Apply the GLC-trained landslide model (if trained) to the region's own 0.25 deg cell."""
    from earthpulse.config import DATA_DIR
    from earthpulse.hazard import landslide as ls

    mp = WORK_DIR / "landslide_model.json"
    glc = DATA_DIR / "external" / "glc.csv"
    if not mp.exists():
        return None
    res = json.loads(mp.read_text())
    dep = res["deploy"]
    cell = ls.cell_of(lat, lon)
    # percentile tables and alarm thresholds from the SAME point series the model is applied to (forecast + replay),
    # so 'top 1% of days' really means 1% of days at this site
    arc_c, _ = om.archive(lat, lon, start="1998-01-01", end="2017-12-31",
                          daily="precipitation_sum,temperature_2m_mean,soil_moisture_7_to_28cm_mean")
    ixc = om.indices(arc_c["precipitation_sum"], None, arc_c["temperature_2m_mean"])
    featc = {"sm": arc_c["soil_moisture_7_to_28cm_mean"], **{k: ixc[k] for k in ("r1", "r3", "r7", "api", "t7")}}
    tables = {k: om.ecdf_table(featc[k]) for k in dep["features"]}
    def prob(feats):
        X = np.column_stack([om.to_pct(feats[k], tables[k]) for k in dep["features"]])
        sc = slope.predict(np.array(dep["weights"]), (X - np.array(dep["mu"])) / np.array(dep["sd"]))
        return slope.apply_isotonic(sc, dep["isotonic"]) if dep.get("calibrator") == "isotonic" else slope.apply_platt(sc, *dep["platt"])
    # alarm thresholds at the same frequency as LHASA's 90/95/99th percentiles (fair comparison, equal alarm cost)
    def _cal(sc):
        return slope.apply_isotonic(sc, dep["isotonic"]) if dep.get("calibrator") == "isotonic" else slope.apply_platt(sc, *dep["platt"])
    p_clim = prob(featc)
    p_clim = p_clim[np.isfinite(p_clim)]
    thr = [float(np.quantile(p_clim, q)) for q in (0.90, 0.95, 0.99)]
    featf = {"sm": smf, **{k: ixf[k] for k in ("r1", "r3", "r7", "api", "t7")}}
    pf = prob(featf)
    levels = [int(sum(v >= t for t in thr)) for v in pf]
    near = []
    if glc.exists():
        for e in ls.load_events(glc):
            if abs(e["lat"] - lat) < 0.6 and abs(e["lon"] - lon) < 0.6:
                near.append(e)
    base = float(np.mean(p_clim))
    return {"validation": {k: v for k, v in res.items() if k != "deploy"}, "cell": list(cell), "thresholds": thr,
            "climatological_mean": base, "dates": [str(t) for t in fc["time"]], "p_daily": np.round(pf, 5).tolist(),
            "p72": slope.window_prob(pf, 3).tolist(), "levels": levels,
            "events_nearby": {"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {k: e[k] for k in ("date", "title", "trigger", "size", "fatalities", "source")},
                 "geometry": {"type": "Point", "coordinates": [e["lon"], e["lat"]]}} for e in near]},
            "browser_model": {**dep, "tables": tables, "thresholds": thr}}


def run_hazards(region: Region) -> dict[str, Any]:
    lon, lat = region.geometry.centroid.x, region.geometry.centroid.y
    today = np.datetime64(datetime.utcnow().date())
    # ---------------------------------------------------------------- 1. forcing
    arc, arc_meta = om.archive(lat, lon)
    rlat, rlon, _ = om.best_river_cell(lat, lon)
    q_hist, q_fc, q_meta = om.flood(rlat, rlon)
    q_meta["note"] = ("Main-stem GloFAS cell chosen automatically (largest monsoon flow within 0.1 deg). GloFAS (~5 km) "
                      "smooths steep Himalayan flash floods, so absolute peaks are underestimated; thresholds are "
                      "relative to its own climatology.")
    ix = om.indices(arc["precipitation_sum"], arc["snowfall_sum"], arc["temperature_2m_mean"])
    sm = arc["soil_moisture_7_to_28cm_mean"]
    tables = {f: om.ecdf_table(sm if f == "sm" else ix[f]) for f in slope.FEATURES}
    r7c = ix["r7"][np.isfinite(ix["r7"])]
    lhasa = {"p90": float(np.percentile(r7c, 90)), "p95": float(np.percentile(r7c, 95)), "p99": float(np.percentile(r7c, 99))}
    # ---------------------------------------------------------------- 2. flood return levels + labels
    brk = flood.detect_break(q_hist["time"], q_hist["river_discharge"])
    qyr = q_hist["time"].astype("datetime64[Y]").astype(int) + 1970
    by = brk["break_year"] if brk.get("significant") else None
    yrs, amax, amax_dates = flood.annual_maxima(q_hist["time"], q_hist["river_discharge"])
    pre = yrs < by if by else np.ones(len(yrs), bool)
    rl = flood.return_levels(amax[pre])  # consistent historical record (labels, validation, event checks)
    rl_live = flood.return_levels(amax[~pre]) if by and (~pre).sum() >= 4 else rl  # same product as live forecasts
    qmap = dict(zip(q_hist["time"], q_hist["river_discharge"]))
    q_al = np.array([qmap.get(t, np.nan) for t in arc["time"]])
    fut = np.full(len(q_al), np.nan)
    for k in range(4):  # max discharge over today .. +3 days
        sh = np.concatenate([q_al[k:], np.full(k, np.nan)])
        fut = np.fmax(fut, sh)
    y = (fut >= rl["Q2"]).astype(float)
    ayr = arc["time"].astype("datetime64[Y]").astype(int) + 1970
    ok = np.isfinite(fut) & np.isfinite(sm) & ((ayr < by) if by else True)  # exclude the post-splice record from training/testing
    X = _features(ix, sm, tables)
    # ---------------------------------------------------------------- 3. river high-flow model: calibrated, no leakage
    years = arc["time"].astype("datetime64[Y]").astype(int) + 1970
    CAL_END = 2012
    cm = slope.calibrated_model(X[ok], y[ok], years[ok], 2005, CAL_END, l2=10.0, gbm=True, mode="oof")
    te_ok = cm.pop("test_mask")
    cm.pop("p_test")
    dep = cm["deploy"]
    te = np.zeros(len(y), bool)
    te[np.nonzero(ok)[0][te_ok]] = True
    thr_r3 = 50.0
    base_thr = (ix["r3"][te] >= thr_r3).astype(float)
    base = float(y[ok & (years <= CAL_END)].mean())
    val = {
        "label": "GloFAS discharge >= 2-year flood level (Q2) within the next 3 days",
        "train": f"{int(years[ok].min())}-{CAL_END} (calibrator: leave-one-year-out within training years)",
        "test": f"{CAL_END + 1}-{int(years[ok].max())} (never used for fitting or calibration)",
        "logistic": cm["test"], "logistic_before_fix": cm["test_before_fix"], "calibrator": cm["calibrator"],
        "gradient_boosting": cm.get("gradient_boosting"), "splits": cm["splits"],
        "fixed_rain_threshold": {"rule": f"3-day rain >= {thr_r3:.0f} mm", **slope.summary_metrics(base_thr * 0.99 + 0.005, y[te], base),
                                 "hit_rate": float(((base_thr == 1) & (y[te] == 1)).sum() / max(y[te].sum(), 1)),
                                 "false_alarm_ratio": float(((base_thr == 1) & (y[te] == 0)).sum() / max(base_thr.sum(), 1))},
        "roc": cm["roc"], "reliability": cm["reliability"], "base_rate": base,
        "coefficients": {f: float(c) for f, c in zip(slope.FEATURES, dep["weights"][1:])},
        "slope_trigger": {"orange_days_per_year": float((r7c >= lhasa["p95"]).mean() * 365.25),
                          "red_days_per_year": float((r7c >= lhasa["p99"]).mean() * 365.25),
                          "note": "Percentile thresholds fire on a fixed share of days by design."},
        "note": "Calibration fitted only on training-period out-of-fold predictions (previously in-sample Platt, which "
                "under-predicted badly on unseen years). Gradient boosting reported as a comparison.",
    }

    def calibrate(sc):
        return slope.apply_isotonic(sc, dep["isotonic"]) if dep["calibrator"] == "isotonic" else slope.apply_platt(sc, *dep["platt"])

    def p_daily(Xf):
        return calibrate(slope.predict(np.array(dep["weights"]), (Xf - np.array(dep["mu"])) / np.array(dep["sd"])))

    # hindcasts of documented events with the TRAIN-ONLY model (no leakage for post-2010 events)
    hind = []
    for e in ev.EVENTS:
        d = np.datetime64(e.date)
        m = (arc["time"] >= d - np.timedelta64(10, "D")) & (arc["time"] <= d + np.timedelta64(3, "D"))
        if not m.any():
            continue
        pd_ = p_daily(X[m])
        w3 = slope.window_prob(pd_, 3)
        k = int(np.argmin(np.abs(arc["time"][m] - (d - np.timedelta64(1, "D")))))
        r7m = ix["r7"][m]
        sl_max = max(alerts.lhasa_level(float(v), **lhasa) for v in r7m[: k + 2])
        hind.append({"date": e.date, "event": e.kind, "place": e.place, "rain_triggered": e.rain_triggered,
                     "p72_day_before": float(w3[k]), "level_day_before": slope_level_name(alerts.slope_level(float(w3[k]))),
                     "r7_mm_on_day": float(r7m[min(k + 1, len(r7m) - 1)]), "slope_trigger_level": slope_level_name(sl_max),
                     "slope_outcome": ("hit" if sl_max >= 2 else "miss") if e.rain_triggered else
                                      ("correctly quiet" if sl_max == 0 else "flagged") if e.rain_triggered is False else "n/a (debated)",
                     "series": {"dates": [str(t) for t in arc["time"][m]], "p_daily": np.round(pd_, 4).tolist(),
                                "r7": np.round(r7m, 1).tolist(),
                                "rain": np.round(np.nan_to_num(arc["precipitation_sum"][m]), 1).tolist()},
                     "source": e.source, "note": e.note})
    # ---------------------------------------------------------------- 4. live forecast (snapshot at build time)
    fc, fc_meta = om.forecast(lat, lon)
    ixf = om.indices(fc["precipitation_sum"], fc["snowfall_sum"], fc["temperature_2m_mean"])
    smf = fc["soil_moisture_deep"]
    smf = np.where(np.isfinite(smf), smf, np.nanmedian(sm))
    pf = p_daily(_features(ixf, smf, tables))
    t_idx = int(np.argmin(np.abs(fc["time"] - today)))
    windows = {f"p{h}h": slope.window_prob(pf, h // 24).tolist() for h in (24, 72, 168)}
    lhasa_series = [alerts.lhasa_level(float(v), **lhasa) for v in ixf["r7"]]
    ls_out = landslide_forecast(region, lat, lon, fc, ixf, smf)
    if ls_out:
        ls_levels = ls_out["levels"]
    else:
        ls_levels = lhasa_series
    # ---------------------------------------------------------------- 5. physics: P(FS<1) maps for forecast days
    sm_dry, sm_wet = float(np.nanpercentile(sm, 5)), float(np.nanpercentile(sm, 99))
    tr_utm = Transformer.from_crs(4326, 32644, always_xy=True)
    w_, s_, e_, n_ = region.bbox
    xs, ys = tr_utm.transform([w_, e_], [s_, n_])
    x = np.arange(min(xs), max(xs), 30.0)
    yv = np.arange(max(ys), min(ys), -30.0)
    elev = ancillary.dem(32644, x, yv)
    slp, _ = ancillary.slope_aspect(elev, 30.0, -30.0)
    wc = ancillary.worldcover(32644, x, yv)
    rock = slp > 50  # cliffs: bedrock, infinite-slope soil model not applicable
    slp = np.where((wc == 80) | (wc == 70) | ~np.isfinite(elev) | rock, np.nan, slp)
    root_c = np.where(wc == 10, 5.0, np.where(np.isin(wc, (20, 30)), 1.5, 0.0))  # kPa root cohesion (literature)
    days_out = list(range(t_idx, len(fc["time"])))
    wets = slope.wetness(smf[days_out], sm_dry, sm_wet)
    pfs = np.stack([slope.prob_failure(slp, float(wi), c_add=root_c) for wi in wets])
    susceptibility = slope.prob_failure(slp, 1.0, c_add=root_c)
    regional_pfs = [float(np.nanmean(p[slp > 25])) for p in pfs]
    # ---------------------------------------------------------------- 6. DEHFE Kalman filter on NISAR settlement series
    kf_out = None
    stacks = sorted((WORK_DIR / region.id / "gdm").glob("stack_D*.npz")) + sorted((WORK_DIR / region.id / "gdm").glob("stack_A*.npz"))
    if stacks:
        z = np.load(stacks[0])
        wcg = ancillary.worldcover(int(z["epsg"]), z["x"], z["y"])
        rel = (z["p_reliable"] > 0.5) if "p_reliable" in z.files else np.isfinite(z["p_rate_per_yr"])
        town = rel & (wcg == 50)
        if town.sum() < 10:
            town = rel
        disp, sig = z["displacement"], z["sigma"]
        od = np.array(z["dates"]).astype("datetime64[D]")
        obs = np.array([np.nanmedian(disp[k][town]) for k in range(len(od))])
        scatter = np.array([1.4826 * np.nanmedian(np.abs(disp[k][town] - obs[k])) for k in range(len(od))])
        osig = np.sqrt((scatter / np.sqrt(max(town.sum(), 1))) ** 2 + 0.002 ** 2)
        osig[0] = 0.002
        days = np.arange(od[0], fc["time"][-1] + np.timedelta64(1, "D"))
        rmap = dict(zip(arc["time"], arc["precipitation_sum"]))
        rmap.update(dict(zip(fc["time"], fc["precipitation_sum"])))
        rain = np.array([np.nan_to_num(rmap.get(d, 0.0)) for d in days])
        params = assim.fit(days, rain, od, obs, osig)
        mean, var, _, _ = assim.run(days, rain, od, obs, osig, params["q_d"], params["q_v"], params["beta_m_per_mm"])
        hc = assim.hindcast(days, rain, od, obs, osig, params)
        kf_out = {"stack": stacks[0].stem.replace("stack_", ""), "pixels": int(town.sum()), "params": params,
                  "days": [str(d) for d in days], "mean_mm": np.round(mean[:, 0] * 1000, 2).tolist(),
                  "sd_mm": np.round(np.sqrt(var[:, 0]) * 1000, 2).tolist(),
                  "velocity_mm_yr": round(float(mean[-1, 1] * 365.25 * 1000), 2),
                  "velocity_sd_mm_yr": round(float(np.sqrt(var[-1, 1]) * 365.25 * 1000), 2),
                  "obs_dates": [str(d) for d in od], "obs_mm": np.round(obs * 1000, 2).tolist(),
                  "obs_sd_mm": np.round(osig * 1000, 2).tolist(), "hindcast": hc,
                  "next_pass": str(od[-1] + np.timedelta64(12 * max(1, int(np.ceil((today - od[-1]).astype(int) / 12))), "D")),
                  "note": "Next pass estimated from NISAR's 12-day repeat; processed products appear some days later."}
    # ---------------------------------------------------------------- 7. flood forecast + checks
    qf = q_fc["river_discharge"]
    qf_max = q_fc.get("river_discharge_max", qf)
    fi = int(np.argmin(np.abs(q_fc["time"] - today)))
    q7 = float(np.nanmax(qf_max[fi:fi + 7])) if np.isfinite(qf_max[fi:fi + 7]).any() else float("nan")
    flv = flood.level_for(q7, rl_live)
    floods = {"return_levels": rl_live, "return_levels_historical": rl, "record_break": {**brk, "used": by,
              "note": (f"GloFAS discharge steps up by x{brk.get('jjas_mean_ratio_after_before')} in mean monsoon flow from "
                       f"{by}: most likely a change of GloFAS product version in the record, not a real flood regime. "
                       f"Models are trained/tested on the consistent pre-{by} record; live alerts use return levels from "
                       f"the {by}-onward record (same product as the forecasts; short record, uncertain).") if by else None}, "annual_maxima": {"years": yrs.tolist(), "q": np.round(amax, 1).tolist(), "dates": amax_dates},
              "forecast": {"dates": [str(t) for t in q_fc["time"]], "q": np.round(qf, 2).tolist(),
                           "q_max": np.round(qf_max, 2).tolist()},
              "recent": {"dates": [str(t) for t in q_hist["time"][-120:]], "q": np.round(q_hist["river_discharge"][-120:], 2).tolist()},
              "max_next_7d": q7, "level_next_7d": flv,
              "events": [*flood.event_check(q_hist["time"], q_hist["river_discharge"], rl,
                                            [e for e in ev.EVENTS if not by or int(e.date[:4]) < by]),
                         *flood.event_check(q_hist["time"], q_hist["river_discharge"], rl_live,
                                            [e for e in ev.EVENTS if by and int(e.date[:4]) >= by])],
              "pod_annual_max_above_Q5": None}
    # ---------------------------------------------------------------- 8. alerts (today)
    p72_today = windows["p72h"][t_idx]
    slope_lvl = max(ls_levels[t_idx:t_idx + 3])
    river_lvl = alerts.slope_level(p72_today)
    alert = alerts.combine(
        slope_lvl, river_lvl, flv,
        (f"Landslide model (trained on {ls_out['validation']['events_used']} NASA-catalogued Uttarakhand landslides) is at "
         f"its {['', 'watch', 'warning', 'critical'][slope_lvl]} level for the next 72 h." if ls_out else
         f"7-day rain reaches {max(ixf['r7'][t_idx:t_idx + 3]):.0f} mm, above the site's "
         f"{['', '90th', '95th', '99th'][slope_lvl]} percentile (LHASA-style landslide trigger)."),
        f"{p72_today:.0%} chance of the Alaknanda reaching its 2-year flood level within 72 h (model AUC "
        f"{val['logistic']['roc_auc']:.2f} on unseen years).",
        f"Forecast Alaknanda discharge reaches {q7:.0f} m3/s (>= {['', '2-year', '5-year', '20-year'][flv]} flood level of the current GloFAS product).")
    alert.update({"issued_utc": fc_meta["issued_utc"], "valid_from": str(fc["time"][t_idx]),
                  "valid_to": str(fc["time"][min(t_idx + 2, len(fc["time"]) - 1)]),
                  "inputs": [{"name": "Rain, snow, temperature forecast", "source": fc_meta["source"], "fresh": fc_meta["issued_utc"]},
                             {"name": "Soil moisture 9-27 cm (forecast) / 7-28 cm (ERA5-Land)", "source": "Open-Meteo", "fresh": fc_meta["issued_utc"]},
                             {"name": "River discharge", "source": q_meta["source"], "fresh": str(q_hist["time"][-1])},
                             {"name": "Ground deformation", "source": "NISAR GUNW (this project)", "fresh": kf_out["obs_dates"][-1] if kf_out else "n/a"}]})
    # ---------------------------------------------------------------- 9. replay of the full advisory system (no leakage)
    replay = None
    if ls_out:
        lsm = ls_out["browser_model"]
        feats_hist = {"sm": sm, **{k: ix[k] for k in ("r1", "r3", "r7", "api", "t7")}}
        Xl = np.column_stack([om.to_pct(feats_hist[k], lsm["tables"][k]) for k in lsm["features"]])
        sc = slope.predict(np.array(lsm["weights"]), (Xl - np.array(lsm["mu"])) / np.array(lsm["sd"]))
        p_ls_hist = slope.apply_isotonic(sc, lsm["isotonic"]) if lsm.get("calibrator") == "isotonic" else slope.apply_platt(sc, *lsm["platt"])
        p_ls_hist = np.where(np.isfinite(sm), p_ls_hist, 0.0)
        slope_lv = np.array([sum(v >= t for t in lsm["thresholds"]) for v in p_ls_hist])
        Xr = X.copy()
        p_riv = np.where(np.isfinite(sm), p_daily(np.nan_to_num(Xr)), 0.0)
        p72r = slope.window_prob(p_riv, 3)
        river_lv = np.array([alerts.slope_level(float(v)) for v in p72r])
        qlv = np.array([flood.level_for(float(v), rl) for v in q_al])
        alert_lv = np.maximum(np.maximum(slope_lv, river_lv), qlv)
        ry0, ry1 = CAL_END + 1, (by - 1) if by else int(ayr.max())
        win = (ayr >= ry0) & (ayr <= ry1) & np.isfinite(sm)
        # targets: catalogued landslides within ~60 km (GLC covers to mid-2017) and river high-flow days
        from earthpulse.hazard import landslide as lsmod
        from earthpulse.config import DATA_DIR
        glc_days = set()
        if (DATA_DIR / "external" / "glc.csv").exists():
            for e in lsmod.load_events(DATA_DIR / "external" / "glc.csv"):
                if e["trigger"] in lsmod.RAIN_TRIGGERS and abs(e["lat"] - lat) < 0.6 and abs(e["lon"] - lon) < 0.6:
                    glc_days.add(e["date"])
        tstr = arc["time"].astype(str)
        ls_target = np.array([d in glc_days for d in tstr])
        ls_target3 = np.zeros(len(ls_target), bool)  # landslide within the next 3 days (advisory horizon)
        for k in range(3):
            ls_target3 |= np.concatenate([ls_target[k:], np.zeros(k, bool)])
        glc_win = win & (arc["time"] <= np.datetime64("2017-07-13"))

        def score(level, thr, target, mask):
            pr = (level >= thr) & mask
            tg = target & mask
            tp, fp, fn = int((pr & tg).sum()), int((pr & ~tg).sum()), int((~pr & tg).sum())
            return {"pod": tp / max(tp + fn, 1), "far": fp / max(tp + fp, 1), "csi": tp / max(tp + fp + fn, 1),
                    "alarm_days_per_year": float(pr.sum() / max(mask.sum(), 1) * 365.25), "events": int(tg.sum())}
        replay = {"period": f"{ry0}-{ry1}", "note": "Day-by-day replay using only models fitted on data up to "
                  f"{CAL_END}; river level from the calibrated model, flood level from the observed (reanalysis) discharge as a "
                  "perfect-forecast upper bound; landslide targets from the NASA catalogue (to 2017-07) within ~60 km.",
                  "landslide": {lvl: score(slope_lv, i, ls_target3, glc_win) for i, lvl in ((1, ">= yellow"), (2, ">= orange"), (3, "red"))},
                  "river": {lvl: score(river_lv, i, y.astype(bool), win & ok) for i, lvl in ((1, ">= yellow"), (2, ">= orange"), (3, "red"))},
                  "combined_alert_days_per_year": {LV: float(((alert_lv >= i) & win).sum() / max(win.sum(), 1) * 365.25)
                                                   for i, LV in ((1, "yellow+"), (2, "orange+"), (3, "red"))}}
        # flood POD / FAR on annual maxima >= Q5 in the replay years
        hits, misses = 0, 0
        for yy, qm, dt in zip(yrs, amax, amax_dates):
            if ry0 <= yy <= ry1 and qm >= rl["Q5"]:
                i = int(np.nonzero(tstr == dt)[0][0]) if (tstr == dt).any() else None
                if i is not None and river_lv[max(0, i - 3):i + 1].max() >= 1:
                    hits += 1
                else:
                    misses += 1
        floods["pod_annual_max_above_Q5"] = {"years": f"{ry0}-{ry1}", "floods": hits + misses, "detected": hits,
                                             "pod": hits / max(hits + misses, 1) if hits + misses else None}
        # landslide-model hindcast of documented events (model trained <= 2012; all events are out-of-sample)
        for h in hind:
            d = np.datetime64(h["date"])
            k = np.nonzero(arc["time"] == d - np.timedelta64(1, "D"))[0]
            if k.size:
                i = int(k[0])
                p3 = 1 - np.prod(1 - p_ls_hist[i:i + 3])
                h["landslide_p72_day_before"] = float(p3)
                h["landslide_level_day_before"] = slope_level_name(int(max(slope_lv[i:i + 3])))
    # ---------------------------------------------------------------- 10. combined hazard map (physics x trigger)
    combined = None
    if ls_out:
        thr_or = ls_out["thresholds"][1]
        rel_trig = np.clip(np.array(ls_out["p_daily"])[days_out] / thr_or, 0, 1)
        combined = pfs * rel_trig[:, None, None]
    # ---------------------------------------------------------------- export
    out_dir = WEB_DATA_DIR / region.id / "hazards"
    shutil.rmtree(out_dir, ignore_errors=True)
    lw = LayerWriter(out_dir, 32644, x, yv)
    fdates = [str(fc["time"][d]) for d in days_out]
    lw.cube("pfs", pfs, fdates, title="Chance a slope fails (factor of safety < 1)", units="probability", cmap="YlOrRd",
            vmin=0, vmax=0.6, group="Forecast",
            how_to_read="For each forecast day: share of 200 plausible soil-strength scenarios in which the slope is unstable, "
                        "given forecast soil wetness. Steep, wet slopes turn orange/red. Soil parameters are assumed "
                        "(literature ranges), so read it as relative danger, not a precise probability.")
    if combined is not None:
        lw.cube("combined", combined, fdates, title="Combined hazard: unstable slope x landslide trigger", units="index 0-1",
                cmap="YlOrRd", vmin=0, vmax=0.5, group="Forecast",
                how_to_read="Slope-failure chance (physics) multiplied by how close the day's rain/soil conditions are to the "
                            "landslide model's warning level. Dark red needs both a steep, weak slope and a wet trigger.")
    lw.raster("susceptibility", susceptibility, title="Worst case: fully saturated soil", units="probability", cmap="YlOrRd",
              vmin=0, vmax=1, group="Context", how_to_read="P(FS<1) if the soil were completely saturated - where slopes "
                                                         "would be most dangerous in an extreme monsoon.")
    lw.raster("slope", slp, title="Slope angle", units="degrees", cmap="cividis", vmin=0, vmax=60, group="Context",
              how_to_read="Terrain steepness from the Copernicus DEM (30 m).")
    model = {"schema": "earthpulse-hazard/2", "features": list(slope.FEATURES), "weights": dep["weights"],
             "mu": dep["mu"], "sd": dep["sd"], "calibrator": dep["calibrator"], "platt": dep["platt"], "isotonic": dep["isotonic"],
             "tables": tables, "api_k": 0.9,
             "soil": {"sm_dry": sm_dry, "sm_wet": sm_wet}, "lat": lat, "lon": lon, "river_lat": rlat, "river_lon": rlon,
             "return_levels": rl_live,
             "alert_bands_p72": [0.15, 0.35, 0.6], "lhasa_r7": lhasa,
             "landslide": ls_out["browser_model"] if ls_out else None}
    (out_dir / "model.json").write_text(json.dumps(model))
    manifest = {
        "region": region.id, "generated": datetime.utcnow().isoformat() + "Z", "layers": lw.layers,
        "experimental": True,
        "forcing": {"archive": arc_meta, "forecast": fc_meta, "flood": q_meta},
        "trigger": {"dates": [str(t) for t in fc["time"]], "today_index": t_idx, "p_daily": np.round(pf, 4).tolist(), **windows,
                    "rain": np.round(np.nan_to_num(fc["precipitation_sum"]), 1).tolist(),
                    "soil_moisture": np.round(smf, 3).tolist(), "regional_pfs": regional_pfs, "pfs_dates": fdates,
                    "r7": np.round(ixf["r7"], 1).tolist(), "slope_level": lhasa_series, "lhasa": lhasa},
        "validation": val, "hindcasts": hind, "kalman": kf_out, "flood": floods, "alert": alert, "replay": replay,
        "nisar_soil_moisture": nisar_soil_moisture(region, arc),
        "landslide": {k: v for k, v in ls_out.items() if k != "browser_model"} if ls_out else None,
        "levels": alerts.LEVELS,
        "assumptions": [
            "Soil cohesion 3-12 kPa (+5 kPa root cohesion under forest, +1.5 kPa under shrub/grass), friction angle "
            "28-38 deg, soil depth 1-3 m (literature ranges for Himalayan colluvium). Slopes > 50 deg (bedrock cliffs), "
            "snow/ice and water are excluded.",
            "Slope wetness from ERA5-Land / forecast soil moisture, rescaled between its 5th and 99th percentiles.",
            "Landslide trigger: logistic model on rain/soil-moisture percentiles trained on 186 NASA Global Landslide Catalog "
            "events (2007-2012), isotonic calibration from leave-one-year-out predictions, tested on untouched 2013-2017 against "
            "the LHASA 7-day-rain rule; alert thresholds at the site's 90th/95th/99th percentile of model probability "
            "(1998-2017). Falls back to LHASA if the model is absent.",
            "River model label: GloFAS discharge above the 2-year flood level within 3 days; trained on 1996-2012 with "
            "leave-one-year-out calibration, tested on untouched 2013-2019 (records after the 2020 GloFAS product break are "
            "excluded from training/testing; live alerts use post-break return levels).",
            "Daily hazard probabilities are combined assuming independence between days.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(_clean(manifest), default=float))
    log.info("hazards %s: AUC test %.2f, BSS %.2f, p72 today %.2f, flood level %d, KF hindcast %s",
             region.id, val["logistic"]["roc_auc"], val["logistic"]["brier_skill_score"] or 0, p72_today, flv,
             kf_out and (round(kf_out["hindcast"].get("rmse_kf_mm", 0), 1), round(kf_out["hindcast"].get("rmse_persistence_mm", 0), 1)))
    return manifest


def slope_level_name(i: int) -> str:
    return alerts.LEVELS[i]["name"]
