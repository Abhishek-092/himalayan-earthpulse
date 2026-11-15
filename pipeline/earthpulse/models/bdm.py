"""BDM - Biophysical Dynamics Model -> FSTS (Forest Structural Transition Signature).

From the NISAR GCOV stack (gamma-0, RTC, HH + HV, L-band):
  1. 2x2 multilook to 20 m, valid/water/forest masks (ESA WorldCover tree cover)
  2. complex-Wishart omnibus + sequential R_j test (diagonal dual-pol) -> change points with p-values & dates
  3. "potential disturbance" = significant change in which HV (volume scattering from branches/trunks) DROPS
     inside forest; recovery = later significant HV increase or return toward the pre-change level
  4. GLCM texture (HV) before vs after -> multi-feature agreement (confidence)
  5. dual-pol radar vegetation index RVI = 4 HV / (HH + HV) as structural indicator time series
  6. fragmentation & edge density of forest before/after disturbance
  7. validation against Sentinel-2 dNBR (independent optical evidence) -> precision/recall/F1/IoU
Wording rule: outputs are "potential disturbance", never "deforestation" - radar change alone cannot confirm it.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import numpy as np
from scipy import ndimage

from earthpulse.catalog.discover import parse_granule_name
from earthpulse.catalog.stacks import region_granules, select_stacks
from earthpulse.change.omnibus import sequential_change
from earthpulse.config import WORK_DIR, Region
from earthpulse.features.texture import glcm_features
from earthpulse.geo import ancillary
from earthpulse.geo.area import masked_area_km2, pixel_area_m2
from earthpulse.io.nisar import fetch_many
from earthpulse.preprocess.speckle import estimate_enl, quegan, to_db
from earthpulse.temporal.framework import TemporalConfig, analyze, summarize_series
from earthpulse.validation import sentinel2
from earthpulse.validation.metrics import categorical

log = logging.getLogger(__name__)


def _ml2(a: np.ndarray) -> np.ndarray:
    H2, W2 = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    with np.errstate(all="ignore"):
        return np.nanmean(a[:H2, :W2].reshape(H2 // 2, 2, W2 // 2, 2), axis=(1, 3))


def load_gcov_stack(region: Region, workers: int = 12):
    gs = region_granules(region)
    stacks = select_stacks(gs, "GCOV")
    if not stacks:
        raise RuntimeError(f"no GCOV stack for {region.id}")
    from earthpulse.io.nisar import cache_path

    def ncached(ms):
        return sum(cache_path(region.id, "GCOV", g.granule_id).exists() for g in ms)
    # prefer the stack already ingested; use only cached dates when enough exist (>= 8)
    key, members = max(stacks.items(), key=lambda kv: (ncached(kv[1]) >= 8, len(kv[1])))
    cached = [g for g in members if cache_path(region.id, "GCOV", g.granule_id).exists()]
    if len(cached) >= 8:
        members = cached
    subs = fetch_many(region, [g.to_json() for g in members], workers=workers)
    shape0 = max({s.shape for s in subs}, key=lambda sh: sum(s.shape == sh for s in subs))
    subs = [s for s in subs if s.shape == shape0 and "HHHH" in s.layers and "HVHV" in s.layers]
    subs.sort(key=lambda s: parse_granule_name(s.granule_id)["ref_time"])
    hh = np.stack([_ml2(np.where(s.layers["HHHH"] > 0, s.layers["HHHH"], np.nan)) for s in subs]).astype(np.float32)
    hv = np.stack([_ml2(np.where(s.layers["HVHV"] > 0, s.layers["HVHV"], np.nan)) for s in subs]).astype(np.float32)
    x = subs[0].x[: hh.shape[2] * 2].reshape(-1, 2).mean(1)
    y = subs[0].y[: hh.shape[1] * 2].reshape(-1, 2).mean(1)
    dates = np.array([np.datetime64(parse_granule_name(s.granule_id)["ref_time"][:10], "D") for s in subs])
    used = {s.granule_id for s in subs}
    return {"key": key, "x": x, "y": y, "epsg": subs[0].epsg, "hh": hh, "hv": hv, "dates": dates,
            "ids": [s.granule_id for s in subs], "direction": members[0].direction,
            "members": [g for g in members if g.granule_id in used], "all_members": stacks[key]}


def _fragmentation(forest: np.ndarray, px_m: float, area_km2: float) -> dict[str, float]:
    lab, n = ndimage.label(forest)
    if n == 0:
        return {"patches": 0, "mean_patch_ha": 0.0, "edge_density_m_per_km2": 0.0, "largest_patch_pct": 0.0}
    sizes = np.bincount(lab.ravel())[1:]
    edges = forest & ~ndimage.binary_erosion(forest)
    return {"patches": int(n), "mean_patch_ha": float(sizes.mean() * px_m * px_m / 1e4),
            "edge_density_m_per_km2": float(edges.sum() * px_m / max(area_km2, 1e-9)),
            "largest_patch_pct": float(100 * sizes.max() / max(forest.sum(), 1))}


def run_bdm(region: Region, alpha: float = 0.01) -> dict[str, Any]:
    st = load_gcov_stack(region)
    x, y, epsg, dates = st["x"], st["y"], st["epsg"], st["dates"]
    hh, hv = st["hh"], st["hv"]
    T = len(dates)
    out_dir = WORK_DIR / region.id / "bdm"
    out_dir.mkdir(parents=True, exist_ok=True)
    wc = ancillary.worldcover(epsg, x, y)
    occ = np.nan_to_num(ancillary.water_occurrence(epsg, x, y))
    elev = ancillary.dem(epsg, x, y)
    slope, aspect = ancillary.slope_aspect(elev, x[1] - x[0], y[1] - y[0])
    # ---- quality masks the product does not apply (GCOV: noiseCorrectionApplied/shadowMaskingApplied = False) ----
    from earthpulse.preprocess import masks as pm
    from earthpulse.preprocess.noise import noise_correct, noise_floor_mask

    geom = pm.load_geometry(region.id, "GCOV", st["all_members"])
    rm = pm.radar_masks(geom, x, y, elev)
    nz = pm.nesz_maps(geom, x, y)
    hh_raw, hv_raw = hh.copy(), hv.copy()  # raw intensities keep the Wishart statistics for ENL + omnibus test
    noisy = np.zeros(hv.shape[1:], bool)
    if "HV" in nz:
        noisy = np.mean([noise_floor_mask(hv[t], nz["HV"]) for t in range(T)], axis=0) > 0.5
        hv = np.stack([noise_correct(hv[t], nz["HV"]) for t in range(T)]).astype(np.float32)
    if "HH" in nz:
        hh = np.stack([noise_correct(hh[t], nz["HH"]) for t in range(T)]).astype(np.float32)
    raw_valid = np.all(np.isfinite(hh) & np.isfinite(hv), axis=0)
    valid = raw_valid & ~rm["layover"] & ~rm["shadow"] & ~noisy
    forest = (wc == 10) & valid & (occ < 20)
    snow = pm.snow_dates(region, dates)
    pass_times = [parse_granule_name(g)["ref_time"] for g in st["ids"]]
    frz = pm.freeze_maps(region, pass_times, elev)
    pre = {"layover_pct": round(100 * float((rm["layover"] & raw_valid).mean()), 2),
           "shadow_pct": round(100 * float((rm["shadow"] & raw_valid).mean()), 2),
           "noise_floor_pct": round(100 * float((noisy & raw_valid).mean()), 2),
           "noise_corrected": "HV" in nz, "geometry_available": rm["available"],
           "incidence_range_deg": rm.get("incidence_range"),
           "product_flags": geom.attrs.get("processing") if geom else None,
           "snow": snow, "freeze": {k: v for k, v in frz.items() if k != "frozen"}}
    enl = float(np.clip(estimate_enl(hv_raw[0]), 3.0, 60.0))
    log.info("BDM %s: %d dates, grid %s, ENL %.1f, forest px %d", region.id, T, hh.shape[1:], enl, forest.sum())

    # --- omnibus change detection (raw multilooked intensities keep Wishart statistics) -------------------
    data = np.stack([np.nan_to_num(hh_raw, nan=1e-6), np.nan_to_num(hv_raw, nan=1e-6)], axis=1)
    data = np.maximum(data, 1e-6)
    ch = sequential_change(data, enl=enl, alpha=alpha, diagonal=True, ref_channel=1)
    darker = ch.change_matrix & (ch.direction < 0)
    brighter = ch.change_matrix & (ch.direction > 0)
    any_dark = darker.any(0) & forest
    first_dark = np.where(any_dark, np.argmax(darker, axis=0), -1)
    # recovery: a brighter change after the first darkening
    after = np.arange(T)[:, None, None] > first_dark[None]
    recovered_change = (brighter & after).any(0) & any_dark

    # --- filtered series for display/temporal analysis ---------------------------------------------------
    hv_f = quegan(hv, 5)
    hh_f = quegan(hh, 5)
    hv_db = to_db(hv_f)
    rvi = 4 * hv_f / (hh_f + hv_f)
    tr = analyze(dates, np.moveaxis(hv_db, 0, -1), None, TemporalConfig(baseline_n=3, k_sigma=2.0, min_persist=2))

    # --- texture corroboration -----------------------------------------------------------------------------
    nb = min(3, T // 2)
    pre_db = to_db(np.nanmean(hv_f[:nb], 0))
    post_db = to_db(np.nanmean(hv_f[-nb:], 0))
    GLCM_ALL = ("contrast", "entropy", "homogeneity", "energy", "correlation", "dissimilarity")
    tex_pre = glcm_features(pre_db, window=7, levels=8, features=GLCM_ALL)
    tex_post = glcm_features(post_db, window=7, levels=8, features=GLCM_ALL)
    d_ent = tex_post["entropy"] - tex_pre["entropy"]
    glcm_change = {f"glcm_{k}_change": (tex_post[k] - tex_pre[k]).astype(np.float32) for k in GLCM_ALL}
    glcm_pre = {f"glcm_{k}_pre": tex_pre[k].astype(np.float32) for k in GLCM_ALL}
    ratio_db = to_db(hv_f) - to_db(hh_f)  # HV/HH cross-pol ratio (dB): higher = more volume scattering
    ratio_pre = np.nanmean(ratio_db[:nb], 0)
    ratio_change = np.nanmean(ratio_db[-nb:], 0) - ratio_pre
    lo, hi = np.nanpercentile(d_ent[forest], [5, 95]) if forest.any() else (0, 0)
    texture_shift = (d_ent < lo) | (d_ent > hi)
    db_drop = (post_db - pre_db) < -1.5
    agreement = any_dark & (texture_shift | db_drop)

    # --- confidence: high = significant (p<0.001) + texture/dB agreement; medium = significant; low otherwise
    conf = np.zeros(any_dark.shape, np.uint8)
    conf[any_dark] = 1
    conf[any_dark & ((ch.min_pvalue < 1e-3) | agreement)] = 2
    conf[any_dark & (ch.min_pvalue < 1e-3) & agreement] = 3
    # remove isolated single pixels (speckle-scale false alarms)
    clean = ndimage.binary_opening(any_dark, np.ones((2, 2)))
    disturbance = clean & forest
    snow_flag = np.array(snow["snow"] + [False])  # index -1 (no change) -> False
    onset_idx = np.where(first_dark >= 0, first_dark, 0)
    frozen_at_onset = np.take_along_axis(frz["frozen"], onset_idx[None], 0)[0]
    onset_on_snow = disturbance & (snow_flag[np.where(first_dark >= 0, first_dark, len(snow_flag) - 1)] | frozen_at_onset)
    lasting = disturbance & ~recovered_change & ~onset_on_snow
    conf[~disturbance] = 0

    area = pixel_area_m2(epsg, x, y)
    px = float(abs(x[1] - x[0]))
    forest_km2 = masked_area_km2(forest, area)
    dist_km2 = masked_area_km2(disturbance, area)
    by_date = []
    for t in range(1, T):
        m = disturbance & (first_dark == t)
        by_date.append({"date": str(dates[t]), "new_disturbance_km2": round(masked_area_km2(m, area), 4)})
    cum = np.cumsum([d["new_disturbance_km2"] for d in by_date]).tolist()
    for d, c in zip(by_date, cum):
        d["cumulative_km2"] = round(c, 4)
    frag_pre = _fragmentation(forest, px, forest_km2)
    frag_post = _fragmentation(forest & ~disturbance, px, forest_km2)

    # regional series (forest median HV, RVI; disturbed-pixel median HV)
    def med_series(arr, m):
        return [float(np.nanmedian(a[m])) if np.isfinite(a[m]).any() else None for a in arr]
    series = {"dates": [str(d) for d in dates], "forest_hv_db": med_series(hv_db, forest & ~disturbance),
              "disturbed_hv_db": med_series(hv_db, disturbance), "forest_rvi": med_series(rvi, forest & ~disturbance),
              "disturbed_rvi": med_series(rvi, disturbance), "forest_hh_db": med_series(to_db(hh_f), forest & ~disturbance)}
    dist_temporal = None
    if disturbance.any() and all(v is not None for v in series["disturbed_hv_db"]):
        dist_temporal = summarize_series(dates, np.array(series["disturbed_hv_db"]), None,
                                         TemporalConfig(baseline_n=3, k_sigma=2.0))

    # --- Sentinel-2 dNBR validation around the main disturbance window ----------------------------------
    validation = None
    if disturbance.sum() > 20:
        counts = np.bincount(first_dark[disturbance], minlength=T)
        tpk = int(np.argmax(counts))
        pre_end = str(dates[max(tpk - 1, 0)])
        pre_start = str(dates[max(tpk - 1, 0)] - np.timedelta64(45, "D"))
        post_start = str(dates[tpk])
        post_end = str(dates[min(tpk + 3, T - 1)])
        try:
            dn = sentinel2.dnbr(epsg, x, y, (pre_start, pre_end), (post_start, post_end))
        except Exception as e:
            log.warning("Sentinel-2 validation failed: %s", e)
            dn = None
        if dn is not None:
            ref = dn["dnbr"] > 0.1  # USGS low-severity burn / canopy-loss threshold
            obs = forest & np.isfinite(dn["dnbr"])
            # compare at 60 m blocks to tolerate geolocation/resolution differences
            f3 = lambda a: ndimage.maximum_filter(a.astype(np.uint8), 3).astype(bool)
            m_pix = categorical(disturbance, ref, obs)
            m_blk = categorical(f3(disturbance)[::3, ::3], f3(ref)[::3, ::3], obs[::3, ::3])
            from earthpulse.validation.metrics import area_and_boundary
            ab = area_and_boundary(disturbance, ref, obs, area, px)
            ab_l = area_and_boundary(lasting, ref, obs, area, px)
            m_blk_l = categorical(f3(lasting)[::3, ::3], f3(ref)[::3, ::3], obs[::3, ::3])
            validation = {"source": "Sentinel-2 L2A dNBR (Planetary Computer)", "pre_window": [pre_start, pre_end],
                          "post_window": [post_start, post_end], "threshold": 0.1, "pixel": m_pix, "area_boundary": ab,
                          "lasting_only": {"area_boundary": ab_l, "block_60m": m_blk_l},
                          "block_60m": m_blk, "s2_valid_fraction": float(obs.sum() / max(forest.sum(), 1))}
            np.savez_compressed(out_dir / "s2_validation.npz", dnbr=dn["dnbr"].astype(np.float32),
                                pre_rgb=np.stack([dn["pre"]["B04"], dn["pre"]["B03"], dn["pre"]["B02"]]).astype(np.float32),
                                post_rgb=np.stack([dn["post"]["B04"], dn["post"]["B03"], dn["post"]["B02"]]).astype(np.float32))

    # ---- coherence variation from NISAR GUNW (independent of backscatter) ----
    from earthpulse.preprocess.coherence import coherence_stack, masked_series
    coh = coherence_stack(region, x, y, epsg)
    coh_extra: dict[str, Any] = {}
    coherence_summary = None
    if coh is not None:
        C = coh["coherence"]
        k3 = max(1, len(C) // 3)
        coh_mean = np.nanmean(C, 0)
        coh_change = np.nanmean(C[-k3:], 0) - np.nanmean(C[:k3], 0)
        undist = forest & ~disturbance
        cs_f, cs_d = masked_series(C, undist), masked_series(C, disturbance) if disturbance.any() else [None] * len(C)
        coherence_summary = {
            "stack": coh["stack"], "pairs": coh["pairs"], "mid_dates": [str(d) for d in coh["mid_dates"]],
            "temporal_baseline_days": coh["temporal_baseline_days"], "forest_median": cs_f, "disturbed_median": cs_d,
            "forest_mean_coherence": float(np.nanmedian(coh_mean[undist])) if undist.any() else None,
            "disturbed_mean_coherence": float(np.nanmedian(coh_mean[disturbance])) if disturbance.any() else None,
            "forest_coherence_change": float(np.nanmedian(coh_change[undist])) if undist.any() else None,
            "lasting_coherence_change": float(np.nanmedian(coh_change[lasting])) if lasting.any() else None,
            "note": "L-band 12-day coherence from NISAR GUNW (80 m, resampled). Forest coherence is low and seasonal "
                    "(wind, moisture, snow); a lasting coherence change where HV dropped supports a structural change.",
            "granules": coh["granules"]}
        coh_extra = {"coherence_mean": coh_mean.astype(np.float32), "coherence_change": coh_change.astype(np.float32)}

    np.savez_compressed(
        out_dir / "bdm.npz", **coh_extra, **glcm_change, **glcm_pre,
        hvhh_ratio_pre=ratio_pre.astype(np.float32), hvhh_ratio_change=ratio_change.astype(np.float32), x=x, y=y, epsg=epsg, dates=dates.astype(str), hv_db=hv_db.astype(np.float32),
        hh_db=to_db(hh_f).astype(np.float32), rvi=rvi.astype(np.float32), forest=forest, disturbance=disturbance,
        first_dark=first_dark.astype(np.int16), recovered=recovered_change & disturbance, confidence=conf,
        lasting=lasting, snow_affected=onset_on_snow, layover=rm["layover"], shadow=rm["shadow"], noise_floor=noisy,
        p_nochange=ch.p_nochange.astype(np.float32), min_p=ch.min_pvalue.astype(np.float32),
        n_changes=ch.n_changes, entropy_change=d_ent.astype(np.float32), slope=slope, elev=elev.astype(np.float32),
        hv_raw_first=to_db(hv[0]).astype(np.float32), hv_filt_first=hv_db[0].astype(np.float32),
        worldcover=wc, **{f"p_{k}": v.astype(np.float32) for k, v in tr.params.items()})
    summary = {
        "region": region.id, "model": "BDM", "signature": "FSTS", "stack": st["key"], "granules": st["ids"],
        "dates": [str(d) for d in dates], "enl": enl, "alpha": alpha, "grid_m": px,
        "forest_area_km2": round(forest_km2, 2), "potential_disturbance_km2": round(dist_km2, 3),
        "potential_disturbance_pct_of_forest": round(100 * dist_km2 / max(forest_km2, 1e-9), 3),
        "recovering_km2": round(masked_area_km2(recovered_change & disturbance, area), 3),
        "snow_affected_km2": round(masked_area_km2(onset_on_snow, area), 3),
        "persistent_disturbance_km2": round(masked_area_km2(lasting, area), 3),
        "preprocessing": pre,
        "confidence_km2": {lvl: round(masked_area_km2(conf == i, area), 3)
                           for i, lvl in ((3, "high"), (2, "medium"), (1, "low"))},
        "disturbance_by_date": by_date, "fragmentation_before": frag_pre, "fragmentation_after": frag_post,
        "disturbance_rate_km2_per_yr": round(dist_km2 / max(float((dates[-1] - dates[0]).astype(int)) / 365.25, 1e-6), 3),
        "mean_slope_disturbed_deg": float(np.nanmean(slope[disturbance])) if disturbance.any() else None,
        "series": series, "disturbed_temporal": dist_temporal, "validation": validation,
        "coherence": coherence_summary,
        "second_opinion": None if coherence_summary is None or coherence_summary["lasting_coherence_change"] is None else {
            "lasting_vs_forest_coherence_change": round(coherence_summary["lasting_coherence_change"] - coherence_summary["forest_coherence_change"], 3),
            "lasting_vs_forest_hvhh_change_db": round(float(np.nanmedian(ratio_change[lasting]) - np.nanmedian(ratio_change[forest & ~disturbance])), 2),
            "supports_structural_change": bool(abs(coherence_summary["lasting_coherence_change"] - coherence_summary["forest_coherence_change"]) > 0.05
                                               or abs(float(np.nanmedian(ratio_change[lasting]) - np.nanmedian(ratio_change[forest & ~disturbance]))) > 0.5)},
        "hvhh_ratio_db": {"forest_pre": float(np.nanmedian(ratio_pre[forest])) if forest.any() else None,
                          "forest_change": float(np.nanmedian(ratio_change[forest & ~disturbance])) if forest.any() else None,
                          "lasting_change": float(np.nanmedian(ratio_change[lasting])) if lasting.any() else None},
        "glcm_change_lasting": {k.replace("glcm_", "").replace("_change", ""): float(np.nanmedian(v[lasting])) if lasting.any() else None
                                for k, v in glcm_change.items()},
        "limitations": [
            "Reversible winter drops in HV (frozen canopy, snow on higher slopes, moisture) are flagged by the change test; "
            "they are separated as 'recovered' or 'freeze/snow-affected' (ERA5 temperature lapse-adjusted to each pixel at the "
            "pass hour, ERA5-Land snow depth), but part of the winter signal remains unexplained at ~9 km meteorological "
            "resolution. Only 'lasting' changes are candidates for real disturbance.",
            "Radar backscatter change indicates potential disturbance; it cannot by itself confirm deforestation "
            "or measure biomass.",
            "Soil and canopy moisture (rain, monsoon) also change HV backscatter; the omnibus test flags any "
            "statistically significant change, so wet-season effects can create false alarms.",
            "Steep slopes facing away from the radar are partly in radar shadow (masked where invalid).",
            "Sentinel-2 dNBR is limited by monsoon cloud cover and measures burn/greenness, not structure.",
        ],
        "generated": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    pm.write_processing_log(
        region, "bdm", st["members"],
        params={"multilook": "2x2 (20 m)", "speckle_filter": "Quegan multitemporal 5x5 (display/temporal), raw multilooked "
                "intensities for the omnibus test", "enl": enl, "omnibus_alpha": alpha, "test": "complex-Wishart omnibus + "
                "sequential R_j (diagonal dual-pol)", "noise_correction": "noise-floor mask (within 3 dB of NESZ on >50% of "
                "dates); NESZ-subtracted values for display/temporal series; ENL + omnibus on raw intensities (Wishart)", "layover_shadow": "DEM + product LOS/incidence cubes, local + cast "
                "shadow", "snow_freeze": "onset date with ERA5-Land snow depth > 1 cm (centroid) OR pixel temperature < -1 C at the "
                "pass hour (ERA5 2 m temperature, lapse rate 6.5 K/km to pixel elevation) -> freeze/snow-affected", "glcm": "7x7, 8 levels, 4 directions",
                "forest_mask": "ESA WorldCover 2021 class 10", "water_mask": "JRC occurrence < 20%"},
        ancillary=["Copernicus DEM GLO-30", "ESA WorldCover 2021 v200", "JRC Global Surface Water 1984-2021",
                   "ERA5-Land snow depth (Open-Meteo)", "Sentinel-2 L2A (validation)"],
        qc={k: pre[k] for k in ("layover_pct", "shadow_pct", "noise_floor_pct", "noise_corrected", "geometry_available")},
        outputs=["bdm.npz", "summary.json", "s2_validation.npz"],
        product_processing=geom.attrs.get("processing") if geom else None)
    return summary
