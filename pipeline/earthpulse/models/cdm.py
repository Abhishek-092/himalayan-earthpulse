"""CDM - Cryospheric Dynamics Model -> GMES (Glacier Mobility and Evolution Signature).

Two independent NISAR velocity estimates, cross-checked with each other and with ITS_LIVE:
  A. GOFF stack: NISAR's own 12/24-day pixel offsets (slant-range + along-track), quality-filtered, projected to
     horizontal east/north with the surface-parallel-flow assumption (DEM slope), robust median over all pairs
     -> mean velocity, monthly velocity (seasonality), dispersion-based uncertainty.
  B. Long-baseline feature tracking (our own): normalised cross-correlation of speckle-filtered GCOV HH images of
     the same track/frame months apart. Same map grid, so offsets are directly east/north metres.
Plus: glacier mask (RGI 7), upper/middle/lower zone statistics, centre-line velocity profile, terminus note.
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
from earthpulse.config import WORK_DIR, Region
from earthpulse.geo import ancillary
from earthpulse.geo.area import masked_area_km2, pixel_area_m2
from earthpulse.io.nisar import Subset, cache_path, fetch_many, interp_cube
from earthpulse.preprocess.speckle import refined_lee, to_db
from earthpulse.tracking.ncc import filter_offsets, track
from earthpulse.validation import itslive
from earthpulse.validation.metrics import continuous

log = logging.getLogger(__name__)
DAYS = 365.25


def _date(iso: str) -> np.datetime64:
    return np.datetime64(iso[:10], "D")


# ------------------------------------------------------------------------------------------------ masks
def glacier_mask(region: Region, epsg: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """Returns (all-glacier mask, main-glacier mask, outlines). Main glacier = largest RGI outline in the AOI."""
    outlines = ancillary.glacier_outlines(region.bbox)
    if outlines:
        allm = ancillary.rasterize_geoms([g for _, _, g in outlines], epsg, x, y)
        main = max(outlines, key=lambda o: o[2].intersection(region.geometry).area)
        mainm = ancillary.rasterize_geoms([main[2]], epsg, x, y)
        return allm, mainm, outlines
    # Fallback 1: ITS_LIVE land-ice mask (derived from RGI), public COG
    from earthpulse.validation.itslive import landice
    li = landice(epsg, x, y, (region.geometry.centroid.x, region.geometry.centroid.y))
    if li is not None and li.sum() > 50:
        allm = li
        lab, n = ndimage.label(allm)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        return allm, lab == np.argmax(sizes), [("its_live_landice", "", None)]
    wc = ancillary.worldcover(epsg, x, y)
    allm = wc == 70
    lab, n = ndimage.label(allm)
    if n == 0:
        return allm, allm, []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return allm, lab == np.argmax(sizes), []


# ------------------------------------------------------------------------------------------------ GOFF
def goff_velocity(region: Region, key: str, members) -> dict[str, Any] | None:
    subs = [Subset.load(p) for g in members if (p := cache_path(region.id, "GOFF", g.granule_id)).exists()]
    if len(subs) < 3:
        return None
    gp = cache_path(region.id, "GOFF", members[0].granule_id, geometry=True)
    geom = Subset.load(gp) if gp.exists() else None
    if geom is None or "cube_alongTrackUnitVectorX" not in geom.layers:
        log.warning("GOFF %s: geometry missing", key)
        return None
    x, y, epsg = subs[0].x, subs[0].y, subs[0].epsg
    elev = ancillary.dem(epsg, x, y)
    dx, dy = x[1] - x[0], y[1] - y[0]
    gy, gx = np.gradient(np.where(np.isfinite(elev), elev, np.nanmedian(elev)), abs(dy), abs(dx))
    sx, sy = gx, (-gy if dy < 0 else gy)  # dh/dE, dh/dN
    a = geom.attrs
    def cube(n):
        return interp_cube(geom.layers[f"cube_{n}"], a["heights"], a["cube_x"], a["cube_y"], x, y, elev)
    lE, lN = cube("losUnitVectorX"), cube("losUnitVectorY")
    lU = np.sqrt(np.clip(1 - lE ** 2 - lN ** 2, 0, 1))
    aE, aN = cube("alongTrackUnitVectorX"), cube("alongTrackUnitVectorY")
    # Observations (metres): slant-range offset dr = -(l . d) [range increase = motion away from sensor];
    # along-track offset da = a . d.   With dU = sx dE + sy dN (surface-parallel flow):
    m11, m12 = -(lE + lU * sx), -(lN + lU * sy)
    m21, m22 = aE, aN
    det = m11 * m22 - m12 * m21
    vel_e, vel_n, mids, ids = [], [], [], []
    for s in subs:
        info = parse_granule_name(s.granule_id)
        t0, t1 = _date(info["ref_time"]), _date(info["sec_time"])
        dt_yr = float((t1 - t0).astype(int)) / DAYS
        if s.shape != subs[0].shape:
            continue
        dr = s.layers["slantRangeOffset"].astype(float)
        da = s.layers["alongTrackOffset"].astype(float)
        good = np.isfinite(dr) & np.isfinite(da)
        if "correlationSurfacePeak" in s.layers:
            good &= s.layers["correlationSurfacePeak"] >= 0.15
        if "snr" in s.layers:
            good &= s.layers["snr"] >= 4
        # remove per-pair bias using off-glacier (stable) ground later; here keep raw
        with np.errstate(invalid="ignore", divide="ignore"):
            dE = (m22 * dr - m12 * da) / det
            dN = (-m21 * dr + m11 * da) / det
        dE[~good | (np.abs(det) < 0.05)] = np.nan
        dN[~good | (np.abs(det) < 0.05)] = np.nan
        vel_e.append(dE / dt_yr)
        vel_n.append(dN / dt_yr)
        mids.append(t0 + (t1 - t0) // 2)
        ids.append(s.granule_id)
    VE, VN = np.stack(vel_e), np.stack(vel_n)
    return {"x": x, "y": y, "epsg": epsg, "elev": elev, "ve": VE, "vn": VN, "mid": np.array(mids), "ids": ids,
            "key": key, "units": s.attrs.get("offset_units", "m")}


def _debias_and_stack(r: dict[str, Any], stable: np.ndarray) -> dict[str, np.ndarray]:
    VE, VN = r["ve"].copy(), r["vn"].copy()
    for k in range(VE.shape[0]):  # remove per-pair constant bias from stable ground
        VE[k] -= np.nanmedian(VE[k][stable]) if np.isfinite(VE[k][stable]).sum() > 20 else 0
        VN[k] -= np.nanmedian(VN[k][stable]) if np.isfinite(VN[k][stable]).sum() > 20 else 0
    ve = np.nanmedian(VE, 0)
    vn = np.nanmedian(VN, 0)
    n = np.sum(np.isfinite(VE), 0)
    mad = 1.4826 * np.nanmedian(np.hypot(VE - ve, VN - vn), 0)
    sigma = 1.253 * mad / np.sqrt(np.maximum(n, 1))  # std error of the median
    speed = np.hypot(ve, vn)
    speed[n < 3] = np.nan
    months = r["mid"].astype("datetime64[M]")
    monthly = {}
    for m in np.unique(months):
        sel = months == m
        if sel.sum() >= 1:
            monthly[str(m)] = np.hypot(np.nanmedian(VE[sel], 0), np.nanmedian(VN[sel], 0))
    return {"ve": ve, "vn": vn, "speed": speed, "sigma": sigma, "n": n, "monthly": monthly}


# ------------------------------------------------------------------------------------------------ NCC on GCOV
def gcov_tracking(region: Region, gs, stable_fn) -> dict[str, Any] | None:
    stacks = select_stacks(gs, "GCOV")
    if not stacks:
        return None
    key, members = max(stacks.items(), key=lambda kv: len(kv[1]))
    members = [g for g in members if g.coverage >= 0.95] or members
    # the longest baseline pair (first vs last) + two ~6-month pairs for seasonal context
    pick_idx = sorted({0, len(members) // 2, len(members) - 1})
    picks = [members[i] for i in pick_idx]
    subs = fetch_many(region, [g.to_json() for g in picks], workers=4)
    if len(subs) < 2:
        return None
    imgs, dates = [], []
    for s in subs:
        hh = s.layers.get("HHHH")
        if hh is None:
            continue
        hh = hh.astype(float)
        hh[(hh <= 0) | ~np.isfinite(hh)] = np.nan
        # 2x2 average to 20 m: speckle reduction, faster tracking
        H2, W2 = hh.shape[0] // 2 * 2, hh.shape[1] // 2 * 2
        h2 = np.nanmean(hh[:H2, :W2].reshape(H2 // 2, 2, W2 // 2, 2), axis=(1, 3))
        imgs.append(to_db(refined_lee(h2, enl=8.0)))
        dates.append(_date(parse_granule_name(s.granule_id)["ref_time"]))
    x = subs[0].x[: imgs[0].shape[1] * 2].reshape(-1, 2).mean(1)
    y = subs[0].y[: imgs[0].shape[0] * 2].reshape(-1, 2).mean(1)
    stable, glacier = stable_fn(subs[0].epsg, x, y)
    results = []
    for i, j in [(0, len(imgs) - 1)] + ([(0, 1), (1, len(imgs) - 1)] if len(imgs) == 3 else []):
        dt_yr = float((dates[j] - dates[i]).astype(int)) / DAYS
        if dt_yr <= 0.05:
            continue
        search = int(np.clip(np.ceil(90 * dt_yr / 20) + 3, 4, 12))  # allow up to ~90 m/yr at 20 m pixels
        f = filter_offsets(track(imgs[i], imgs[j], chip=24, search=search, step=4))
        px = abs(x[1] - x[0])
        de = f.dx * px  # column -> east
        dn = -f.dy * px  # row increases southward
        st = stable[np.ix_(f.rows, f.cols)]
        gl = glacier[np.ix_(f.rows, f.cols)]
        if np.isfinite(de[st]).sum() > 20:
            de -= np.nanmedian(de[st])
            dn -= np.nanmedian(dn[st])
        stable_noise = float(np.nanstd(np.hypot(de[st], dn[st]))) if np.isfinite(de[st]).any() else np.nan
        results.append({"i": i, "j": j, "dates": [str(dates[i]), str(dates[j])], "dt_yr": dt_yr,
                        "ve": de / dt_yr, "vn": dn / dt_yr, "peak": f.peak, "rows": f.rows, "cols": f.cols,
                        "x": x[f.cols], "y": y[f.rows], "stable_noise_m": stable_noise,
                        "glacier": gl})
    bs = [{"date": str(d), "glacier_hh_db_median": float(np.nanmedian(im[glacier])) if glacier.any() else None,
           "stable_hh_db_median": float(np.nanmedian(im[stable])) if stable.any() else None} for d, im in zip(dates, imgs)]
    return {"stack": key, "granules": [s.granule_id for s in subs], "pairs": results, "epsg": subs[0].epsg,
            "backscatter": bs}


# ------------------------------------------------------------------------------------------------ run
def _profile(speed: np.ndarray, elev: np.ndarray, main: np.ndarray, x, y, band: float = 50.0):
    """Centre-line approximation: centroid of the main glacier in each elevation band, top -> terminus."""
    pts = []
    if main.sum() < 10:
        return pts
    e = np.where(main, elev, np.nan)
    lo, hi = np.nanpercentile(e, 1), np.nanpercentile(e, 99)
    yy, xx = np.mgrid[0:len(y), 0:len(x)]
    dist = 0.0
    prev = None
    for b in np.arange(hi, lo, -band):
        sel = main & (elev <= b) & (elev > b - band)
        if sel.sum() < 3:
            continue
        cy, cx = int(np.round(yy[sel].mean())), int(np.round(xx[sel].mean()))
        win = speed[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3]
        px, py = float(x[cx]), float(y[cy])
        if prev is not None:
            dist += float(np.hypot(px - prev[0], py - prev[1]))
        prev = (px, py)
        pts.append({"elev_m": float(b - band / 2), "dist_m": round(dist, 1), "x": px, "y": py,
                    "speed_m_yr": float(np.nanmedian(win)) if np.isfinite(win).any() else None})
    return pts


def run_cdm(region: Region) -> dict[str, Any]:
    gs = region_granules(region)
    out_dir = WORK_DIR / region.id / "cdm"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"region": region.id, "model": "CDM", "signature": "GMES", "limitations": [
        "Surface velocity is not mass balance; no ice-thickness or elevation-change inference is made.",
        "GOFF projection to horizontal velocity assumes flow parallel to the (Copernicus DEM) surface.",
        "Feature tracking needs visible surface texture; fresh snow or melt can decorrelate the images.",
        "ITS_LIVE is a 2014-2022 composite, so differences can include real interannual change.",
        "Terminus retreat over one year (~10-20 m at Himalayan glaciers) is near the 10-20 m pixel size.",
    ]}
    stacks = select_stacks(gs, "GOFF")
    goff = None
    for key, members in stacks.items():
        r = goff_velocity(region, key, members)
        if r is not None and (goff is None or len(r["ids"]) > len(goff["ids"])):
            goff = r
    def masks(epsg, x, y):
        allm, main, _ = glacier_mask(region, epsg, x, y)
        elev = ancillary.dem(epsg, x, y)
        slope, _ = ancillary.slope_aspect(elev, x[1] - x[0], y[1] - y[0])
        wc = ancillary.worldcover(epsg, x, y)
        buf = ndimage.binary_dilation(allm, iterations=3)
        stable = ~buf & (slope < 25) & (wc != 80) & (wc != 70) & np.isfinite(elev)
        return stable, allm
    if goff is not None:
        x, y, epsg = goff["x"], goff["y"], goff["epsg"]
        allm, main, outlines = glacier_mask(region, epsg, x, y)
        stable, _ = masks(epsg, x, y)
        from earthpulse.preprocess import masks as pm
        geom_g = pm.load_geometry(region.id, "GOFF", stacks[goff["key"]])
        rm = pm.radar_masks(geom_g, x, y, goff["elev"])
        bad_geom = rm["layover"] | rm["shadow"]
        goff["ve"][:, bad_geom] = np.nan
        goff["vn"][:, bad_geom] = np.nan
        st = _debias_and_stack(goff, stable)
        conf = np.zeros(st["speed"].shape, np.float32)  # 3 high, 2 medium, 1 low, 0 not evaluated
        fin = np.isfinite(st["speed"])
        conf[fin] = 1
        conf[fin & (st["n"] >= 5) & (st["sigma"] < 8)] = 2
        conf[fin & (st["n"] >= 10) & (st["sigma"] < 3)] = 3
        snow = pm.snow_dates(region, goff["mid"])
        area = pixel_area_m2(epsg, x, y)
        elev = goff["elev"]
        # zones of the main glacier by elevation thirds (upper/middle/lower)
        zones = {}
        if main.any():
            e = elev[main]
            t1, t2 = np.nanpercentile(e, [33.3, 66.7])
            for name, m in (("upper", main & (elev >= t2)), ("middle", main & (elev >= t1) & (elev < t2)),
                            ("lower", main & (elev < t1))):
                v = st["speed"][m]
                zones[name] = {"median_m_yr": float(np.nanmedian(v)) if np.isfinite(v).any() else None,
                               "p90_m_yr": float(np.nanpercentile(v[np.isfinite(v)], 90)) if np.isfinite(v).any() else None,
                               "coverage": float(np.isfinite(v).mean()) if v.size else 0.0}
        flow_dir = np.degrees(np.arctan2(st["ve"], st["vn"])) % 360  # azimuth clockwise from north
        lon_c, lat_c = region.geometry.centroid.x, region.geometry.centroid.y
        il = itslive.velocity(epsg, x, y, (lon_c, lat_c))
        cmp_mask = main & np.isfinite(st["speed"]) & np.isfinite(il["v"])
        val = continuous(st["speed"][cmp_mask], il["v"][cmp_mask])
        stable_v = st["speed"][stable]
        monthly_main = {m: (float(np.nanmedian(v[main])) if np.isfinite(v[main]).any() else None)
                        for m, v in st["monthly"].items()}
        # glacier coherence from NISAR GUNW (decorrelates over fast ice, melt and fresh snow)
        from earthpulse.preprocess.coherence import coherence_stack, masked_series
        coh = coherence_stack(region, x, y, epsg)
        coh_npz, coh_sum = {}, None
        if coh is not None:
            C = coh["coherence"]
            coh_mean = np.nanmean(C, 0)
            stable_c = stable & ~allm
            fast = main & (st["speed"] > np.nanpercentile(st["speed"][main], 75)) if np.isfinite(st["speed"][main]).any() else main
            from scipy.stats import spearmanr
            okc = main & np.isfinite(coh_mean) & np.isfinite(st["speed"])
            rho = spearmanr(coh_mean[okc], st["speed"][okc])[0] if okc.sum() > 30 else None
            coh_sum = {"stack": coh["stack"], "pairs": coh["pairs"], "mid_dates": [str(d) for d in coh["mid_dates"]],
                       "temporal_baseline_days": coh["temporal_baseline_days"],
                       "glacier_median": masked_series(C, main), "fast_ice_median": masked_series(C, fast),
                       "stable_median": masked_series(C, stable_c),
                       "glacier_mean_coherence": float(np.nanmedian(coh_mean[main])) if main.any() else None,
                       "stable_mean_coherence": float(np.nanmedian(coh_mean[stable_c])) if stable_c.any() else None,
                       "spearman_coherence_vs_speed": None if rho is None or not np.isfinite(rho) else round(float(rho), 3),
                       "note": "12-day L-band coherence (GUNW, 80 m). Ice moving metres per pair, melt and fresh snow "
                               "decorrelate the signal, so low glacier coherence vs stable rock is expected; a negative "
                               "link with GOFF speed is an independent consistency check on the flow map.",
                       "granules": coh["granules"]}
            coh_npz = {"coherence_mean": coh_mean.astype(np.float32)}
        np.savez_compressed(out_dir / "goff_velocity.npz", **coh_npz, x=x, y=y, epsg=epsg, ve=st["ve"].astype(np.float32),
                            vn=st["vn"].astype(np.float32), speed=st["speed"].astype(np.float32),
                            sigma=st["sigma"].astype(np.float32), n=st["n"], flow_dir=flow_dir.astype(np.float32),
                            glacier=allm, main=main, stable=stable, elev=elev.astype(np.float32),
                            itslive_v=il["v"].astype(np.float32),
                            monthly_keys=np.array(list(st["monthly"].keys())),
                            monthly=np.stack(list(st["monthly"].values())).astype(np.float32),
                            confidence_class=conf, layover=rm["layover"], shadow=rm["shadow"])
        # terminus: lowest 2% of the main glacier by elevation (static outline; retreat over 1 yr is below resolution)
        term = None
        if main.any():
            e_main = np.where(main, elev, np.nan)
            low = main & (e_main <= np.nanpercentile(e_main, 2))
            rr, cc = np.nonzero(low)
            from pyproj import Transformer
            lon_t, lat_t = Transformer.from_crs(epsg, 4326, always_xy=True).transform(float(x[cc].mean()), float(y[rr].mean()))
            term = {"lon": round(lon_t, 5), "lat": round(lat_t, 5), "elevation_m": round(float(np.nanmin(e_main)), 0),
                    "speed_m_yr": float(np.nanmedian(st["speed"][low])) if np.isfinite(st["speed"][low]).any() else None,
                    "note": "Position of the lowest part of the glacier outline (ITS_LIVE/RGI, static). Terminus displacement "
                            "over one year (~10-20 m) is below the 20-80 m resolution and is not claimed."}
        summary["goff"] = {
            "stack": goff["key"], "n_pairs": len(goff["ids"]), "granules": goff["ids"],
            "date_range": [str(goff["mid"].min()), str(goff["mid"].max())],
            "glacier_area_km2": round(masked_area_km2(allm, area), 2),
            "main_glacier_area_km2": round(masked_area_km2(main, area), 2),
            "main_glacier": None,
            "speed_main_median_m_yr": float(np.nanmedian(st["speed"][main])) if np.isfinite(st["speed"][main]).any() else None,
            "speed_main_p95_m_yr": float(np.nanpercentile(st["speed"][main & np.isfinite(st["speed"])], 95))
            if (main & np.isfinite(st["speed"])).any() else None,
            "stable_ground_median_m_yr": float(np.nanmedian(stable_v)) if np.isfinite(stable_v).any() else None,
            "zones": zones, "monthly_main_median_m_yr": monthly_main,
            "profile": _profile(st["speed"], elev, main, x, y),
            "validation_itslive": {**val, "source": il["source"]},
            "terminus": term, "coherence": coh_sum,
            "snow_persistence": {"fraction_of_pair_dates_snow_covered": round(float(np.mean(snow["snow"])), 3), **snow},
            "radar_masks": {"layover_pct": round(100 * float(rm["layover"].mean()), 2),
                            "shadow_pct": round(100 * float(rm["shadow"].mean()), 2), "available": rm["available"]},
            "confidence_km2": {lvl: round(masked_area_km2(main & (conf == i), area), 2)
                               for i, lvl in ((3, "high"), (2, "medium"), (1, "low"))},
            "outline_source": ("ITS_LIVE land-ice mask (from RGI)" if outlines and outlines[0][0] == "its_live_landice"
                               else "RGI 7.0" if outlines else "ESA WorldCover snow/ice"),
        }
    # independent method B
    def stable_fn(epsg, x, y):
        return masks(epsg, x, y)
    ncc = gcov_tracking(region, gs, stable_fn)
    if ncc is not None and ncc["pairs"]:
        main_pair = ncc["pairs"][0]
        spd = np.hypot(main_pair["ve"], main_pair["vn"])
        np.savez_compressed(out_dir / "ncc_velocity.npz", x=main_pair["x"], y=main_pair["y"], epsg=ncc["epsg"],
                            ve=main_pair["ve"].astype(np.float32), vn=main_pair["vn"].astype(np.float32),
                            speed=spd.astype(np.float32), peak=main_pair["peak"].astype(np.float32),
                            glacier=main_pair["glacier"])
        entry = {"stack": ncc["stack"], "granules": ncc["granules"], "pairs": [
            {"dates": p["dates"], "dt_yr": round(p["dt_yr"], 3), "stable_noise_m": p["stable_noise_m"],
             "glacier_median_m_yr": float(np.nanmedian(np.hypot(p["ve"], p["vn"])[p["glacier"]]))
             if np.isfinite(np.hypot(p["ve"], p["vn"])[p["glacier"]]).any() else None,
             "coverage_glacier": float(np.isfinite(p["ve"][p["glacier"]]).mean()) if p["glacier"].any() else 0.0}
            for p in ncc["pairs"]]}
        if goff is not None:  # cross-method agreement on the NCC grid
            gx = np.searchsorted(goff["x"], main_pair["x"]).clip(0, len(goff["x"]) - 1)
            gy = np.searchsorted(-goff["y"], -main_pair["y"]).clip(0, len(goff["y"]) - 1)
            gsp = np.load(out_dir / "goff_velocity.npz")["speed"][np.ix_(gy, gx)]
            entry["agreement_with_goff"] = continuous(spd[main_pair["glacier"]], gsp[main_pair["glacier"]])
        entry["backscatter_variation"] = ncc.get("backscatter")
        summary["ncc"] = entry
    summary["generated"] = datetime.utcnow().isoformat() + "Z"
    if goff is not None:
        from earthpulse.preprocess import masks as pm
        members = [g for g in gs if g.granule_id in set(goff["ids"]) | set((ncc or {}).get("granules", []))]
        pm.write_processing_log(
            region, "cdm", members,
            params={"goff_filters": "correlation peak >= 0.15, SNR >= 4, layover/shadow masked", "goff_projection":
                    "LOS + along-track unit vectors, surface-parallel flow (DEM)", "stacking": "per-pair stable-ground "
                    "debias, robust median, monthly medians", "ncc": "Refined Lee (ENL 8) on 2x2 multilooked GCOV HH dB, "
                    "chip 24 px, step 4 px, Gaussian sub-pixel peak, outlier median filter", "glacier_mask":
                    "ITS_LIVE land-ice (RGI-derived)"},
            ancillary=["Copernicus DEM GLO-30", "ESA WorldCover 2021", "ITS_LIVE v2 static mosaic", "ERA5-Land snow depth"],
            qc={"stable_ground_median_m_yr": summary.get("goff", {}).get("stable_ground_median_m_yr"),
                "radar_masks": summary.get("goff", {}).get("radar_masks")},
            outputs=["goff_velocity.npz", "ncc_velocity.npz", "summary.json"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary
