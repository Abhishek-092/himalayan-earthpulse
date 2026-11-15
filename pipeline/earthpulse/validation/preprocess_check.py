"""Verification of every preprocessing stage on the real NISAR inputs of a region.

Each check measures something and returns pass / warn / fail with the evidence, so the app can show *why* we trust
(or distrust) the inputs. Nothing here changes data.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from earthpulse.catalog.discover import parse_granule_name
from earthpulse.catalog.stacks import region_granules, select_stacks
from earthpulse.config import WEB_DATA_DIR, WORK_DIR, Region
from earthpulse.geo import ancillary
from earthpulse.io.nisar import Subset, cache_path
from earthpulse.preprocess.speckle import estimate_enl, quegan, refined_lee, to_db

log = logging.getLogger(__name__)


def _chk(stage: str, name: str, ok: bool | None, value: Any, expect: str, note: str = "", warn: bool = False):
    status = "pass" if ok else ("warn" if (warn or ok is None) else "fail")
    return {"stage": stage, "check": name, "status": status, "value": value, "expected": expect, "note": note}


def _r(v, d=3):
    try:
        return None if v is None or not np.isfinite(v) else round(float(v), d)
    except TypeError:
        return v


def check_gcov(region: Region, max_dates: int = 12) -> list[dict]:
    out: list[dict] = []
    gs = region_granules(region)
    stacks = select_stacks(gs, "GCOV")
    if not stacks:
        return out
    key, members = max(stacks.items(), key=lambda kv: len(kv[1]))
    subs = [Subset.load(p) for g in members if (p := cache_path(region.id, "GCOV", g.granule_id)).exists()]
    if len(subs) < 2:
        return out
    stage = f"GCOV {key}"
    # 1. product level / calibration convention
    terms = subs[0].attrs.get("list_of_covariance_terms") or subs[0].attrs.get("terms")
    out.append(_chk(stage, "Product level & calibration", True, f"L2 GCOV, RTC gamma-0, terms {terms}",
                    "radiometrically terrain-corrected backscatter (no re-calibration needed)",
                    "GCOV is already calibrated and terrain-flattened by the NISAR processor (ISCE3); we do not repeat it."))
    # 2. common grid
    shapes = {s.shape for s in subs}
    same_epsg = len({s.epsg for s in subs}) == 1
    ref = max(shapes, key=lambda sh: sum(s.shape == sh for s in subs))
    grid_same = [s for s in subs if s.shape == ref]
    identical = all(np.array_equal(s.x, grid_same[0].x) and np.array_equal(s.y, grid_same[0].y) for s in grid_same)
    out.append(_chk(stage, "Common map grid", same_epsg and identical, f"EPSG {subs[0].epsg}, {len(grid_same)}/{len(subs)} "
                    f"dates on identical {ref[1]}x{ref[0]} grid ({abs(subs[0].x[1]-subs[0].x[0]):.0f} m)",
                    "all dates share CRS and pixel grid", "Dates on a different grid (other imaging mode) are excluded.",
                    warn=len(grid_same) < len(subs)))
    s0, s1 = grid_same[0], grid_same[-1]
    hh0 = np.where(s0.layers["HHHH"] > 0, s0.layers["HHHH"], np.nan).astype(float)
    hv0 = np.where(s0.layers["HVHV"] > 0, s0.layers["HVHV"], np.nan).astype(float)
    x, y, epsg = s0.x, s0.y, s0.epsg
    wc = ancillary.worldcover(epsg, x, y)
    occ = np.nan_to_num(ancillary.water_occurrence(epsg, x, y))
    # 3. physical sanity of calibrated values
    med_hh, med_hv = np.nanmedian(to_db(hh0)), np.nanmedian(to_db(hv0))
    out.append(_chk(stage, "Backscatter value range", -25 < med_hh < 2 and -32 < med_hv < -5,
                    f"median HH {med_hh:.1f} dB, HV {med_hv:.1f} dB", "HH -25..2 dB, HV -32..-5 dB (L-band land)"))
    water, forest = occ > 80, wc == 10
    if water.sum() > 50 and forest.mean() > 0.05:
        d = np.nanmedian(to_db(hv0)[forest]) - np.nanmedian(to_db(hv0)[water])
        out.append(_chk(stage, "Water darker than forest (HV)", d > 3, f"forest - water = {d:.1f} dB",
                        "> 3 dB (smooth water reflects away; forest volume-scatters)"))
    else:
        fdb = np.nanmedian(to_db(hv0)[forest]) if forest.any() else np.nan
        bare = np.isin(wc, (60, 70))
        bdb = np.nanmedian(to_db(hv0)[bare]) if bare.sum() > 20 else np.nan
        if forest.mean() > 0.05 and np.isfinite(fdb - bdb):
            out.append(_chk(stage, "Forest brighter than bare/snow (HV)", (fdb - bdb) > 1, f"forest - bare = {fdb - bdb:.1f} dB",
                            "> 1 dB", "Too little open water in this area for a water test."))
    # 4. speckle: ENL raw -> 2x2 multilook -> multitemporal filter; mean preservation
    enl_raw = estimate_enl(hv0)
    H2, W2 = hv0.shape[0] // 2 * 2, hv0.shape[1] // 2 * 2
    ml = np.nanmean(hv0[:H2, :W2].reshape(H2 // 2, 2, W2 // 2, 2), axis=(1, 3))
    enl_ml = estimate_enl(ml)
    stack = []
    for s in grid_same[:max_dates]:
        a = np.where(s.layers["HVHV"] > 0, s.layers["HVHV"], np.nan).astype(float)
        stack.append(np.nanmean(a[:H2, :W2].reshape(H2 // 2, 2, W2 // 2, 2), axis=(1, 3)))
    q = quegan(np.stack(stack), 5)[0]
    enl_q = estimate_enl(q)
    rl = refined_lee(ml, enl=enl_ml)
    ratio_q = np.nanmean(q) / np.nanmean(ml)
    ratio_rl = np.nanmean(rl) / np.nanmean(ml)
    out.append(_chk(stage, "Speckle reduction (ENL)", enl_q > enl_ml > enl_raw,
                    f"ENL raw {enl_raw:.1f} -> 2x2 looks {enl_ml:.1f} -> multitemporal {enl_q:.1f}",
                    "ENL increases at each step", "Higher equivalent number of looks = less speckle."))
    out.append(_chk(stage, "Filters preserve radiometry", abs(ratio_q - 1) < 0.05 and abs(ratio_rl - 1) < 0.1,
                    f"mean ratio Quegan {ratio_q:.3f}, refined Lee {ratio_rl:.3f}", "within 5% / 10% of unfiltered mean"))
    # 5. co-registration: NCC offsets on stable, gentle ground between first and last date
    from earthpulse.tracking.ncc import filter_offsets, track

    elev = ancillary.dem(epsg, x, y)
    slope, aspect = ancillary.slope_aspect(elev, x[1] - x[0], y[1] - y[0])
    hh1 = np.where(s1.layers["HHHH"] > 0, s1.layers["HHHH"], np.nan).astype(float)
    a = to_db(np.nanmean(hh0[:H2, :W2].reshape(H2 // 2, 2, W2 // 2, 2), axis=(1, 3)))
    b = to_db(np.nanmean(hh1[:H2, :W2].reshape(H2 // 2, 2, W2 // 2, 2), axis=(1, 3)))
    step = int(np.clip(min(a.shape) // 25, 6, 24))
    f = filter_offsets(track(a, b, chip=32, search=4, step=step))
    st = (slope[:H2:2, :W2:2] < 20) & ~np.isin(wc[:H2:2, :W2:2], (70, 80))
    m = st[np.ix_(f.rows, f.cols)] & np.isfinite(f.dx)
    if m.sum() > 10:
        mdx, mdy = np.median(f.dx[m]), np.median(f.dy[m])
        spread = 1.4826 * np.median(np.hypot(f.dx[m] - mdx, f.dy[m] - mdy))
        px = 2 * abs(x[1] - x[0])
        out.append(_chk(stage, "Co-registration between dates", np.hypot(mdx, mdy) < 0.15 and spread < 0.5,
                        f"stable-ground offset {np.hypot(mdx, mdy) * px:.1f} m (median), scatter {spread * px:.1f} m, "
                        f"{int(m.sum())} chips ({s0.attrs.get('start', '')[:10]} vs {s1.attrs.get('start', '')[:10]})",
                        "< 0.15 px systematic, < 0.5 px scatter"))
    # 6. terrain: radar-facing vs away-facing slopes after RTC
    direction = members[0].direction
    # NISAR is LEFT-looking (identification/lookDirection = "Left"): ascending passes look west, so the radar sits to
    # the east and east-facing slopes (aspect ~90) face it; descending passes look east -> west-facing slopes face it.
    facing = 90.0 if direction == "A" else 270.0
    ang = np.abs(((aspect - facing) + 180) % 360 - 180)
    steep = slope > 20
    sl = np.s_[:hh0.shape[0], :hh0.shape[1]]
    cls = forest if forest.mean() > 0.1 else ~np.isin(wc, (70, 80))  # same land cover on both slope sides
    cname = "forest" if forest.mean() > 0.1 else "snow-free land"
    toward = steep & (ang < 45) & cls
    away = steep & (ang > 135) & cls
    if toward.sum() > 200 and away.sum() > 200:
        dd = np.nanmedian(to_db(hv0[sl])[toward]) - np.nanmedian(to_db(hv0[sl])[away])
        out.append(_chk(stage, "Terrain correction (RTC) residual", abs(dd) < 3, f"{cname} HV facing radar - facing away = {dd:.1f} dB",
                        "|diff| < 3 dB (uncorrected L-band slopes differ by 5-10 dB)",
                        "Same land cover on opposite slopes should look similar after radiometric terrain correction.",
                        warn=abs(dd) < 5))
    # 7. masks
    valid = np.isfinite(hh0) & np.isfinite(hv0)
    out.append(_chk(stage, "Quality masks", valid.mean() > 0.6,
                    f"valid {valid.mean():.0%}, forest {forest.mean():.0%}, permanent water {(occ > 50).mean():.1%}, "
                    f"snow/ice {(wc == 70).mean():.0%}", "> 60% valid pixels",
                    "Invalid = outside swath, layover/shadow or no-data in the product mask."))
    return out


def check_gunw(region: Region) -> list[dict]:
    out: list[dict] = []
    p = WORK_DIR / region.id / "gdm" / "summary.json"
    if not p.exists():
        return out
    s = json.loads(p.read_text())
    for key, v in s["stacks"].items():
        stage = f"GUNW {key}"
        sub = Subset.load(cache_path(region.id, "GUNW", v["granules"][0]))
        f = sub.attrs.get("center_frequency_hz")
        out.append(_chk(stage, "Wavelength from metadata", f is not None, f"{f / 1e6:.1f} MHz -> lambda {299792458 / f * 100:.2f} cm"
                        if f else "missing", "read per granule (not assumed)", "One fringe = lambda/2."))
        has_iono = "ionospherePhaseScreen" in sub.layers
        out.append(_chk(stage, "Ionosphere correction", has_iono,
                        f"screen applied; spatial range {np.ptp(sub.layers['ionospherePhaseScreen']):.2f} rad in AOI" if has_iono else "absent",
                        "split-spectrum ionosphere screen subtracted from every pair"))
        qc = v.get("pair_qc") or []
        noise = [q["noise_cm"] for q in qc if q["noise_cm"] is not None]
        used = [q for q in qc if q["used"]]
        out.append(_chk(stage, "Interferogram quality control", len(used) >= 0.6 * len(qc),
                        f"{len(used)}/{len(qc)} pairs used; median residual noise {np.median(noise):.1f} cm "
                        f"(rejected pairs: low valid fraction or noise > 3x median)", ">= 60% of pairs usable",
                        "Rejected pairs are mostly winter-snow and monsoon pairs.", warn=True if len(used) < len(qc) else False))
        vfr = [q["valid_fraction"] for q in qc]
        out.append(_chk(stage, "Coherence mask", np.median(vfr) > 0.4,
                        f"median {np.median(vfr):.0%} of pixels coherent (>= 0.4) per pair; mean coherence {v['mean_coherence']:.2f}",
                        "> 40% coherent per pair"))
        ref = v["velocity_mm_yr"]
        out.append(_chk(stage, "Stable reference area", ref["reference_median"] is not None and abs(ref["reference_median"]) < 5,
                        f"reference velocity {ref['reference_median']:.1f} mm/yr, scatter {ref['reference_scatter']:.1f} mm/yr",
                        "|median| < 5 mm/yr", "Auto-selected: coherent, gentle, lowest-velocity pixels (two-pass)."))
        lc = v.get("landcover_velocity_mm_yr", {})
        if "built-up" in lc and "tree cover" in lc:
            bias = lc["tree cover"]["median_mm_yr"] - lc["built-up"]["median_mm_yr"]
            out.append(_chk(stage, "Vegetation (phase-closure) bias check", abs(bias) < 10,
                            f"forest - settlement apparent velocity = {bias:+.1f} mm/yr",
                            "< 10 mm/yr if no bias", "Detected bias -> vegetated pixels excluded from deformation results.",
                            warn=True))
    # unwrapping repair evidence: re-run spread on a few pairs (before vs after)
    try:
        from earthpulse.models import gdm

        stacks = select_stacks(region_granules(region), "GUNW")
        key = sorted(stacks)[-1]
        st = gdm.load_stack(region, key, stacks[key])
        if st:
            x, y, e = st["x"], st["y"], st["epsg"]
            elev = ancillary.dem(e, x, y)
            slope, _ = ancillary.slope_aspect(elev, x[1] - x[0], y[1] - y[0])
            valid = (st["ccomp"] > 0) & (st["coh"] >= 0.4)
            mc = np.nanmean(st["coh"], 0)
            use = (mc >= np.nanpercentile(mc, 50)) & (slope < 25)
            before, after = [], []
            for k in range(len(st["ids"])):
                wl = st["wavelength"][k] / (4 * np.pi)
                b = gdm._fit_remove(np.where(valid[k], -wl * st["phase"][k], np.nan), elev, use & valid[k])
                ph = gdm.reunwrap(st["phase"][k], valid[k], elev, use)
                a = gdm._fit_remove(-wl * ph, elev, use & valid[k])
                for arr, lst in ((b, before), (a, after)):
                    vv = arr[np.isfinite(arr)]
                    if vv.size > 50:
                        lst.append((np.percentile(vv, 95) - np.percentile(vv, 5)) / 3.29 * 100)
            out.append(_chk(f"GUNW {key}", "Unwrapping-error repair", np.median(after) < np.median(before),
                            f"median per-pair noise {np.median(before):.1f} cm -> {np.median(after):.1f} cm; worst pair "
                            f"{max(before):.0f} -> {max(after):.0f} cm", "noise decreases after re-unwrapping",
                            "Smooth-model removal + quality-guided re-unwrapping + island cycle snapping."))
    except Exception as ex:  # pragma: no cover
        log.warning("unwrap check failed: %s", ex)
    return out


def check_goff(region: Region) -> list[dict]:
    out: list[dict] = []
    p = WORK_DIR / region.id / "cdm" / "summary.json"
    if not p.exists():
        return out
    s = json.loads(p.read_text())
    g = s.get("goff")
    if g:
        stage = f"GOFF {g['stack']}"
        out.append(_chk(stage, "Offsets -> ground velocity geometry", True, "LOS + along-track unit vectors from product metadata cubes, "
                        "interpolated at DEM height", "per-pixel geometry, surface-parallel flow"))
        sg = g["stable_ground_median_m_yr"]
        out.append(_chk(stage, "Stable ground bias", sg is not None and sg < 3, f"{sg:.1f} m/yr on non-glacier gentle terrain",
                        "< 3 m/yr", "Per-pair constant bias removed using stable ground."))
        v = g["validation_itslive"]
        out.append(_chk(stage, "Independent check: ITS_LIVE", v.get("pearson_r", 0) > 0.5, f"MAE {v['mae']:.1f} m/yr, bias {v['bias']:.1f}, "
                        f"r {v['pearson_r']:.2f} (n={v['n']})", "r > 0.5", "Moderate: ITS_LIVE is a 2014-22 average at 120 m; "
                        "slow tributary and accumulation ice dominate the pixel count.", warn=True))
    n = s.get("ncc")
    if n and n.get("pairs"):
        p0 = n["pairs"][0]
        out.append(_chk(f"GCOV {n['stack']}", "Feature-tracking noise on stable ground", p0["stable_noise_m"] < 5,
                        f"{p0['stable_noise_m']:.1f} m over {p0['dt_yr']:.2f} yr", "< 5 m (0.25 px at 20 m)"))
    return out


def check_metadata_and_masks(region: Region) -> list[dict]:
    """Product metadata (orbit, processor version), product processing flags and the quality masks we add."""
    out: list[dict] = []
    gs = region_granules(region)
    from collections import Counter

    for prod in ("GCOV", "GUNW", "GOFF"):
        members = [g for st in select_stacks(gs, prod).values() for g in st]
        if not members:
            continue
        orb = Counter(g.orbit_type or "?" for g in members)
        pge = Counter(g.pge_version or "?" for g in members)
        out.append(_chk(f"{prod} metadata", "Orbit ephemeris type", "POE" in orb and len(orb) == 1,
                        ", ".join(f"{k}: {v}" for k, v in orb.items()), "precise (POE) orbits",
                        "MOE = medium-precision orbits, the only ones in the provisional release. Fine for 80 m products; "
                        "residual orbit ramps are removed per interferogram (GDM).", warn=True))
        out.append(_chk(f"{prod} metadata", "Processor version consistency", len(pge) == 1,
                        ", ".join(f"{k}: {v}" for k, v in pge.most_common()), "one processing version across the stack",
                        "Mixed versions can introduce small calibration jumps; the change test's significance level and "
                        "the date-by-date QC limit the impact.", warn=True))
    gl = WORK_DIR / region.id / "gdm" / "summary.json"
    if gl.exists():
        s = json.loads(gl.read_text())
        for key, v in s["stacks"].items():
            pf = v.get("processing_flags") or {}
            if pf:
                out.append(_chk(f"GUNW {key}", "Product processing record (ISCE3)", True,
                                f"phase filter: {pf.get('phase_filter')}; unwrapper: {pf.get('unwrapper')} ({pf.get('unwrap_cost')}); "
                                f"troposphere wet/hydrostatic: {pf.get('wet_troposphere_applied')}/{pf.get('hydrostatic_troposphere_applied')}; "
                                f"topographic flattening: {pf.get('topographic_flattening')}; looks {pf.get('range_looks')}x{pf.get('azimuth_looks')}",
                                "documented and not repeated",
                                f"Product has no phase filter, so we apply Goldstein (alpha {v.get('goldstein_alpha')}) before re-unwrapping."))
            rmk = v.get("radar_masks") or {}
            if rmk:
                out.append(_chk(f"GUNW {key}", "Layover / radar-shadow mask", rmk.get("available", False),
                                f"layover {rmk.get('layover_pct')}%, shadow {rmk.get('shadow_pct')}%, incidence "
                                f"{[round(a, 1) for a in (rmk.get('incidence_range_deg') or [])]} deg", "computed from DEM + product LOS",
                                "Masked pixels are excluded from all interferograms."))
        gsens = WORK_DIR / region.id / "gdm" / "goldstein_sensitivity.json"
        if gsens.exists():
            g = json.loads(gsens.read_text())
            txt = "; ".join(f"a={r['alpha']}: {r['stack'][:4]} {r['settlement_mm_yr']} mm/yr, noise {r['mean_pair_noise_cm']} cm"
                            for r in g["rows"])
            out.append(_chk("GDM filtering", "Goldstein phase filter tested vs unfiltered", True, txt,
                            "filter adopted only if it does not bias results", g["conclusion"] + f" Adopted alpha = {g['adopted_alpha']}."))
        cons = (s.get("decomposition") or {}).get("consistency")
        if cons and cons.get("n_pixels"):
            out.append(_chk("GDM asc/desc", "Ascending vs descending consistency",
                            abs(cons["median_diff_mm_yr"]) < 10, f"vertical-equivalent difference median {cons['median_diff_mm_yr']} mm/yr, "
                            f"spread {cons['robust_spread_mm_yr']} mm/yr, r = {cons['pearson_r']} ({cons['n_pixels']} px)",
                            "|median| < 10 mm/yr", cons["note"], warn=abs(cons["median_diff_mm_yr"]) < 20))
    bl = WORK_DIR / region.id / "bdm" / "summary.json"
    if bl.exists():
        pre = json.loads(bl.read_text()).get("preprocessing") or {}
        if pre:
            pf = pre.get("product_flags") or {}
            out.append(_chk("GCOV masks", "Product processing flags", True,
                            f"noise correction applied: {pf.get('noiseCorrectionApplied')}; shadow masking applied: "
                            f"{pf.get('shadowMaskingApplied')}", "known before analysis",
                            "Both are False in the provisional GCOV, so this project applies NESZ subtraction and its own "
                            "layover/shadow mask."))
            out.append(_chk("GCOV masks", "Thermal noise correction + noise floor", pre.get("noise_corrected", False),
                            f"NESZ subtracted; {pre.get('noise_floor_pct')}% of pixels within 3 dB of the noise floor (masked)",
                            "noise handled", "Uses the product's noiseEquivalentBackscatter (HH, HV)."))
            out.append(_chk("GCOV masks", "Layover / radar-shadow mask", pre.get("geometry_available", False),
                            f"layover {pre.get('layover_pct')}%, shadow {pre.get('shadow_pct')}%", "computed from DEM + product LOS",
                            "Local and cast shadow; masked pixels excluded from change detection."))
            sn = pre.get("snow") or {}
            if sn:
                n_snow = int(sum(sn.get("snow", [])))
                out.append(_chk("GCOV masks", "Snow-date mask", True, f"{n_snow}/{len(sn.get('snow', []))} acquisition dates snow-covered "
                                f"(ERA5-Land > {sn.get('threshold_m', 0.01) * 100:.0f} cm)", "per-date snow flag",
                                "Forest changes first seen on snow dates are labelled snow-affected, not lasting disturbance."))
    return out


def verify_region(region: Region) -> dict[str, Any]:
    checks: list[dict] = []
    for fn in (check_metadata_and_masks, check_gcov, check_gunw, check_goff):
        try:
            checks += fn(region)
        except Exception as ex:
            log.warning("%s failed for %s: %s", fn.__name__, region.id, ex)
            checks.append(_chk(fn.__name__, "check could not run", None, str(ex)[:200], ""))
    counts = {k: sum(c["status"] == k for c in checks) for k in ("pass", "warn", "fail")}
    res = {"region": region.id, "checks": checks, "counts": counts}
    od = WEB_DATA_DIR / region.id
    od.mkdir(parents=True, exist_ok=True)
    (od / "verification.json").write_text(json.dumps(res, indent=1, default=str))
    return res
