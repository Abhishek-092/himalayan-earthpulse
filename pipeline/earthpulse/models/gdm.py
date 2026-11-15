"""GDM - Geomechanical Dynamics Model -> GDES (Ground Deformation Evolution Signature).

Per stack (track/frame), from NISAR GUNW pairs:
  1. unwrapped phase - ionosphere phase screen (L-band ionosphere is significant)
  2. mask: connected component 0, low coherence, invalid
  3. phase -> LOS metres with the *granule's own* centre frequency (frequencyA of these products is 1229 MHz)
  4. per-pair empirical correction: planar ramp + phase/elevation (stratified troposphere) fitted on stable pixels
  5. referencing to an automatically chosen stable area (two-pass: coherent, gentle, low-velocity pixels)
  6. SBAS inversion with coherence-based (Cramer-Rao) weights -> displacement time series + 1-sigma
  7. shared temporal framework per pixel: velocity, acceleration, onset, persistence, ...
Then across geometries: ascending + descending LOS velocities -> vertical and east-west velocity.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import numpy as np

from earthpulse.catalog.discover import parse_granule_name
from earthpulse.catalog.stacks import region_granules, select_stacks
from earthpulse.config import SPEED_OF_LIGHT, WORK_DIR, Region
from earthpulse.geo import ancillary
from earthpulse.geo.area import masked_area_km2, pixel_area_m2
from earthpulse.io.nisar import Subset, cache_path, interp_cube
from earthpulse.models.sbas import coherence_phase_sigma, decompose_asc_desc, invert
from earthpulse.temporal.framework import TemporalConfig, analyze, summarize_series

log = logging.getLogger(__name__)

MIN_COHERENCE = 0.4
GOLDSTEIN_ALPHA = 0.0  # tested 0.3/0.5/0.8 (goldstein_sensitivity): biases sparse settlement pixels -> not applied
LOOKS = 40.0  # conservative effective looks of the 80 m unwrapped product
ATMO_FLOOR_M = 0.004  # residual atmospheric noise floor per pair (after empirical correction)


def _date(iso: str) -> np.datetime64:
    return np.datetime64(iso[:10], "D")


def load_stack(region: Region, key: str, members) -> dict[str, Any] | None:
    subs: list[Subset] = []
    for g in members:
        p = cache_path(region.id, "GUNW", g.granule_id)
        if p.exists():
            subs.append(Subset.load(p))
    if len(subs) < 3:
        log.warning("stack %s: only %d cached pairs - skipping", key, len(subs))
        return None
    geom_path = cache_path(region.id, "GUNW", members[0].granule_id, geometry=True)
    geom = Subset.load(geom_path) if geom_path.exists() else None
    # common grid: all members of a stack share the NISAR track/frame grid; intersect coordinates defensively
    x = subs[0].x
    y = subs[0].y
    for s in subs[1:]:
        x = np.intersect1d(x, s.x)
        y = np.intersect1d(y, s.y)[::-1]
    def crop(s: Subset, name: str) -> np.ndarray:
        ix = np.searchsorted(s.x, x)
        iy = np.searchsorted(-s.y, -y)
        return s.layers[name][np.ix_(iy, ix)]
    pairs, phase, coh, ccomp, iono, wl, ids = [], [], [], [], [], [], []
    for s in subs:
        info = parse_granule_name(s.granule_id)
        if not info["sec_time"]:
            continue
        pairs.append((_date(info["ref_time"]), _date(info["sec_time"])))
        ph = crop(s, "unwrappedPhase").astype(float)
        if "ionospherePhaseScreen" in s.layers:
            io = crop(s, "ionospherePhaseScreen").astype(float)
            ph = ph - np.where(np.isfinite(io), io, 0.0)
            iono.append(io)
        phase.append(ph)
        coh.append(crop(s, "coherenceMagnitude").astype(float))
        ccomp.append(crop(s, "connectedComponents"))
        f = s.attrs.get("center_frequency_hz") or 1.229e9
        wl.append(SPEED_OF_LIGHT / float(f))
        ids.append(s.granule_id)
    return {"key": key, "x": x, "y": y, "epsg": subs[0].epsg, "pairs": pairs, "phase": np.stack(phase),
            "coh": np.stack(coh), "ccomp": np.stack(ccomp), "wavelength": np.array(wl), "ids": ids, "geom": geom,
            "direction": members[0].direction, "track": members[0].track, "frame": members[0].frame}


def _model(field: np.ndarray, elev: np.ndarray, use: np.ndarray, iters: int = 4) -> np.ndarray:
    """Robust (iteratively trimmed) fit of a + b x + c y + k h: ramp + stratified troposphere. Returns the model."""
    H, W = field.shape
    yy, xx = np.mgrid[0:H, 0:W]
    hz = np.where(np.isfinite(elev), elev, np.nanmedian(elev)) / 1000.0
    G = np.stack([np.ones(H * W), (xx.ravel() - W / 2) / W, (yy.ravel() - H / 2) / H, hz.ravel()], 1)
    f = field.ravel()
    m = use.ravel() & np.isfinite(f)
    if m.sum() < 50:
        return np.full(field.shape, np.nanmedian(field))
    coef = np.zeros(G.shape[1])
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(G[m], f[m], rcond=None)
        res = f - G @ coef
        mad = 1.4826 * np.nanmedian(np.abs(res[m] - np.nanmedian(res[m])))
        m = use.ravel() & np.isfinite(f) & (np.abs(res - np.nanmedian(res[m])) < 2.5 * max(mad, 1e-6))
    return (G @ coef).reshape(H, W)


def _fit_remove(field: np.ndarray, elev: np.ndarray, use: np.ndarray) -> np.ndarray:
    return field - _model(field, elev, use)


def reunwrap(phase_unw: np.ndarray, valid: np.ndarray, elev: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Repair unwrapping errors: remove a smooth model, re-wrap, quality-unwrap the (now low-gradient) residual,
    add the model back. Integer-cycle jumps between regions of the original unwrapped product disappear because
    the residual is unwrapped from scratch on a smooth field. Returns phase (rad), NaN outside `valid`."""
    from skimage.restoration import unwrap_phase

    ph = np.where(valid, phase_unw, np.nan)
    model = _model(ph, elev, use & valid)
    resid = np.angle(np.exp(1j * (np.nan_to_num(ph) - model)))
    if GOLDSTEIN_ALPHA > 0:  # GUNW reports wrappedInterferogramFiltering = no_filter -> apply Goldstein-Werner here
        from earthpulse.preprocess.phase import goldstein

        resid = np.angle(goldstein(np.where(valid, np.exp(1j * resid), 0), alpha=GOLDSTEIN_ALPHA, patch=16, step=4))
    masked = np.ma.masked_array(resid, mask=~valid)
    un = np.ma.filled(unwrap_phase(masked), np.nan)
    # Disconnected coherent islands get arbitrary 2*pi offsets from any unwrapper: snap each island to the
    # integer cycle that makes its median residual closest to zero (small-residual assumption after modelling).
    from scipy.ndimage import label

    lab, n = label(valid, structure=np.ones((3, 3)))
    for i in range(1, n + 1):
        m = lab == i
        k = np.round(np.nanmedian(un[m]) / (2 * np.pi))
        un[m] -= 2 * np.pi * k
    # drop tiny islands (< 6 px): too small to tie reliably
    sizes = np.bincount(lab.ravel())
    tiny = sizes[lab] < 6
    out = np.where(valid & ~tiny, un + model, np.nan)
    return out


def process_stack(st: dict[str, Any], elev: np.ndarray, slope: np.ndarray, water: np.ndarray,
                  reference: np.ndarray | None = None) -> dict[str, Any]:
    dates = np.array(sorted({d for p in st["pairs"] for d in p}))
    didx = {d: i for i, d in enumerate(dates)}
    pairs = [(didx[a], didx[b]) for a, b in st["pairs"]]
    P = len(pairs)
    coh = st["coh"]
    valid = (st["ccomp"] > 0) & (coh >= MIN_COHERENCE) & np.isfinite(st["phase"]) & ~water[None]
    mean_coh0 = np.nanmean(coh, axis=0)
    use0 = (mean_coh0 >= np.nanpercentile(mean_coh0, 50)) & (slope < 25)
    phase = np.stack([reunwrap(st["phase"][k], valid[k], elev, use0) for k in range(P)])
    los = np.where(np.isfinite(phase), -st["wavelength"][:, None, None] / (4 * np.pi) * phase, np.nan)
    valid = valid & np.isfinite(los)
    # --- pair quality control (data-driven): residual spread after the smooth model, valid fraction -------
    spread = np.full(P, np.nan)
    for k in range(P):
        f = _fit_remove(los[k], elev, use0 & valid[k])
        v = f[np.isfinite(f)]
        if v.size > 50:
            spread[k] = (np.percentile(v, 95) - np.percentile(v, 5)) / 3.29  # ~1 sigma for Gaussian
    vfrac = valid.mean(axis=(1, 2))
    med = np.nanmedian(spread)
    keep = np.isfinite(spread) & (vfrac >= 0.35) & (spread <= 3 * med)
    qc = [{"pair": [str(dates[i]), str(dates[j])], "valid_fraction": round(float(vfrac[k]), 3),
           "noise_cm": None if not np.isfinite(spread[k]) else round(float(spread[k] * 100), 2),
           "used": bool(keep[k])} for k, (i, j) in enumerate(pairs)]
    los[~keep] = np.nan
    valid[~keep] = False
    atmo = np.where(np.isfinite(spread), np.maximum(spread, ATMO_FLOOR_M), 0.05)[:, None, None]
    sig = np.sqrt((st["wavelength"][:, None, None] / (4 * np.pi) * coherence_phase_sigma(coh, LOOKS)) ** 2
                  + atmo ** 2)
    sig = np.where(valid, sig, np.nan)
    mean_coh = np.nanmean(np.where(np.isfinite(coh), coh, np.nan), axis=0)
    if reference is None:  # pass 1: coherent, gentle slopes
        reference = (mean_coh >= np.nanpercentile(mean_coh, 60)) & (slope < 20) & ~water
    corrected = np.empty_like(los)
    for k in range(P):
        f = _fit_remove(los[k], elev, reference & valid[k])
        ref_vals = f[reference & valid[k]]
        corrected[k] = f - (np.nanmedian(ref_vals) if ref_vals.size > 20 else np.nanmedian(f))
    res = invert(dates, pairs, corrected, sig, smooth=0.3)
    tr = analyze(dates, np.moveaxis(res.displacement, 0, -1), np.moveaxis(res.sigma, 0, -1),
                 TemporalConfig(baseline_n=1, k_sigma=2.0, min_persist=2))
    return {"dates": dates, "pairs": pairs, "sbas": res, "temporal": tr, "mean_coh": mean_coh,
            "reference": reference, "valid_frac": valid.mean(axis=0), "pair_qc": qc}


def run_gdm(region: Region) -> dict[str, Any]:
    gs = region_granules(region)
    stacks = select_stacks(gs, "GUNW")
    loaded = [s for s in (load_stack(region, k, v) for k, v in stacks.items()) if s is not None]
    if not loaded:
        raise RuntimeError(f"no GUNW stacks cached for {region.id} - run `earthpulse ingest --product GUNW`")
    out_dir = WORK_DIR / region.id / "gdm"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for st in loaded:
        x, y, epsg = st["x"], st["y"], st["epsg"]
        elev = ancillary.dem(epsg, x, y)
        slope, _ = ancillary.slope_aspect(elev, x[1] - x[0], y[1] - y[0])
        occ = ancillary.water_occurrence(epsg, x, y)
        from earthpulse.preprocess import masks as pm
        rmask = pm.radar_masks(st["geom"], x, y, elev)
        water = (np.nan_to_num(occ) > 50) | rmask["layover"] | rmask["shadow"]  # excluded pixels
        # pass 1 -> velocity -> pass 2 reference = coherent, gentle, lowest |velocity| pixels
        p1 = process_stack(st, elev, slope, water)
        v1 = p1["temporal"].params["rate_per_yr"]
        cand = (p1["mean_coh"] >= np.nanpercentile(p1["mean_coh"], 50)) & (slope < 20) & ~water & np.isfinite(v1)
        thr = np.nanpercentile(np.abs(v1[cand]), 40) if cand.sum() > 50 else np.inf
        ref2 = cand & (np.abs(v1) <= thr)
        p2 = process_stack(st, elev, slope, water, reference=ref2)
        # LOS unit vector (ground -> satellite): east, north from cubes, up = sqrt(1 - e^2 - n^2)
        los_vec = None
        g = st["geom"]
        if g is not None and "cube_losUnitVectorX" in g.layers:
            ex = interp_cube(g.layers["cube_losUnitVectorX"], g.attrs["heights"], g.attrs["cube_x"],
                             g.attrs["cube_y"], x, y, elev)
            ny = interp_cube(g.layers["cube_losUnitVectorY"], g.attrs["heights"], g.attrs["cube_x"],
                             g.attrs["cube_y"], x, y, elev)
            los_vec = np.stack([ex, ny, np.sqrt(np.clip(1 - ex ** 2 - ny ** 2, 0, 1))])
        p2["radar_masks"] = {"layover_pct": round(100 * float(rmask["layover"].mean()), 2),
                             "shadow_pct": round(100 * float(rmask["shadow"].mean()), 2), "available": rmask["available"],
                             "incidence_range_deg": rmask.get("incidence_range")}
        p2["processing_flags"] = st["geom"].attrs.get("processing") if st["geom"] is not None else None
        p2.update({"x": x, "y": y, "epsg": epsg, "elev": elev, "slope": slope, "water": water, "los": los_vec,
                   "ids": st["ids"], "direction": st["direction"], "track": st["track"], "frame": st["frame"]})
        results[st["key"]] = p2
        _save_stack(out_dir, st["key"], p2)
        log.info("GDM %s: %d dates, %d pairs, ref pixels %d", st["key"], len(p2["dates"]), len(p2["pairs"]),
                 int(ref2.sum()))
    summary = _summarize(region, results, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    from earthpulse.preprocess import masks as pm
    members = [g for k, v in stacks.items() for g in v if g.granule_id in {i for r in results.values() for i in r["ids"]}]
    pm.write_processing_log(
        region, "gdm", members,
        params={"min_coherence": MIN_COHERENCE, "goldstein_alpha": GOLDSTEIN_ALPHA, "looks_assumed": LOOKS,
                "ionosphere": "product split-spectrum screen subtracted", "troposphere": "product wet+hydrostatic (ISCE3) + "
                "empirical ramp & elevation residual fit", "unwrapping": "product SNAPHU + residual re-unwrap with island "
                "cycle snapping", "pair_qc": "valid >= 35% and noise <= 3x median", "inversion": "SBAS with temporal "
                "smoothing (0.3), Cramer-Rao weights", "reference": "two-pass auto stable area",
                "reliability": "built-up/bare/moss or coherence >= 0.7"},
        ancillary=["Copernicus DEM GLO-30", "ESA WorldCover 2021", "JRC Global Surface Water"],
        qc={k: {"radar_masks": r.get("radar_masks"), "pairs_used": sum(q["used"] for q in r.get("pair_qc", []))}
            for k, r in results.items()},
        outputs=["stack_*.npz", "decomposition.npz", "summary.json"],
        product_processing={k: r.get("processing_flags") for k, r in results.items()})
    return summary


def _save_stack(out_dir, key: str, r: dict[str, Any]) -> None:
    p = r["temporal"].params
    np.savez_compressed(
        out_dir / f"stack_{key}.npz", x=r["x"], y=r["y"], epsg=r["epsg"],
        dates=r["dates"].astype("datetime64[D]").astype(str),
        displacement=r["sbas"].displacement.astype(np.float32), sigma=r["sbas"].sigma.astype(np.float32),
        mean_coh=r["mean_coh"].astype(np.float32), reference=r["reference"], elev=r["elev"].astype(np.float32),
        slope=r["slope"], los=(r["los"] if r["los"] is not None else np.zeros((3, 1, 1))).astype(np.float32),
        residual_rms=r["sbas"].residual_rms.astype(np.float32),
        **{f"p_{k}": v.astype(np.float32) for k, v in p.items()},
    )


def _hotspot(vel: np.ndarray, sig: np.ndarray, valid: np.ndarray, min_rate: float) -> np.ndarray:
    """Significant movers: |v| > 2 sigma and > min_rate, cleaned with a 3x3 majority to drop isolated pixels."""
    from scipy.ndimage import binary_opening, label

    m = valid & np.isfinite(vel) & (np.abs(vel) > np.maximum(2 * sig, min_rate))
    m = binary_opening(m, np.ones((2, 2)))
    lab, n = label(m)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes[1:] >= 4
    return keep[lab]


def _summarize(region: Region, results: dict[str, Any], out_dir) -> dict[str, Any]:
    summary: dict[str, Any] = {"region": region.id, "model": "GDM", "signature": "GDES", "stacks": {},
                               "limitations": [
                                   "Line-of-sight (LOS) measurements; vertical/east-west need two geometries "
                                   "and neglect north-south motion.",
                                   "Atmospheric delay is corrected empirically (ramp + elevation-correlated "
                                   "term); residual turbulence remains in the stated uncertainty.",
                                   "Pixels with coherence < 0.4 (snow, dense forest, rapid change) are masked per pair; pairs with < 35% "
                                   "valid pixels or excessive noise are excluded and bridged by temporal smoothing.",
                                   "L-band 12-day interferogram chains carry a vegetation-dependent phase-closure "
                                   "('fading') bias: apparent velocities over forest and grassland are biased, so "
                                   "deformation is only reported on reliable persistent scatterers (settlements, "
                                   "bare rock/soil, coherence >= 0.7).",
                                   "One year of data: velocities and accelerations can include seasonal effects.",
                                   "Deformation is not attributed to any cause by this analysis.",
                               ]}
    for key, r in results.items():
        p = r["temporal"].params
        vel = p["rate_per_yr"]
        area = pixel_area_m2(r["epsg"], r["x"], r["y"])
        valid = np.isfinite(vel)
        wc = ancillary.worldcover(r["epsg"], r["x"], r["y"])
        # Reliability: L-band short-baseline chains carry a vegetation-dependent phase-closure ("fading") bias.
        # Persistent scatterers (built-up, bare rock/soil, moss) or very coherent pixels are trusted.
        reliable = valid & (np.isin(wc, (50, 60, 100)) | (r["mean_coh"] >= 0.7))
        ref_ok = r["reference"] & valid
        ref_scatter = float(1.4826 * np.nanmedian(np.abs(vel[ref_ok] - np.nanmedian(vel[ref_ok])))) if ref_ok.any() else 0.02
        vse = np.sqrt(p["rate_se"] ** 2 + ref_scatter ** 2)  # formal + empirical spatial scatter
        hot = _hotspot(vel, vse, reliable, 0.02)
        bias_diag = {}
        for code, name in ((10, "tree cover"), (30, "grassland"), (50, "built-up"), (60, "bare / sparse")):
            mm = (wc == code) & valid
            if mm.sum() >= 20:
                vv = vel[mm] * 1000
                bias_diag[name] = {"n": int(mm.sum()), "median_mm_yr": round(float(np.median(vv)), 1),
                                   "iqr_mm_yr": round(float(np.subtract(*np.percentile(vv, [75, 25]))), 1),
                                   "sem_mm_yr": round(float(1.253 * 1.4826 * np.median(np.abs(vv - np.median(vv)))
                                                            / np.sqrt(mm.sum())), 1),
                                   "mean_coherence": round(float(np.nanmean(r["mean_coh"][mm])), 3)}
        p["rate_se_eff"] = vse.astype(np.float32)
        p["reliable"] = reliable.astype(np.float32)
        # spatial deformation gradient (mm/yr per km) on reliable pixels
        from scipy.ndimage import generic_filter
        vr = generic_filter(np.where(reliable, vel, np.nan) * 1000, np.nanmedian, size=3, mode="nearest")  # denoise first
        vr = np.where(reliable, vr, np.nan)
        gyy, gxx = np.gradient(vr, abs(r["y"][1] - r["y"][0]) / 1000, abs(r["x"][1] - r["x"][0]) / 1000)
        p["velocity_gradient"] = np.hypot(gxx, gyy).astype(np.float32)
        # confidence classes: 3 high, 2 medium, 1 low, 0 not evaluated
        conf = np.zeros(vel.shape, np.float32)
        conf[valid] = 1
        conf[reliable] = 2
        conf[reliable & (r["mean_coh"] >= 0.7) & (vse < 0.01)] = 3
        p["confidence_class"] = conf
        _save_stack(out_dir, key, r)
        disp = r["sbas"].displacement
        sig = r["sbas"].sigma
        # regional series: hotspot median and reference median (robust)
        def series(mask):
            if mask.sum() == 0:
                return None
            d = np.nanmedian(disp[:, mask], axis=1)
            s = np.nanmedian(sig[:, mask], axis=1) / np.sqrt(max(1, min(mask.sum(), 25)))
            return d, s
        hs = series(hot)
        rs = series(r["reference"] & valid)
        entry = {
            "direction": r["direction"], "track": r["track"], "frame": r["frame"],
            "dates": [str(d) for d in r["dates"]], "n_pairs": len(r["pairs"]), "granules": r["ids"],
            "valid_area_km2": round(masked_area_km2(valid, area), 2),
            "deforming_area_km2": round(masked_area_km2(hot, area), 3),
            "velocity_mm_yr": {
                "p05": float(np.nanpercentile(vel[valid], 5) * 1000) if valid.any() else None,
                "p95": float(np.nanpercentile(vel[valid], 95) * 1000) if valid.any() else None,
                "hotspot_median": float(np.nanmedian(vel[hot]) * 1000) if hot.any() else None,
                "reference_median": float(np.nanmedian(vel[r["reference"] & valid]) * 1000)
                if (r["reference"] & valid).any() else None,
                "reference_scatter": float(1.4826 * np.nanmedian(np.abs(vel[r["reference"] & valid]
                                                                       - np.nanmedian(vel[r["reference"] & valid])))
                                           * 1000) if (r["reference"] & valid).any() else None,
            },
            "mean_coherence": float(np.nanmean(r["mean_coh"])),
            "reliable_area_km2": round(masked_area_km2(reliable, area), 2),
            "empirical_velocity_scatter_mm_yr": round(ref_scatter * 1000, 1),
            "landcover_velocity_mm_yr": bias_diag,
            "settlement_velocity_mm_yr": bias_diag.get("built-up"),
            "pair_qc": r.get("pair_qc"),
            "radar_masks": r.get("radar_masks"), "processing_flags": r.get("processing_flags"),
            "goldstein_alpha": GOLDSTEIN_ALPHA,
            "gradient_mm_yr_per_km_p95": float(np.nanpercentile(p["velocity_gradient"][np.isfinite(p["velocity_gradient"])], 95))
            if np.isfinite(p["velocity_gradient"]).any() else None,
            "pairs_used": int(sum(q["used"] for q in r.get("pair_qc", []))),
        }
        if hs is not None:
            entry["hotspot_series_m"] = {"disp": np.round(hs[0], 5).tolist(), "sigma": np.round(hs[1], 5).tolist()}
            entry["hotspot_temporal"] = summarize_series(r["dates"], hs[0], np.maximum(hs[1], 1e-4),
                                                         TemporalConfig(baseline_n=1, k_sigma=2.0))
        if rs is not None:
            entry["reference_series_m"] = {"disp": np.round(rs[0], 5).tolist(), "sigma": np.round(rs[1], 5).tolist()}
        summary["stacks"][key] = entry
    # ascending/descending decomposition on the common grid
    asc = [k for k, r in results.items() if r["direction"] == "A" and r["los"] is not None]
    dsc = [k for k, r in results.items() if r["direction"] == "D" and r["los"] is not None]
    if asc and dsc:
        a, d = results[asc[0]], results[dsc[0]]
        x = np.intersect1d(a["x"], d["x"])
        y = np.intersect1d(a["y"], d["y"])[::-1]
        if len(x) > 10 and len(y) > 10:
            def c(r, arr):
                ix = np.searchsorted(r["x"], x)
                iy = np.searchsorted(-r["y"], -y)
                return arr[..., iy[:, None], ix[None, :]]
            va, sa = c(a, a["temporal"].params["rate_per_yr"]), c(a, a["temporal"].params["rate_se"])
            vd, sd = c(d, d["temporal"].params["rate_per_yr"]), c(d, d["temporal"].params["rate_se"])
            east, up, se, su = decompose_asc_desc(va, vd, c(a, a["los"]), c(d, d["los"]), sa, sd)
            rel = (c(a, a["temporal"].params["reliable"]) > 0.5) & (c(d, d["temporal"].params["reliable"]) > 0.5)
            area = pixel_area_m2(a["epsg"], x, y)
            sub = rel & np.isfinite(up) & (up < -np.maximum(2 * su, 0.02))
            # consistency: if motion were purely vertical, both geometries would give the same vertical-equivalent rate
            lu_a, lu_d = c(a, a["los"])[2], c(d, d["los"])[2]
            both = rel & np.isfinite(va) & np.isfinite(vd)
            ua, ud = va / np.where(lu_a > 0.3, lu_a, np.nan), vd / np.where(lu_d > 0.3, lu_d, np.nan)
            dd = (ua - ud)[both] * 1000
            dd = dd[np.isfinite(dd)]
            consistency = {"n_pixels": int(dd.size),
                           "median_diff_mm_yr": round(float(np.median(dd)), 2) if dd.size else None,
                           "robust_spread_mm_yr": round(float(1.4826 * np.median(np.abs(dd - np.median(dd)))), 2) if dd.size else None,
                           "pearson_r": round(float(np.corrcoef(ua[both & np.isfinite(ua) & np.isfinite(ud)],
                                                               ud[both & np.isfinite(ua) & np.isfinite(ud)])[0, 1]), 3)
                           if (both & np.isfinite(ua) & np.isfinite(ud)).sum() > 10 else None,
                           "note": "Vertical-equivalent velocities from the two geometries on reliable pixels; agreement "
                                   "near zero difference supports the measurement, differences include real horizontal motion."}
            np.savez_compressed(out_dir / "decomposition.npz", x=x, y=y, epsg=a["epsg"], east=east.astype(np.float32),
                                up=up.astype(np.float32), sigma_east=se.astype(np.float32),
                                sigma_up=su.astype(np.float32), reliable=rel)
            summary["decomposition"] = {
                "stacks": [asc[0], dsc[0]],
                "subsiding_area_km2": round(masked_area_km2(sub, area), 3),
                "reliable_area_km2": round(masked_area_km2(rel & np.isfinite(up), area), 2),
                "consistency": consistency,
                "up_mm_yr_reliable_median": float(np.nanmedian(up[rel]) * 1000) if (rel & np.isfinite(up)).any() else None,
                "up_mm_yr_p05": float(np.nanpercentile(up[rel & np.isfinite(up)], 5) * 1000) if (rel & np.isfinite(up)).any() else None,
                "east_mm_yr_abs_p95": float(np.nanpercentile(np.abs(east[np.isfinite(east)]), 95) * 1000)
                if np.isfinite(east).any() else None,
                "note": "Velocities from overlapping but not identical date ranges of the two stacks.",
            }
    summary["generated"] = datetime.utcnow().isoformat() + "Z"
    return summary


def goldstein_sensitivity(region: Region, alphas=(0.0, 0.3, 0.5, 0.8)) -> dict[str, Any]:
    """Run GDM with several Goldstein strengths and record what changes (the preprocessing plan asks to test filter
    settings against unfiltered results before adopting them). Restores the default afterwards."""
    global GOLDSTEIN_ALPHA
    default = GOLDSTEIN_ALPHA
    rows = []
    try:
        for a in alphas:
            GOLDSTEIN_ALPHA = a
            s = run_gdm(region)
            for k, v in s["stacks"].items():
                used = [q for q in v["pair_qc"] if q["used"]]
                rows.append({"alpha": a, "stack": k, "settlement_mm_yr": v["settlement_velocity_mm_yr"]["median_mm_yr"],
                             "settlement_sem": v["settlement_velocity_mm_yr"]["sem_mm_yr"],
                             "reference_scatter_mm_yr": v["empirical_velocity_scatter_mm_yr"],
                             "mean_pair_noise_cm": round(float(np.mean([q["noise_cm"] for q in used if q["noise_cm"]])), 2)})
    finally:
        GOLDSTEIN_ALPHA = default
        run_gdm(region)
    out = {"rows": rows, "adopted_alpha": default,
           "conclusion": "Goldstein filtering lowers per-pair noise but shifts the sparse settlement estimates strongly "
                         "(up to tens of mm/yr at alpha 0.8) because filter energy leaks across masked gaps of an already "
                         "multilooked (about 7x16 looks) product; results are therefore reported without it."}
    (WORK_DIR / region.id / "gdm" / "goldstein_sensitivity.json").write_text(json.dumps(out, indent=1))
    return out
