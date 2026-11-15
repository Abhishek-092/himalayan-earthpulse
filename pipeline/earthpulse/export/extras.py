"""Extra web assets used by the Explore / Story UI:

* radar/   - the raw NISAR radar picture (HH magenta / HV green composite) per GCOV date subset, for the "Raw radar" button
* <model>/findings.geojson - vectorised key findings (download button)
* story/   - real cloudy Sentinel-2 true-colour scene next to the same-week NISAR radar image ("radar sees through clouds")
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform

from earthpulse.catalog.discover import parse_granule_name
from earthpulse.catalog.stacks import region_granules, select_stacks
from earthpulse.config import WEB_DATA_DIR, WORK_DIR, Region
from earthpulse.export.web import _write_json, corners
from earthpulse.geo.ancillary import _lonlat_bounds, _warp, grid_transform
from earthpulse.io.nisar import Subset, cache_path
from earthpulse.preprocess.speckle import to_db

log = logging.getLogger(__name__)


def _ml2(a):
    H2, W2 = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    with np.errstate(all="ignore"):
        return np.nanmean(a[:H2, :W2].reshape(H2 // 2, 2, W2 // 2, 2), axis=(1, 3))


def radar_rgb(s: Subset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hh = _ml2(np.where(s.layers["HHHH"] > 0, s.layers["HHHH"], np.nan).astype(float))
    hv = _ml2(np.where(s.layers["HVHV"] > 0, s.layers["HVHV"], np.nan).astype(float))
    hhd, hvd = to_db(hh), to_db(hv)
    def st(a):
        lo, hi = np.nanpercentile(a, [2, 98])
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    r, g = st(hhd), st(hvd)
    rgb = np.stack([r, g, r], -1)
    alpha = np.isfinite(hh) & np.isfinite(hv)
    img = np.dstack([(np.nan_to_num(rgb) * 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    x = s.x[: hh.shape[1] * 2].reshape(-1, 2).mean(1)
    y = s.y[: hh.shape[0] * 2].reshape(-1, 2).mean(1)
    return img, x, y


def export_radar(region: Region, max_frames: int = 30) -> dict[str, Any] | None:
    stacks = select_stacks(region_granules(region), "GCOV")
    best = None
    for key, members in stacks.items():
        cached = [g for g in members if cache_path(region.id, "GCOV", g.granule_id).exists()]
        if cached and (best is None or len(cached) > len(best[1])):
            best = (key, cached)
    if best is None:
        return None
    key, cached = best
    od = WEB_DATA_DIR / region.id / "radar"
    od.mkdir(parents=True, exist_ok=True)
    frames, dates, crn = [], [], None
    shape0 = None
    for g in cached[:max_frames]:
        s = Subset.load(cache_path(region.id, "GCOV", g.granule_id))
        if "HHHH" not in s.layers or "HVHV" not in s.layers:
            continue
        if shape0 is None:
            shape0 = s.shape
        if s.shape != shape0:
            continue
        img, x, y = radar_rgb(s)
        d = parse_granule_name(g.granule_id)["ref_time"][:10]
        fn = f"radar_{d}.png"
        Image.fromarray(img).save(od / fn, optimize=True)
        frames.append(fn)
        dates.append(d)
        crn = corners(s.epsg, x, y)
    meta = {"stack": key, "frames": frames, "dates": dates, "corners": crn,
            "title": "Raw NISAR radar (HH magenta, HV green)",
            "how_to_read": "The radar picture itself, before any analysis: green = vegetation volume scattering (HV), "
                           "magenta = surfaces, rock and buildings (HH), black = smooth water or radar shadow."}
    (od / "radar.json").write_text(json.dumps(meta))
    return meta


def _vectorize(mask: np.ndarray, epsg: int, x: np.ndarray, y: np.ndarray, props_fn, min_px: int = 4,
               simplify_m: float = 15.0) -> list[dict]:
    tr = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    feats = []
    lab_img = mask.astype(np.uint8)
    for geom, val in shapes(lab_img, mask=mask, transform=grid_transform(x, y)):
        if val != 1:
            continue
        g = shape(geom)
        if g.area < min_px * abs((x[1] - x[0]) * (y[1] - y[0])):
            continue
        props = props_fn(g)
        props["area_km2"] = round(g.area / 1e6, 4)
        g2 = shp_transform(tr, g.simplify(simplify_m))
        feats.append({"type": "Feature", "properties": props, "geometry": mapping(g2)})
    return feats


def export_findings(region: Region) -> dict[str, int]:
    out = {}
    wd = WORK_DIR / region.id
    src = {"region": region.id, "source": "NISAR L2 (provisional), Himalayan EarthPulse"}
    if (wd / "bdm" / "bdm.npz").exists():
        z = np.load(wd / "bdm" / "bdm.npz")
        dates = list(z["dates"])
        persistent = z["disturbance"] & ~z["recovered"]
        fd = z["first_dark"]
        x, y = z["x"], z["y"]
        from rasterio.features import geometry_mask

        def props(g):
            m = ~geometry_mask([g], (len(y), len(x)), grid_transform(x, y))
            idx = fd[m & persistent]
            on = dates[int(np.median(idx))] if idx.size else None
            return {"type": "persistent potential forest disturbance", "first_detected": on}
        feats = _vectorize(persistent, int(z["epsg"]), x, y, props)
        _write_fc(WEB_DATA_DIR / region.id / "bdm" / "findings.geojson", feats, src)
        out["bdm"] = len(feats)
    stacks = sorted((wd / "gdm").glob("stack_*.npz")) if (wd / "gdm").exists() else []
    if stacks:
        z = np.load(stacks[-1])
        v = z["p_rate_per_yr"]
        se = z["p_rate_se_eff"] if "p_rate_se_eff" in z.files else z["p_rate_se"]
        rel = z["p_reliable"] > 0.5 if "p_reliable" in z.files else np.isfinite(v)
        hot = rel & np.isfinite(v) & (np.abs(v) > np.maximum(2 * se, 0.02))
        x, y = z["x"], z["y"]
        from rasterio.features import geometry_mask

        def props_g(g):
            m = ~geometry_mask([g], (len(y), len(x)), grid_transform(x, y))
            return {"type": "significant line-of-sight motion (reliable pixels)",
                    "median_velocity_mm_yr": round(float(np.nanmedian(v[m & hot]) * 1000), 1)}
        feats = _vectorize(hot, int(z["epsg"]), x, y, props_g, min_px=1, simplify_m=40)
        _write_fc(WEB_DATA_DIR / region.id / "gdm" / "findings.geojson", feats, src)
        out["gdm"] = len(feats)
    if (wd / "cdm" / "goff_velocity.npz").exists():
        z = np.load(wd / "cdm" / "goff_velocity.npz")
        sp = z["speed"]
        gl = z["glacier"]
        thr = float(np.nanpercentile(sp[gl & np.isfinite(sp)], 75)) if (gl & np.isfinite(sp)).any() else 20
        fast = gl & np.isfinite(sp) & (sp > max(thr, 15))
        x, y = z["x"], z["y"]
        from rasterio.features import geometry_mask

        def props_c(g):
            m = ~geometry_mask([g], (len(y), len(x)), grid_transform(x, y))
            return {"type": f"fast-flowing ice (> {max(thr, 15):.0f} m/yr)",
                    "median_speed_m_yr": round(float(np.nanmedian(sp[m & fast])), 1)}
        feats = _vectorize(fast, int(z["epsg"]), x, y, props_c, min_px=3, simplify_m=60)
        _write_fc(WEB_DATA_DIR / region.id / "cdm" / "findings.geojson", feats, src)
        out["cdm"] = len(feats)
    return out


def _write_fc(path, feats, props):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "properties": props, "features": feats}))


def export_story_clouds(region: Region, months=("2026-06-15", "2026-09-15")) -> dict[str, Any] | None:
    """Find a monsoon NISAR GCOV date and the cloudiest Sentinel-2 scene within +-3 days over the region."""
    import planetary_computer
    import pystac_client

    stacks = select_stacks(region_granules(region), "GCOV")
    cands = []
    for key, members in stacks.items():
        for g in members:
            d = g.ref_time[:10]
            if months[0] <= d <= months[1] and cache_path(region.id, "GCOV", g.granule_id).exists():
                cands.append((d, g))
    if not cands:
        return None
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    lon, lat = region.geometry.centroid.x, region.geometry.centroid.y
    best = None
    for d, g in sorted(cands):
        t = datetime.fromisoformat(d)
        items = cat.search(collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [lon, lat]},
                           datetime=f"{(t - timedelta(days=3)).date()}/{(t + timedelta(days=3)).date()}",
                           max_items=10).item_collection()
        for it in items:
            cc = it.properties.get("eo:cloud_cover", 0)
            if best is None or cc > best[0]:
                best = (cc, d, g, it)
        if best and best[0] > 85:
            break
    if best is None:
        return None
    cc, d, g, it = best
    s = Subset.load(cache_path(region.id, "GCOV", g.granule_id))
    img, x, y = radar_rgb(s)
    od = WEB_DATA_DIR / "story"
    od.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(od / "radar_monsoon.png", optimize=True)
    # TCI "visual" asset is a 3-band 8-bit COG; warp each band
    import rasterio
    from rasterio.warp import reproject

    bands = []
    with rasterio.open(it.assets["visual"].href) as src:
        for b in (1, 2, 3):
            dst = np.zeros((len(y), len(x)), np.float32)
            reproject(rasterio.band(src, b), dst, dst_transform=grid_transform(x, y), dst_crs=f"EPSG:{s.epsg}",
                      resampling=Resampling.bilinear)
            bands.append(dst)
    opt = np.clip(np.stack(bands, -1), 0, 255).astype(np.uint8)
    Image.fromarray(opt).save(od / "optical_monsoon.png", optimize=True)
    meta = {"region": region.id, "nisar_date": d, "nisar_granule": g.granule_id, "s2_date": it.datetime.date().isoformat(),
            "s2_item": it.id, "s2_cloud_cover": round(float(cc), 1), "corners": corners(s.epsg, x, y)}
    (od / "clouds.json").write_text(json.dumps(meta))
    return meta


# ------------------------------------------------------------------------------------------------ hotspots + place
def _top_clusters(mask: np.ndarray, score: np.ndarray, value: np.ndarray, epsg: int, x: np.ndarray, y: np.ndarray,
                  elev: np.ndarray | None, conf: np.ndarray | None, n: int = 5, min_px: int = 3) -> list[dict]:
    """Largest/strongest connected clusters of a result mask -> point hotspots (lat, lon, elevation, area, value)."""
    from scipy import ndimage

    lab, k = ndimage.label(mask)
    if k == 0:
        return []
    idx = np.arange(1, k + 1)
    sizes = ndimage.sum(np.ones_like(lab), lab, idx)
    strength = ndimage.sum(np.nan_to_num(score), lab, idx)
    order = [i for i in np.argsort(-strength) if sizes[i] >= min_px][:n]
    px_area = abs((x[1] - x[0]) * (y[1] - y[0]))
    tr = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    out = []
    for i in order:
        m = lab == idx[i]
        rr, cc = np.nonzero(m)
        k = int(np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2))  # a member pixel nearest the centre
        lon, lat = tr(float(x[cc[k]]), float(y[rr[k]]))
        c = None
        if conf is not None:
            vals = conf[m][conf[m] > 0]
            c = {3: "High", 2: "Medium", 1: "Low"}.get(int(np.round(np.median(vals))), None) if vals.size else None
        out.append({"lat": round(lat, 5), "lon": round(lon, 5), "area_km2": round(float(sizes[i] * px_area / 1e6), 3),
                    "value": round(float(np.nanmedian(value[m])), 2),
                    "elevation_m": round(float(np.nanmedian(elev[m])), 0) if elev is not None and np.isfinite(elev[m]).any() else None,
                    "confidence": c})
    return out


def export_hotspots(region: Region) -> dict[str, int]:
    wd = WORK_DIR / region.id
    res = {}
    stacks = sorted((wd / "gdm").glob("stack_*.npz")) if (wd / "gdm").exists() else []
    if stacks:
        z = np.load(stacks[-1])
        v = z["p_rate_per_yr"] * 1000
        se = (z["p_rate_se_eff"] if "p_rate_se_eff" in z.files else z["p_rate_se"]) * 1000
        rel = z["p_reliable"] > 0.5 if "p_reliable" in z.files else np.isfinite(v)
        hot = rel & np.isfinite(v) & (np.abs(v) > np.maximum(2 * se, 20))
        hs = _top_clusters(hot, np.abs(v), v, int(z["epsg"]), z["x"], z["y"], z["elev"],
                           z["p_confidence_class"] if "p_confidence_class" in z.files else None, min_px=1)
        for h in hs:
            h["kind"] = "ground"
            h["confidence"] = "Low"  # single viewing direction, local cluster: needs ground checking
            h["text"] = (f"A small patch ({h['area_km2'] * 100:.0f} ha) moving about {abs(h['value']) / 10:.0f} cm a year "
                         f"{'away from' if h['value'] < 0 else 'toward'} the satellite - possibly slope creep or a small landslide, "
                         f"possibly a processing artefact. Worth a ground check.")
        _write_json(WEB_DATA_DIR / region.id / "gdm" / "hotspots.json", {"stack": stacks[-1].stem, "units": "mm/yr", "hotspots": hs})
        res["gdm"] = len(hs)
    if (wd / "cdm" / "goff_velocity.npz").exists():
        z = np.load(wd / "cdm" / "goff_velocity.npz")
        sp = z["speed"]
        gl = z["main"] if z["main"].any() else z["glacier"]
        ok = gl & np.isfinite(sp)
        thr = float(np.nanpercentile(sp[ok], 90)) if ok.any() else 30
        fast = ok & (sp > thr)
        hs = _top_clusters(fast, sp, sp, int(z["epsg"]), z["x"], z["y"], z["elev"],
                           z["confidence_class"] if "confidence_class" in z.files else None, n=4)
        for h in hs:
            h["kind"] = "glacier"
            h["text"] = f"Fast-flowing ice: about {h['value']:.0f} m a year ({h['value'] / 365.25 * 100:.0f} cm a day)."
        # terminus as a landmark
        s = json.loads((wd / "cdm" / "summary.json").read_text()).get("goff", {})
        if s.get("terminus"):
            t = s["terminus"]
            hs.append({"kind": "glacier", "lat": t["lat"], "lon": t["lon"], "elevation_m": t["elevation_m"], "area_km2": None,
                       "value": t.get("speed_m_yr"), "confidence": "Low", "landmark": "snout",
                       "text": f"The glacier snout (approximate), {t['elevation_m']:.0f} m - where the ice melts into the river."})
        _write_json(WEB_DATA_DIR / region.id / "cdm" / "hotspots.json", {"units": "m/yr", "hotspots": hs})
        res["cdm"] = len(hs)
    if (wd / "bdm" / "bdm.npz").exists():
        z = np.load(wd / "bdm" / "bdm.npz")
        lasting = z["lasting"] if "lasting" in z.files else z["disturbance"] & ~z["recovered"]
        dates = [str(d) for d in z["dates"]]
        fd = z["first_dark"]
        conf = z["confidence"]
        hs = _top_clusters(lasting, conf.astype(float), fd.astype(float), int(z["epsg"]), z["x"], z["y"], z["elev"], conf)
        summ = json.loads((wd / "bdm" / "summary.json").read_text())
        so = summ.get("second_opinion") or {}
        for h in hs:
            h["kind"] = "forest"
            i = int(round(h["value"])) if np.isfinite(h["value"]) else -1
            h["first_seen"] = dates[i] if 0 <= i < len(dates) else None
            h["text"] = (f"Forest patch of {h['area_km2'] * 100:.0f} ha whose radar echo dropped"
                         f"{' from ' + h['first_seen'] if h['first_seen'] else ''} and did not recover"
                         f"{' - not confirmed by other radar measures' if so and not so.get('supports_structural_change') else ''}.")
        _write_json(WEB_DATA_DIR / region.id / "bdm" / "hotspots.json", {"hotspots": hs})
        res["bdm"] = len(hs)
    return res


def _reverse_geocode(lat: float, lon: float) -> dict[str, str]:
    """District / state from OpenStreetMap Nominatim (cached; generic client name, 1 request per region)."""
    import requests

    from earthpulse.config import DATA_DIR
    cache = DATA_DIR / "external" / "geocode_cache.json"
    store = json.loads(cache.read_text()) if cache.exists() else {}
    key = f"{lat:.3f},{lon:.3f}"
    if key not in store:
        try:
            r = requests.get("https://nominatim.openstreetmap.org/reverse",
                             params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10, "accept-language": "en"},
                             headers={"User-Agent": "himalayan-earthpulse/1.0 (research prototype)"}, timeout=30)
            a = r.json().get("address", {})
            store[key] = {"place": a.get("village") or a.get("town") or a.get("city") or a.get("hamlet") or "",
                          "district": a.get("state_district") or a.get("county") or "", "state": a.get("state") or "",
                          "country": a.get("country") or ""}
        except Exception as e:
            log.warning("reverse geocode failed: %s", e)
            return {}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(store, indent=1))
    return store[key]


def export_place(region: Region) -> dict[str, Any]:
    from pyproj import Geod

    from earthpulse.geo import ancillary
    c = region.geometry.centroid
    lon0, lat0, lon1, lat1 = region.geometry.bounds
    area = abs(Geod(ellps="WGS84").geometry_area_perimeter(region.geometry)[0]) / 1e6
    epsg = 32600 + int((c.x + 180) // 6) + 1
    tr = Transformer.from_crs(4326, epsg, always_xy=True).transform
    (x0, y0), (x1, y1) = tr(lon0, lat0), tr(lon1, lat1)
    x = np.linspace(x0, x1, 120)
    y = np.linspace(y1, y0, 120)
    elev = None
    try:
        elev = ancillary.dem(epsg, x, y)
    except Exception as e:
        log.warning("DEM for place failed: %s", e)
    place = {"region": region.id, "name": region.name, "lat": round(c.y, 4), "lon": round(c.x, 4),
             "bbox": [round(v, 4) for v in region.geometry.bounds], "area_km2": round(area, 1),
             "elevation_m": None if elev is None or not np.isfinite(elev).any() else
             {"min": round(float(np.nanpercentile(elev, 1)), 0), "max": round(float(np.nanpercentile(elev, 99)), 0),
              "median": round(float(np.nanmedian(elev)), 0)},
             **_reverse_geocode(c.y, c.x)}
    _write_json(WEB_DATA_DIR / region.id / "place.json", place)
    return place
