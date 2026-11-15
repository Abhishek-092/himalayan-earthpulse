"""Auto-derive everything about a region from its footprint alone.

* NISAR stacks per product (best-covered ascending + descending track/frame)
* land-cover composition (ESA WorldCover), glacier presence (RGI 7 / WorldCover snow-ice), relief (Copernicus DEM)
* model suggestion: CDM if glaciers cover >= 3% of the AOI, BDM if tree cover >= 15%, GDM if interferograms exist
  and the AOI has settlements or steep slopes (where deformation matters)
* InSAR reference strategy (GDM picks the actual pixels from data in its two-pass procedure)
Overrides in sites.yaml (models, reference_bbox, dates, tracks) always win.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pyproj import CRS, Transformer

from earthpulse.catalog.discover import summarize
from earthpulse.catalog.stacks import region_granules, select_stacks
from earthpulse.config import Region
from earthpulse.geo import ancillary

log = logging.getLogger(__name__)


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def resolve_region(region: Region, grid_m: float = 100.0) -> dict[str, Any]:
    gs = region_granules(region, refresh=False)
    avail = summarize(gs)
    stacks = {p: {k: {"count": len(v), "first": v[0].ref_time[:10], "last": (v[-1].sec_time or v[-1].ref_time)[:10],
                      "direction": v[0].direction, "track": v[0].track, "frame": v[0].frame}
                  for k, v in select_stacks(gs, p).items()} for p in ("GCOV", "GUNW", "GOFF")}
    lon, lat = region.geometry.centroid.x, region.geometry.centroid.y
    epsg = utm_epsg(lon, lat)
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    w, s, e, n = region.bbox
    xs, ys = tr.transform([w, e, e, w], [s, s, n, n])
    x = np.arange(min(xs), max(xs), grid_m)
    y = np.arange(max(ys), min(ys), -grid_m)
    wc = ancillary.worldcover(epsg, x, y)
    elev = ancillary.dem(epsg, x, y)
    slope, _ = ancillary.slope_aspect(elev, grid_m, -grid_m)
    valid = wc > 0
    frac = {name: float(((wc == code) & valid).sum() / max(valid.sum(), 1)) for code, name in
            ancillary.WORLDCOVER_CLASSES.items()}
    try:
        outlines = ancillary.glacier_outlines(region.bbox)
        gmask = ancillary.rasterize_geoms([g for _, _, g in outlines], epsg, x, y)
        glacier_frac = float(gmask.mean())
        glacier_names = sorted({nm for _, nm, _ in outlines if nm})[:10]
        glacier_source = "RGI 7.0"
    except Exception as ex:
        log.warning("RGI unavailable (%s) - using WorldCover snow/ice", ex)
        glacier_frac, glacier_names, glacier_source = frac.get("snow and ice", 0.0), [], "WorldCover"
    steep = float(np.nanmean(slope > 25))
    models = []
    reasons = {}
    if glacier_frac >= 0.03 and stacks["GOFF"]:
        models.append("cdm")
        reasons["cdm"] = f"glaciers cover {glacier_frac:.0%} of the area and GOFF offsets exist"
    if frac.get("tree cover", 0) >= 0.15 and stacks["GCOV"]:
        models.append("bdm")
        reasons["bdm"] = f"tree cover is {frac['tree cover']:.0%} of the area"
    if stacks["GUNW"] and (frac.get("built-up", 0) >= 0.005 or steep >= 0.2):
        models.append("gdm")
        reasons["gdm"] = (f"interferograms available; built-up {frac.get('built-up', 0):.1%}, "
                          f"steep (>25 deg) terrain {steep:.0%}")
    suggested = list(models)
    if region.overrides.get("models"):
        models = list(region.overrides["models"])
    return {
        "id": region.id, "epsg": epsg, "availability": avail, "stacks": stacks,
        "land_cover": {k: round(v, 4) for k, v in frac.items() if v > 0.001},
        "glacier_fraction": round(glacier_frac, 4), "glacier_names": glacier_names, "glacier_source": glacier_source,
        "elevation_m": {"min": float(np.nanmin(elev)), "median": float(np.nanmedian(elev)), "max": float(np.nanmax(elev))},
        "steep_fraction": round(steep, 3),
        "models_suggested": suggested, "models": models, "reasons": reasons,
        "reference": region.overrides.get("reference_bbox") or "auto: coherent, gentle (<20 deg), lowest-velocity "
                                                                 "pixels (two-pass, chosen from the data in GDM)",
    }
