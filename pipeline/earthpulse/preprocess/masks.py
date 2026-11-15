"""Shared quality-mask builder used by all models (layover / radar shadow / noise floor / snow dates) + the
per-run processing log ("Preprocessing Manager" of the project plan)."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from typing import Any

import numpy as np

from earthpulse.config import REPO_DIR, WEB_DATA_DIR, WORK_DIR, Region
from earthpulse.io.nisar import Subset, cache_path
from earthpulse.preprocess.geometry import cube_to_grid, layover_shadow
from earthpulse.preprocess.noise import nesz_to_grid


def load_geometry(region_id: str, product: str, members) -> Subset | None:
    for g in members:
        p = cache_path(region_id, product, g.granule_id, geometry=True)
        if p.exists():
            return Subset.load(p)
    return None


def radar_masks(geom: Subset | None, x: np.ndarray, y: np.ndarray, elev: np.ndarray) -> dict[str, Any]:
    """Layover/shadow on the target grid from the product's own LOS/incidence cubes. Empty masks if unavailable."""
    if geom is None or "cube_losUnitVectorX" not in geom.layers or "cube_incidenceAngle" not in geom.layers:
        z = np.zeros(elev.shape, bool)
        return {"layover": z, "shadow": z, "available": False}
    a = geom.attrs
    le = cube_to_grid(geom.layers["cube_losUnitVectorX"], a, x, y, elev)
    ln = cube_to_grid(geom.layers["cube_losUnitVectorY"], a, x, y, elev)
    inc = cube_to_grid(geom.layers["cube_incidenceAngle"], a, x, y, elev)
    m = layover_shadow(elev, float(x[1] - x[0]), float(y[1] - y[0]), le, ln, inc)
    return {**m, "available": True, "incidence_deg": inc,
            "incidence_range": [float(np.nanmin(inc)), float(np.nanmax(inc))]}


def nesz_maps(geom: Subset | None, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    if geom is None or "nesz_x" not in geom.attrs:
        return out
    for pol in ("HH", "HV"):
        k = f"nesz_{pol}"
        if k in geom.layers:
            out[pol] = nesz_to_grid(geom.layers[k], geom.attrs["nesz_x"], geom.attrs["nesz_y"], x, y)
    return out


def snow_dates(region: Region, dates: np.ndarray, threshold_m: float = 0.01) -> dict[str, Any]:
    """ERA5-Land snow depth at the region centroid on each acquisition date (and the 3 days before)."""
    from earthpulse.forcing import openmeteo as om

    d0 = str(np.min(dates) - np.timedelta64(5, "D"))
    d1 = str(np.max(dates))
    sd = om.snow_depth_daily(region.geometry.centroid.y, region.geometry.centroid.x, d0, d1)
    m = dict(zip(sd["time"].astype(str), sd["snow_depth_m"]))
    depth = []
    for d in dates:
        vals = [m.get(str(np.datetime64(d) - np.timedelta64(k, "D")), np.nan) for k in range(0, 4)]
        depth.append(float(np.nanmax(vals)) if np.isfinite(vals).any() else float("nan"))
    depth = np.array(depth)
    return {"dates": [str(d) for d in dates], "snow_depth_m": np.round(depth, 4).tolist(),
            "snow": (depth > threshold_m).tolist(), "threshold_m": threshold_m,
            "grid_elevation_m": sd.get("grid_elevation_m"),
            "note": "ERA5-Land snow depth at the region centroid (max of the pass day and 3 days before). Higher slopes "
                    "can hold snow when the centroid does not, so this is a conservative (minimum) snow flag."}


def git_commit() -> str:
    try:
        h = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(REPO_DIR), "diff", "--quiet", "HEAD", "--", "pipeline/earthpulse"]) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def write_processing_log(region: Region, model: str, members, params: dict[str, Any], ancillary: list[str],
                         qc: dict[str, Any], outputs: list[str], product_processing: dict[str, Any] | None = None) -> dict:
    """Record everything needed to reproduce a model run (inputs, software, parameters, DEM/masks, outputs, QC)."""
    import h5py
    import numpy
    import scipy

    log = {
        "region": region.id, "model": model, "run_utc": datetime.utcnow().isoformat() + "Z",
        "software": {"earthpulse_commit": git_commit(), "python": platform.python_version(), "numpy": numpy.__version__,
                     "scipy": scipy.__version__, "h5py": h5py.__version__, "platform": platform.platform()},
        "inputs": [{"granule": g.granule_id, "product": g.product, "crid": g.crid, "pge_version": getattr(g, "pge_version", ""),
                    "orbit_type": getattr(g, "orbit_type", ""), "date": g.ref_time[:10], "stack": g.stack_key}
                   for g in members],
        "product_processing_flags": product_processing or {},
        "parameters": params, "ancillary": ancillary, "qc": qc, "outputs": outputs,
    }
    d = WORK_DIR / region.id / model
    d.mkdir(parents=True, exist_ok=True)
    (d / "processing_log.json").write_text(json.dumps(log, indent=1, default=str))
    # merged per-region log for the web app
    web = WEB_DATA_DIR / region.id / "processing_log.json"
    merged = json.loads(web.read_text()) if web.exists() else {}
    merged[model] = log
    web.parent.mkdir(parents=True, exist_ok=True)
    web.write_text(json.dumps(merged, default=str))
    return log


LAPSE_K_PER_M = 0.0065


def freeze_maps(region: Region, pass_times_utc: list[str], elev: np.ndarray, threshold_c: float = -1.0) -> dict[str, Any]:
    """Per-pixel frozen-canopy / snow-risk flag at each pass time: ERA5 2 m temperature at the pass hour, adjusted to the
    pixel elevation with the standard lapse rate (6.5 K/km). Frozen vegetation (and fresh snow) lowers L-band HV for
    reasons unrelated to forest loss."""
    from earthpulse.forcing import openmeteo as om

    ts = [np.datetime64(t[:13].replace("T", "T"), "h") for t in pass_times_utc]
    d0 = str(min(ts).astype("datetime64[D]"))
    d1 = str(max(ts).astype("datetime64[D]"))
    tt = om.hourly_temperature(region.geometry.centroid.y, region.geometry.centroid.x, d0, d1)
    tmap = dict(zip(tt["time"], tt["t2m"]))
    grid_t = np.array([tmap.get(t, np.nan) for t in ts])
    z = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    frozen = np.stack([(gt - LAPSE_K_PER_M * (z - tt["grid_elevation_m"])) < threshold_c for gt in grid_t])
    return {"frozen": frozen, "grid_t2m_c": np.round(grid_t, 1).tolist(), "grid_elevation_m": tt["grid_elevation_m"],
            "frozen_fraction_by_date": [round(float(f.mean()), 3) for f in frozen], "threshold_c": threshold_c}
