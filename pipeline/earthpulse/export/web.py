"""Export pipeline results to static web assets (web/public/data), schema "earthpulse/1".

Per raster layer:
  * <id>.png      colourised RGBA overlay (NaN transparent) for MapLibre image sources
  * <id>.bin      Int16 quantised values (value = raw * scale + offset; -32768 = no data) for click-inspect,
                  histograms and the live threshold slider - all computed in the browser
  * manifest entry: title, units, plain-language "how to read", legend, 4 lon/lat corners, grid metadata
Time series cubes: one PNG frame per date + one .bin cube (T*H*W) for per-pixel time series.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from PIL import Image
from pyproj import Transformer

from earthpulse.catalog.discover import summarize
from earthpulse.catalog.stacks import region_granules
from earthpulse.config import WEB_DATA_DIR, WORK_DIR, get_region, load_regions, load_resolved

log = logging.getLogger(__name__)
SCHEMA = "earthpulse/1"


def corners(epsg: int, x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    """MapLibre image corners: top-left, top-right, bottom-right, bottom-left (lon, lat)."""
    dx, dy = x[1] - x[0], y[1] - y[0]
    xl, xr = x[0] - dx / 2, x[-1] + dx / 2
    yt, yb = y[0] - dy / 2, y[-1] + dy / 2
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    pts = [tr.transform(xl, yt), tr.transform(xr, yt), tr.transform(xr, yb), tr.transform(xl, yb)]
    return [[round(a, 6), round(b, 6)] for a, b in pts]


def _rgba(arr: np.ndarray, cmap: str, vmin: float, vmax: float, categorical: dict | None = None,
          alpha: float = 1.0) -> np.ndarray:
    a = np.asarray(arr, float)
    valid = np.isfinite(a)
    out = np.zeros(a.shape + (4,), np.uint8)
    if categorical:
        for val, color in categorical.items():
            m = valid & (a == float(val))
            c = color.lstrip("#")
            out[m] = [int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), int(255 * alpha)]
        return out
    norm = np.clip((a - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgba = (colormaps[cmap](np.nan_to_num(norm)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, int(255 * alpha), 0)
    return rgba


def legend_stops(cmap: str, n: int = 7) -> list[str]:
    cm = colormaps[cmap]
    return ["#%02x%02x%02x" % tuple(int(255 * c) for c in cm(i / (n - 1))[:3]) for i in range(n)]


def _quantize(arr: np.ndarray, vmin: float, vmax: float) -> tuple[np.ndarray, float, float]:
    span = max(vmax - vmin, 1e-12) * 1.2
    lo = vmin - 0.1 * span / 1.2
    scale = 65000.0 / span  # int = (v - lo) * scale - 32500
    q = np.where(np.isfinite(arr), np.clip((arr - lo) * scale - 32500, -32500, 32500), -32768)
    return q.astype("<i2"), float(scale), float(lo)


class LayerWriter:
    def __init__(self, out_dir: Path, epsg: int, x: np.ndarray, y: np.ndarray):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.epsg, self.x, self.y = epsg, x, y
        self.corners = corners(epsg, x, y)
        self.layers: list[dict[str, Any]] = []

    def raster(self, lid: str, arr: np.ndarray, *, title: str, units: str, cmap: str = "viridis",
               vmin: float | None = None, vmax: float | None = None, symmetric: bool = False, how_to_read: str = "",
               categorical: dict | None = None, labels: dict | None = None, group: str = "", bin: bool = True,
               alpha: float = 1.0, display_scale: float = 1.0, extra: dict | None = None,
               x: np.ndarray | None = None, y: np.ndarray | None = None) -> dict[str, Any]:
        arr = np.asarray(arr, float) * display_scale
        xx = self.x if x is None else x
        yy = self.y if y is None else y
        fin = arr[np.isfinite(arr)]
        if vmin is None or vmax is None:
            if fin.size:
                lo, hi = np.percentile(fin, [2, 98])
            else:
                lo, hi = 0.0, 1.0
            if symmetric:
                m = max(abs(lo), abs(hi), 1e-9)
                lo, hi = -m, m
            vmin = float(lo) if vmin is None else vmin
            vmax = float(hi) if vmax is None else vmax
        Image.fromarray(_rgba(arr, cmap, vmin, vmax, categorical, alpha)).save(self.dir / f"{lid}.png", optimize=True)
        entry: dict[str, Any] = {
            "id": lid, "title": title, "units": units, "group": group, "png": f"{lid}.png",
            "corners": corners(self.epsg, xx, yy), "width": int(arr.shape[1]), "height": int(arr.shape[0]),
            "legend": ({"type": "categorical", "items": [{"value": k, "color": v, "label": (labels or {}).get(k, str(k))}
                                                        for k, v in categorical.items()]}
                       if categorical else {"type": "continuous", "cmap": cmap, "stops": legend_stops(cmap),
                                            "min": round(vmin, 4), "max": round(vmax, 4)}),
            "how_to_read": how_to_read,
            "stats": {"valid_fraction": float(np.isfinite(arr).mean()),
                      "p05": float(np.percentile(fin, 5)) if fin.size else None,
                      "median": float(np.median(fin)) if fin.size else None,
                      "p95": float(np.percentile(fin, 95)) if fin.size else None},
        }
        if bin:
            q, scale, off = _quantize(arr, float(np.min(fin)) if fin.size else 0, float(np.max(fin)) if fin.size else 1)
            q.tofile(self.dir / f"{lid}.bin")
            px = abs(float(xx[1] - xx[0])) * abs(float(yy[1] - yy[0]))
            entry["grid"] = {"bin": f"{lid}.bin", "dtype": "int16", "scale": scale, "offset": off,
                             "nodata": -32768, "pixel_area_m2": px}
        if extra:
            entry.update(extra)
        self.layers.append(entry)
        return entry

    def rgb(self, lid: str, rgb: np.ndarray, *, title: str, how_to_read: str = "", group: str = "context",
            gain: float = 3.5) -> dict[str, Any]:
        a = np.clip(np.nan_to_num(np.moveaxis(rgb, 0, -1)) * gain, 0, 1)
        alpha = np.all(np.isfinite(rgb), axis=0)
        img = np.dstack([(a * 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
        Image.fromarray(img).save(self.dir / f"{lid}.png", optimize=True)
        entry = {"id": lid, "title": title, "units": "", "group": group, "png": f"{lid}.png", "corners": self.corners,
                 "width": int(rgb.shape[2]), "height": int(rgb.shape[1]), "legend": {"type": "rgb"},
                 "how_to_read": how_to_read}
        self.layers.append(entry)
        return entry

    def cube(self, cid: str, cube: np.ndarray, dates: list[str], *, title: str, units: str, cmap: str,
             vmin: float, vmax: float, how_to_read: str = "", display_scale: float = 1.0, group: str = "",
             frames: bool = True, sigma: np.ndarray | None = None) -> dict[str, Any]:
        cube = np.asarray(cube, float) * display_scale
        T, H, W = cube.shape
        frame_files = []
        if frames:
            for t in range(T):
                fn = f"{cid}_{t:03d}.png"
                Image.fromarray(_rgba(cube[t], cmap, vmin, vmax)).save(self.dir / fn, optimize=True)
                frame_files.append(fn)
        fin = cube[np.isfinite(cube)]
        q, scale, off = _quantize(cube, float(fin.min()) if fin.size else 0, float(fin.max()) if fin.size else 1)
        q.tofile(self.dir / f"{cid}.bin")
        entry = {"id": cid, "title": title, "units": units, "group": group, "kind": "cube", "dates": dates,
                 "frames": frame_files, "corners": self.corners, "width": W, "height": H,
                 "legend": {"type": "continuous", "cmap": cmap, "stops": legend_stops(cmap), "min": vmin, "max": vmax},
                 "how_to_read": how_to_read,
                 "grid": {"bin": f"{cid}.bin", "dtype": "int16", "scale": scale, "offset": off, "nodata": -32768,
                          "shape": [T, H, W]}}
        if sigma is not None:
            s = np.asarray(sigma, float) * display_scale
            qs, ss, so = _quantize(s, 0.0, float(np.nanmax(s)) if np.isfinite(s).any() else 1.0)
            qs.tofile(self.dir / f"{cid}_sigma.bin")
            entry["sigma_grid"] = {"bin": f"{cid}_sigma.bin", "dtype": "int16", "scale": ss, "offset": so,
                                   "nodata": -32768, "shape": [T, H, W]}
        self.layers.append(entry)
        return entry


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=_json_default, separators=(",", ":")))


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _clean(obj):
    """Replace NaN/inf with None recursively (valid JSON)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ================================================================================================ GDM
def export_gdm(region_id: str) -> dict[str, Any] | None:
    from earthpulse.temporal.framework import BEHAVIOUR_LABELS

    wd = WORK_DIR / region_id / "gdm"
    if not (wd / "summary.json").exists():
        return None
    summary = json.loads((wd / "summary.json").read_text())
    od = WEB_DATA_DIR / region_id / "gdm"
    shutil.rmtree(od, ignore_errors=True)
    layers: list[dict[str, Any]] = []
    for key in summary["stacks"]:
        z = np.load(wd / f"stack_{key}.npz")
        x, y, epsg = z["x"], z["y"], int(z["epsg"])
        w = LayerWriter(od, epsg, x, y)
        geom = "ascending" if key.startswith("A") else "descending"
        vel_all = z["p_rate_per_yr"]
        rel = z["p_reliable"] > 0.5 if "p_reliable" in z.files else np.isfinite(vel_all)
        vel = np.where(rel, vel_all, np.nan)
        lim = 50.0
        w.raster(f"{key}_velocity", vel, title=f"Ground velocity towards the satellite ({geom} pass)", units="mm/yr",
                 cmap="RdBu", vmin=-lim, vmax=lim, display_scale=1000, group="Movement",
                 how_to_read="Reliable pixels only (settlements, bare rock and soil, very coherent ground). Red = "
                             "moving away from the satellite (usually sinking or sliding downslope); blue = moving "
                             "towards it; white = no measurable movement. Line-of-sight component, one viewing direction.")
        w.raster(f"{key}_velocity_all", vel_all, title=f"Diagnostic: all pixels incl. vegetation bias ({geom})",
                 units="mm/yr", cmap="RdBu", vmin=-300, vmax=300, display_scale=1000, group="Quality",
                 how_to_read="Every coherent pixel, before the reliability mask. The broad red/blue patches over "
                             "forest and grassland are mostly the L-band phase-closure (fading) bias of 12-day "
                             "interferogram chains, not real ground motion - which is why the main map hides them.")
        w.raster(f"{key}_velocity_se", z["p_rate_se"], title="Velocity uncertainty (1 sigma)", units="mm/yr",
                 cmap="magma_r", vmin=0, vmax=float(min(20, np.nanpercentile(z["p_rate_se"] * 1000, 98))) if np.isfinite(z["p_rate_se"]).any() else 10,
                 display_scale=1000, group="Quality",
                 how_to_read="How much the velocity could be wrong by (68% confidence). Dark = less certain.")
        w.raster(f"{key}_coherence", z["mean_coh"], title="Mean interferometric coherence", units="0-1",
                 cmap="cividis", vmin=0, vmax=1, group="Quality",
                 how_to_read="How similar the radar echoes stayed between passes. Bright = stable, reliable "
                             "surface (rock, buildings). Dark = snow, dense forest or fast change - masked below 0.35.")
        w.raster(f"{key}_acceleration", z["p_acceleration_per_yr2"], title="Deformation acceleration",
                 units="mm/yr²", cmap="PuOr", symmetric=True, display_scale=1000, group="Movement",
                 how_to_read="Is the movement speeding up (same colour as the velocity) or slowing down? "
                             "With one year of data this can include part of a seasonal cycle.")
        cls = z["p_behaviour_class"]
        cat = {-1: "#3a3a3a", 0: "#d9d9d9", 1: "#f4a261", 2: "#d62828", 3: "#2a9d8f", 4: "#457b9d", 5: "#9d4edd",
               6: "#80b918"}
        w.raster(f"{key}_behaviour", cls, title="How the ground behaves over time", units="class",
                 categorical=cat, labels={k: BEHAVIOUR_LABELS[k] for k in cat}, group="Temporal", bin=True,
                 how_to_read="Each pixel's time series classified by the shared temporal framework: steady trend, "
                             "accelerating, slowing, seasonal, episodic, recovering or stable.")
        if "p_velocity_gradient" in z.files:
            w.raster(f"{key}_gradient", z["p_velocity_gradient"], title="Deformation gradient (reliable pixels)", units="mm/yr per km",
                     cmap="magma", vmin=0, vmax=300, group="Movement",
                     how_to_read="How fast the ground velocity changes across space - sharp contrasts mark the edges of a "
                                 "moving block. At 80 m posting this is dominated by measurement noise unless a real block moves.")
        if "p_confidence_class" in z.files:
            w.raster(f"{key}_confidence", z["p_confidence_class"], title="Measurement confidence", units="class", categorical={0: "#d1d5db", 1: "#fde68a", 2: "#93c5fd", 3: "#1e3a8a"},
                 labels={0: "not evaluated", 1: "low", 2: "medium", 3: "high"},
                     group="Quality", how_to_read="High = persistent scatterer with coherence >= 0.7 and < 10 mm/yr uncertainty; "
                     "medium = reliable land cover; low = measured but vegetation-biased; grey = not evaluated (masked).")
        w.raster(f"{key}_onset", z["p_onset_year"], title="When measurable movement began", units="year",
                 cmap="plasma", group="Temporal",
                 how_to_read="Date at which the displacement first exceeded twice its uncertainty for at least two "
                             "consecutive passes.")
        dates = [str(d) for d in z["dates"]]
        disp = np.where(rel[None], z["displacement"], np.nan)
        lim_d = 60.0
        w.cube(f"{key}_displacement", disp, dates, title=f"Cumulative displacement ({geom})", units="mm",
               cmap="RdBu", vmin=-lim_d, vmax=lim_d, display_scale=1000, group="Movement", sigma=np.where(rel[None], z["sigma"], np.nan),
               how_to_read="Total line-of-sight movement since the first pass (reliable pixels). Use the time slider "
                           "to watch it build up; click a pixel for its history with uncertainty.")
        for l in w.layers:
            l["stack"] = key
        layers += w.layers
        # region-level series for the temporal panel: median displacement of reliable settlement pixels
        from earthpulse.geo import ancillary as _anc
        from earthpulse.temporal.framework import TemporalConfig, summarize_series
        wcg = _anc.worldcover(epsg, x, y)
        town = rel & (wcg == 50)
        if town.sum() < 10:
            town = rel
        ser = np.array([np.nanmedian(z["displacement"][k][town]) for k in range(len(dates))])
        sc = np.array([1.4826 * np.nanmedian(np.abs(z["displacement"][k][town] - ser[k])) for k in range(len(dates))])
        sg = np.sqrt((sc / np.sqrt(max(int(town.sum()), 1))) ** 2 + 0.002 ** 2)
        summary["stacks"][key]["region_series"] = {
            "label": "Median line-of-sight displacement of reliable settlement pixels", "units": "mm", "dates": dates,
            "values": np.round(ser * 1000, 2).tolist(), "sigma": np.round(sg * 1000, 2).tolist(), "pixels": int(town.sum()),
            "temporal": summarize_series(np.array(dates, dtype="datetime64[D]"), ser * 1000, sg * 1000,
                                         TemporalConfig(baseline_n=1, k_sigma=2.0))}
    if (wd / "decomposition.npz").exists():
        z = np.load(wd / "decomposition.npz")
        w = LayerWriter(od, int(z["epsg"]), z["x"], z["y"])
        rel = z["reliable"] if "reliable" in z.files else np.isfinite(z["up"])
        up = np.where(rel, z["up"], np.nan)
        lim = 50.0
        w.raster("vertical_velocity", up, title="Vertical velocity (ascending + descending combined)", units="mm/yr",
                 cmap="RdBu", vmin=-lim, vmax=lim, display_scale=1000, group="Movement",
                 how_to_read="Red = sinking, blue = rising (reliable pixels only). Combining two viewing directions "
                             "separates up-down from east-west motion (north-south motion is assumed small).")
        w.raster("east_velocity", np.where(rel, z["east"], np.nan), title="East-west velocity", units="mm/yr", cmap="PiYG", vmin=-lim,
                 vmax=lim, display_scale=1000, group="Movement",
                 how_to_read="Green = moving east, pink = moving west.")
        w.raster("vertical_velocity_se", z["sigma_up"], title="Vertical velocity uncertainty", units="mm/yr",
                 cmap="magma_r", vmin=0, vmax=10, display_scale=1000, group="Quality",
                 how_to_read="Uncertainty of the vertical velocity (1 sigma).")
        layers += w.layers
    manifest = {"model": "gdm", "region": region_id, "layers": layers, "summary": _clean(summary)}
    _write_json(od / "manifest.json", manifest)
    return {"layers": len(layers), "summary": summary}


# ================================================================================================ CDM
def _vectors(x, y, epsg, ve, vn, speed, step: int, mask: np.ndarray, min_speed: float = 3.0) -> dict[str, Any]:
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    feats = []
    px = abs(float(x[1] - x[0]))
    for i in range(step // 2, len(y), step):
        for j in range(step // 2, len(x), step):
            if not mask[i, j] or not np.isfinite(speed[i, j]) or speed[i, j] < min_speed:
                continue
            L = step * px * 0.8
            ux, uy = ve[i, j] / speed[i, j], vn[i, j] / speed[i, j]
            x0, y0 = x[j] - ux * L / 2, y[i] - uy * L / 2
            x1, y1 = x[j] + ux * L / 2, y[i] + uy * L / 2
            # arrow head
            hx, hy = -ux * L * 0.3, -uy * L * 0.3
            a1 = (x1 + hx * 0.8 - uy * L * 0.18, y1 + hy * 0.8 + ux * L * 0.18)
            a2 = (x1 + hx * 0.8 + uy * L * 0.18, y1 + hy * 0.8 - ux * L * 0.18)
            coords = [tr.transform(*p) for p in ((x0, y0), (x1, y1), a1, (x1, y1), a2)]
            feats.append({"type": "Feature", "properties": {"speed": round(float(speed[i, j]), 1),
                                                            "dir": round(float(np.degrees(np.arctan2(ux, uy)) % 360), 0)},
                          "geometry": {"type": "LineString", "coordinates": [[round(a, 6), round(b, 6)] for a, b in coords]}})
    return {"type": "FeatureCollection", "features": feats}


def _outline_geojson(region_id: str) -> dict[str, Any]:
    from shapely.geometry import mapping

    from earthpulse.geo import ancillary

    r = get_region(region_id)
    try:
        outs = ancillary.glacier_outlines(r.bbox)
    except Exception:
        outs = []
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"rgi_id": rid, "name": name}, "geometry": mapping(g.simplify(0.0003))}
        for rid, name, g in outs if g is not None]}


def export_cdm(region_id: str) -> dict[str, Any] | None:
    wd = WORK_DIR / region_id / "cdm"
    if not (wd / "summary.json").exists():
        return None
    summary = json.loads((wd / "summary.json").read_text())
    od = WEB_DATA_DIR / region_id / "cdm"
    shutil.rmtree(od, ignore_errors=True)
    layers = []
    vmax = 80.0
    if (wd / "goff_velocity.npz").exists():
        z = np.load(wd / "goff_velocity.npz")
        x, y, epsg = z["x"], z["y"], int(z["epsg"])
        w = LayerWriter(od, epsg, x, y)
        speed = np.where(z["glacier"] | (z["speed"] > 0), z["speed"], np.nan)
        gl = z["glacier"]
        fin = speed[gl & np.isfinite(speed)]
        vmax = float(max(30, np.ceil(np.percentile(fin, 98) / 10) * 10)) if fin.size else 80.0
        show = np.where(ndimage_dilate(gl, 2), speed, np.nan)
        w.raster("speed_goff", show, title="Glacier surface speed (NISAR GOFF, stacked)", units="m/yr",
                 cmap="inferno", vmin=0, vmax=vmax, group="Movement",
                 how_to_read="How far the ice surface moves in a year. Bright yellow = fastest flow, usually the "
                             "central trunk; dark = slow or stagnant ice near the snout and margins.")
        w.raster("speed_sigma", np.where(gl, z["sigma"], np.nan), title="Speed uncertainty", units="m/yr",
                 cmap="magma_r", vmin=0, vmax=max(5.0, vmax / 4), group="Quality",
                 how_to_read="Spread of the 12-day measurements around the stacked median (standard error).")
        if "confidence_class" in z.files:
            w.raster("speed_confidence", np.where(ndimage_dilate(gl, 2), z["confidence_class"], np.nan), title="Speed confidence",
                     units="class", categorical={0: "#d1d5db", 1: "#fde68a", 2: "#93c5fd", 3: "#1e3a8a"},
                 labels={0: "not evaluated", 1: "low", 2: "medium", 3: "high"}, group="Quality",
                     how_to_read="High = 10+ offset pairs and < 3 m/yr standard error; medium = 5+ pairs and < 8 m/yr; low = fewer "
                                 "pairs or noisier; grey = not evaluated (no valid offsets or layover/shadow).")
        w.raster("flow_direction", np.where(gl & np.isfinite(speed) & (speed > 3), z["flow_dir"], np.nan),
                 title="Flow direction", units="deg from north", cmap="twilight", vmin=0, vmax=360, group="Movement",
                 how_to_read="Compass direction of ice flow (0 = north, 90 = east, 180 = south, 270 = west).")
        if "coherence_mean" in z.files:
            w.raster("glacier_coherence", z["coherence_mean"], title="Interferometric coherence (12-day mean, NISAR GUNW)",
                     units="0-1", cmap="magma", vmin=0, vmax=0.8, group="Radar",
                     how_to_read="Bright = surface stayed the same between passes (stable rock); dark = decorrelated "
                                 "(moving ice, melt, fresh snow). Fast glacier tongues should be dark - an independent check on the speed map.")
        w.raster("speed_itslive", np.where(ndimage_dilate(gl, 2), z["itslive_v"], np.nan),
                 title="Independent check: ITS_LIVE speed (2014-2022)", units="m/yr", cmap="inferno", vmin=0,
                 vmax=vmax, group="Validation",
                 how_to_read="NASA ITS_LIVE velocity from optical + Sentinel-1 images. Same colours as the NISAR "
                             "map: similar patterns mean our measurement is credible.")
        diff = np.where(gl, z["speed"] - z["itslive_v"], np.nan)
        w.raster("speed_minus_itslive", diff, title="NISAR minus ITS_LIVE", units="m/yr", cmap="RdBu_r",
                 symmetric=True, group="Validation",
                 how_to_read="Red = NISAR faster than the 2014-2022 average; blue = slower. Differences combine "
                             "measurement noise with real year-to-year change.")
        for k, m in zip(z["monthly_keys"], z["monthly"]):
            pass
        if len(z["monthly_keys"]) >= 3:
            w.cube("speed_monthly", np.where(ndimage_dilate(gl, 2)[None], z["monthly"], np.nan),
                   [str(k) for k in z["monthly_keys"]], title="Monthly glacier speed", units="m/yr", cmap="inferno",
                   vmin=0, vmax=vmax, group="Temporal",
                   how_to_read="Speed from the 12-day pairs centred in each month - watch for summer speed-up "
                               "when meltwater lubricates the glacier bed.")
        _write_json(od / "vectors.geojson", _vectors(x, y, epsg, z["ve"], z["vn"], z["speed"], 6, gl))
        layers += w.layers
    if (wd / "ncc_velocity.npz").exists():
        z = np.load(wd / "ncc_velocity.npz")
        w = LayerWriter(od, int(z["epsg"]), z["x"], z["y"])
        gl = z["glacier"]
        w.raster("speed_ncc", np.where(ndimage_dilate(gl, 1), z["speed"], np.nan),
                 title="Glacier speed - our own feature tracking (GCOV, ~1 year)", units="m/yr", cmap="inferno",
                 vmin=0, vmax=vmax, group="Movement",
                 how_to_read="Computed by matching crevasse and debris patterns between two NISAR radar images "
                             "almost a year apart - an independent method from GOFF.")
        layers += w.layers
    _write_json(od / "outlines.geojson", _outline_geojson(region_id))
    g = summary.get("goff")
    if g:
        from collections import Counter

        from earthpulse.catalog.discover import parse_granule_name
        cnt: Counter = Counter()
        for gid in g.get("granules", []):
            info = parse_granule_name(gid)
            t0 = np.datetime64(info["ref_time"][:10])
            t1 = np.datetime64(info["sec_time"][:10]) if info["sec_time"] else t0
            cnt[str((t0 + (t1 - t0) // 2).astype("datetime64[M]"))] += 1
        g["pairs_per_month"] = dict(sorted(cnt.items()))
        from earthpulse.temporal.framework import summarize_series
        mm = {k: v for k, v in (g.get("monthly_main_median_m_yr") or {}).items() if v is not None}
        if len(mm) >= 4:
            md = np.array([k + "-15" for k in mm], dtype="datetime64[D]")
            summary["region_series"] = {"label": "Monthly median glacier speed", "units": "m/yr", "dates": [str(d) for d in md],
                                        "values": list(mm.values()), "n": [cnt.get(k, 0) for k in mm],
                                        "temporal": summarize_series(md, np.array(list(mm.values()), float))}
    manifest = {"model": "cdm", "region": region_id, "layers": layers, "summary": _clean(summary),
                "vectors": "vectors.geojson" if (od / "vectors.geojson").exists() else None,
                "outlines": "outlines.geojson"}
    _write_json(od / "manifest.json", manifest)
    return {"layers": len(layers)}


def ndimage_dilate(m: np.ndarray, it: int) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    return binary_dilation(m, iterations=it)


# ================================================================================================ BDM
def export_bdm(region_id: str) -> dict[str, Any] | None:
    wd = WORK_DIR / region_id / "bdm"
    if not (wd / "summary.json").exists():
        return None
    summary = json.loads((wd / "summary.json").read_text())
    od = WEB_DATA_DIR / region_id / "bdm"
    shutil.rmtree(od, ignore_errors=True)
    z = np.load(wd / "bdm.npz")
    x, y, epsg = z["x"], z["y"], int(z["epsg"])
    dates = [str(d) for d in z["dates"]]
    w = LayerWriter(od, epsg, x, y)
    hv = z["hv_db"]
    lo, hi = np.nanpercentile(hv[np.isfinite(hv)], [2, 98])
    w.cube("hv", hv, dates, title="L-band HV radar brightness (forest volume scattering)", units="dB", cmap="gray",
           vmin=float(lo), vmax=float(hi), group="Radar",
           how_to_read="Bright = lots of branches and trunks scattering the radar (healthy forest). Dark = little "
                       "vegetation structure (bare ground, burned or cleared areas, water). Brightness is NOT "
                       "sunlight - the radar makes its own signal.")
    fdk = z["first_dark"]
    cum = np.stack([np.where(z["disturbance"] & (fdk >= 0) & (fdk <= t), 1.0, np.nan) for t in range(len(dates))])
    w.cube("changed_so_far", cum, dates, title="Forest changed so far (cumulative)", units="changed", cmap="autumn",
           vmin=0, vmax=1, group="Change",
           how_to_read="Every forest pixel where a significant HV drop has been detected up to the selected pass. Drag the "
                       "timeline to watch change accumulate - note the February 2026 jump (reversible winter effects) that later recovers.")
    hh = z["hh_db"]
    lo2, hi2 = np.nanpercentile(hh[np.isfinite(hh)], [2, 98])
    rgb = np.stack([np.clip((hh[0] - lo2) / (hi2 - lo2), 0, 1), np.clip((hv[0] - lo) / (hi - lo), 0, 1),
                    np.clip((hh[0] - lo2) / (hi2 - lo2), 0, 1)])
    w.rgb("hhhv_rgb", rgb, title="Radar colour composite (HH magenta, HV green)", gain=1.0, group="Radar",
          how_to_read="Green = forest (strong HV volume scattering); magenta = bare slopes, rock and buildings "
                      "(strong HH surface or double-bounce scattering); dark = water or radar shadow.")
    fd = z["first_dark"].astype(float)
    fd[~z["disturbance"]] = np.nan
    T = len(dates)
    w.raster("disturbance_date", fd, title="Potential forest disturbance - date first detected", units="pass index",
             cmap="plasma", vmin=1, vmax=max(T - 1, 2), group="Change",
             extra={"dates": dates},
             how_to_read="Forest pixels where L-band HV dropped with statistical significance (Wishart omnibus "
                         "test, 1% false-alarm level). Colour = when it first happened. Possible causes: fire, "
                         "landslide, clearing - radar alone cannot tell which.")
    if "lasting" in z.files:
        cls = np.where(z["lasting"], 3.0, np.where(z["snow_affected"], 2.0, np.where(z["disturbance"], 1.0, np.nan)))
        w.raster("change_type", cls, title="What kind of change? lasting vs snow vs recovered", units="class",
                 categorical={1: "#93c5fd", 2: "#e9d5ff", 3: "#b45309"},
                 labels={1: "changed, then recovered", 2: "first seen on a snow date (snow-affected)", 3: "lasting potential disturbance"},
                 group="Change", how_to_read="Separates the forest changes worth checking on the ground (brown, lasting) from "
                 "reversible ones (frost, snow, moisture) that bounced back.")
        qm = np.where(z["noise_floor"], 1.0, np.where(z["shadow"], 2.0, np.where(z["layover"], 3.0, np.nan)))
        w.raster("quality_masks", qm, title="Masked: noise floor / shadow / layover", units="class",
                 categorical={1: "#9ca3af", 2: "#4b5563", 3: "#a78bfa"},
                 labels={1: "HV within 3 dB of instrument noise", 2: "radar shadow", 3: "layover"}, group="Quality",
                 how_to_read="Pixels the radar cannot measure reliably and that are excluded from change detection.")
    conf = z["confidence"].astype(float)
    conf[conf == 0] = np.nan
    w.raster("disturbance_confidence", conf, title="Disturbance confidence", units="level",
             categorical={1: "#ffd166", 2: "#f77f00", 3: "#d62828"}, labels={1: "low", 2: "medium", 3: "high"},
             group="Change", how_to_read="High = very strong statistical change AND texture/brightness agree; "
                                         "medium = one of them; low = statistically significant only.")
    w.raster("p_nochange", -np.log10(np.clip(z["p_nochange"], 1e-12, 1)), title="Change evidence (-log10 p)",
             units="-log10 p", cmap="YlOrRd", vmin=0, vmax=12, group="Change",
             how_to_read="How unlikely the observed brightness changes would be if nothing had changed. "
                         "2 = 1% chance; 6 = one in a million.")
    rvi = z["rvi"]
    d_rvi = np.nanmean(rvi[-3:], 0) - np.nanmean(rvi[:3], 0)
    w.raster("rvi_change", d_rvi, title="Change in radar vegetation index (RVI)", units="ΔRVI", cmap="BrBG",
             symmetric=True, group="Change",
             how_to_read="Brown = less vegetation structure than at the start of the record; green = more.")
    fmask = z["forest"].astype(bool)
    if "hvhh_ratio_pre" in z.files:
        r = np.where(fmask | z["disturbance"], z["hvhh_ratio_pre"], np.nan)
        lo3, hi3 = np.nanpercentile(z["hvhh_ratio_pre"][np.isfinite(z["hvhh_ratio_pre"])], [2, 98])
        w.raster("hvhh_ratio", np.where(np.isfinite(z["hvhh_ratio_pre"]), z["hvhh_ratio_pre"], np.nan), title="HV/HH cross-pol ratio (start of record)",
                 units="dB", cmap="viridis", vmin=float(lo3), vmax=float(hi3), group="Radar",
                 how_to_read="Ratio of cross-pol (HV) to co-pol (HH) brightness. Higher (yellow) = more volume scattering "
                             "from branches and leaves; lower (purple) = surface or building scattering.")
        w.raster("hvhh_ratio_change", np.where(fmask, z["hvhh_ratio_change"], np.nan), title="Change in HV/HH ratio (forest)",
                 units="dB", cmap="BrBG", symmetric=True, group="Change",
                 how_to_read="Brown = forest became less 'volume-like' (fewer scatterers: loss, damage, or frozen/dry canopy); "
                             "green = more. Independent of absolute calibration because it is a ratio.")
    for k, title, txt in (("contrast", "GLCM contrast change", "texture contrast (local brightness differences)"),
                          ("homogeneity", "GLCM homogeneity change", "texture smoothness"),
                          ("energy", "GLCM energy change", "texture uniformity (repeated patterns)"),
                          ("correlation", "GLCM correlation change", "linear structure of the texture"),
                          ("dissimilarity", "GLCM dissimilarity change", "average grey-level difference between neighbours")):
        key = f"glcm_{k}_change"
        if key in z.files:
            a = np.where(fmask, z[key], np.nan)
            v = float(np.nanpercentile(np.abs(a[np.isfinite(a)]), 98)) if np.isfinite(a).any() else 1.0
            w.raster(f"glcm_{k}", a, title=f"{title} (HV texture, 7x7)", units="Δ", cmap="PuOr", vmin=-v, vmax=v,
                     group="Texture", how_to_read=f"Change in {txt} between the first and last passes, over forest. "
                     "Canopy gaps make the texture rougher (more contrast/dissimilarity, less homogeneity/energy).")
    if "coherence_mean" in z.files:
        w.raster("coherence_mean", z["coherence_mean"], title="Interferometric coherence (12-day mean, NISAR GUNW)",
                 units="0-1", cmap="magma", vmin=0, vmax=0.8, group="Radar",
                 how_to_read="How similar the radar echo stayed between passes 12 days apart. Bright = stable (bare rock, "
                             "buildings); dark = changing surface (forest canopy in wind, water, snow, fields).")
        w.raster("coherence_change", np.where(fmask, z["coherence_change"], np.nan), title="Change in coherence (forest, last third vs first third of pairs)",
                 units="Δ", cmap="RdBu", symmetric=True, group="Change",
                 how_to_read="Red = forest echo became more stable (often canopy loss exposing ground, or dry/frozen "
                             "conditions); blue = less stable (wetter or denser canopy). Seasonal effects dominate over one year.")
    forest = z["forest"].astype(float)
    forest[forest == 0] = np.nan
    w.raster("forest_mask", forest, title="Forest (ESA WorldCover 2021 tree cover)", units="",
             categorical={1: "#2d6a4f"}, labels={1: "tree cover"}, group="Context", alpha=0.6, bin=False,
             how_to_read="Where the analysis is applied.")
    if (wd / "s2_validation.npz").exists():
        s = np.load(wd / "s2_validation.npz")
        w.raster("dnbr", s["dnbr"], title="Independent check: Sentinel-2 burn index change (dNBR)", units="dNBR",
                 cmap="RdYlGn_r", vmin=-0.3, vmax=0.5, group="Validation",
                 how_to_read="Optical satellite evidence: red (> 0.1) = vegetation burned or lost between the two "
                             "windows. Compare with the radar disturbance map.")
        w.rgb("s2_pre", s["pre_rgb"], title="Sentinel-2 true colour (before)", group="Validation")
        w.rgb("s2_post", s["post_rgb"], title="Sentinel-2 true colour (after)", group="Validation")
    # speckle demo crop (Learn page): raw vs filtered first-date HV
    H, W = z["hv_raw_first"].shape
    cy, cx, hs = H // 2, W // 2, min(150, H // 2, W // 2)
    sl = np.s_[cy - hs:cy + hs, cx - hs:cx + hs]
    for name, arr in (("speckle_raw", z["hv_raw_first"][sl]), ("speckle_filtered", z["hv_filt_first"][sl])):
        Image.fromarray(_rgba(arr, "gray", float(lo), float(hi))).save(od / f"{name}.png")
    from earthpulse.temporal.framework import TemporalConfig, summarize_series
    fh = summary.get("series", {}).get("forest_hv_db")
    if fh and all(v is not None for v in fh):
        summary["region_series"] = {"label": "Median HV of undisturbed forest", "units": "dB", "dates": dates,
                                    "values": fh, "temporal": summarize_series(np.array(dates, dtype="datetime64[D]"),
                                                                               np.array(fh, float), None, TemporalConfig(baseline_n=3))}
    manifest = {"model": "bdm", "region": region_id, "layers": w.layers, "summary": _clean(summary),
                "speckle_demo": {"raw": "speckle_raw.png", "filtered": "speckle_filtered.png",
                                 "date": dates[0]}}
    _write_json(od / "manifest.json", manifest)
    return {"layers": len(w.layers)}


# ================================================================================================ EPITN
def export_epitn(region_id: str) -> dict[str, Any] | None:
    wd = WORK_DIR / region_id / "epitn"
    if not (wd / "network.json").exists():
        return None
    net = json.loads((wd / "network.json").read_text())
    od = WEB_DATA_DIR / region_id / "epitn"
    shutil.rmtree(od, ignore_errors=True)
    z = np.load(wd / "layers.npz")
    w = LayerWriter(od, int(z["epsg"]), z["x"], z["y"])
    w.raster("overlap", z["overlap_code"].astype(float), title="Where processes overlap", units="",
             categorical={1: "#2a9d8f", 2: "#e76f51", 3: "#f4a261", 4: "#264653", 5: "#9d4edd", 6: "#e9c46a",
                          7: "#d62828"},
             labels={1: "forest change only", 2: "ground movement only", 3: "forest change + ground movement",
                     4: "glacier flow only", 5: "glacier + forest", 6: "glacier + ground", 7: "all three"},
             group="Interaction", how_to_read="Cells (on a common grid) where more than one Earth process was "
                                               "detected. Overlap is an association to investigate, not proof "
                                               "that one process caused the other.")
    _write_json(od / "manifest.json", {"model": "epitn", "region": region_id, "layers": w.layers,
                                       "network": _clean(net)})
    return {"layers": len(w.layers)}


# ================================================================================================ catalog
MODEL_EXPORTERS = {"gdm": export_gdm, "cdm": export_cdm, "bdm": export_bdm, "epitn": export_epitn}


def export_all(region_ids: list[str] | None = None) -> dict[str, Any]:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    resolved = load_resolved()["regions"]
    regions_out = []
    cat_path = WEB_DATA_DIR / "catalog.json"
    old = json.loads(cat_path.read_text()) if cat_path.exists() else {"regions": []}
    old_by_id = {r["id"]: r for r in old.get("regions", [])}
    targets = set(region_ids) if region_ids else None
    for r in load_regions():
        if targets is not None and r.id not in targets and r.id in old_by_id:
            regions_out.append(old_by_id[r.id])
            continue
        try:
            gs = region_granules(r)
            avail = summarize(gs)
        except Exception as e:
            log.warning("discovery failed for %s: %s", r.id, e)
            avail = {}
        models = {}
        for m, fn in MODEL_EXPORTERS.items():
            res = fn(r.id)
            if res is not None:
                models[m] = {"manifest": f"{r.id}/{m}/manifest.json", "layers": res["layers"]}
        radar = None
        findings = {}
        if models:
            from earthpulse.export.extras import export_findings, export_hotspots, export_place, export_radar
            try:
                radar = export_radar(r) and f"{r.id}/radar/radar.json"
                findings = export_findings(r)
            except Exception as ex:
                log.warning("extras failed for %s: %s", r.id, ex)
            try:
                export_hotspots(r)
                export_place(r)
            except Exception as ex:
                log.warning("hotspots/place failed for %s: %s", r.id, ex)
        regions_out.append({
            **r.to_json(),
            "centroid": [round(r.geometry.centroid.x, 5), round(r.geometry.centroid.y, 5)],
            "availability": {p: {"count": v["count"], "first": v["first"][:10], "last": v["last"][:10],
                                 "stacks": len(v["stacks"])} for p, v in avail.items()},
            "resolved": resolved.get(r.id, {}).get("models_suggested"),
            "models": models,
            "status": "processed" if models else "catalogued",
            "radar": radar,
            "findings": findings,
            "place": f"{r.id}/place.json" if (WEB_DATA_DIR / r.id / "place.json").exists() else None,
            "verification": f"{r.id}/verification.json" if (WEB_DATA_DIR / r.id / "verification.json").exists() else None,
            "hazards": f"{r.id}/hazards/manifest.json" if (WEB_DATA_DIR / r.id / "hazards" / "manifest.json").exists() else None,
        })
        log.info("exported %s: %s", r.id, list(models))
    catalog = {"schema": SCHEMA, "generated": datetime.utcnow().isoformat() + "Z", "mission": "NISAR (NASA-ISRO)",
               "regions": regions_out}
    _write_json(cat_path, _clean(catalog))
    return catalog
