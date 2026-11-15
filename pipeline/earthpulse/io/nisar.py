"""Remote AOI subsetting of NISAR L2 HDF5 products (GCOV / GUNW / GOFF) over HTTPS byte-range requests.

NISAR L2 files are 1-9 GB, but they are chunked (512x512) on a UTM map grid. We open them through an
Earthdata-authenticated fsspec session and read only the chunks intersecting the region, so a full year of data
for a 15 km AOI costs megabytes, not terabytes. Subsets are cached as compressed .npz per granule.

Layer paths follow the NISAR product specification (verified against real granules, CRID P050xx).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pyproj import Transformer

from earthpulse.config import CACHE_DIR, Region

log = logging.getLogger(__name__)
_fs_lock = threading.Lock()
_fs = None


def https_fs():
    """Earthdata-authenticated fsspec HTTPS filesystem (credentials from EARTHDATA_USERNAME/PASSWORD)."""
    global _fs
    with _fs_lock:
        if _fs is None:
            import earthaccess

            auth = earthaccess.login(strategy="environment")
            if not auth or not auth.authenticated:
                raise RuntimeError("Earthdata login failed - check EARTHDATA_USERNAME / EARTHDATA_PASSWORD")
            _fs = earthaccess.get_fsspec_https_session()
        return _fs


@dataclass
class Subset:
    granule_id: str
    product: str
    epsg: int
    x: np.ndarray  # pixel-centre coordinates (m)
    y: np.ndarray
    layers: dict[str, np.ndarray]
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.y), len(self.x))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=self.x, y=self.y, **{f"L_{k}": v for k, v in self.layers.items()},
                            _meta=np.array(json.dumps({"granule_id": self.granule_id, "product": self.product,
                                                       "epsg": self.epsg, "attrs": self.attrs}, default=str)))

    @classmethod
    def load(cls, path: Path) -> "Subset":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["_meta"]))
        layers = {k[2:]: z[k] for k in z.files if k.startswith("L_")}
        return cls(meta["granule_id"], meta["product"], int(meta["epsg"]), z["x"], z["y"], layers, meta["attrs"])


def _window(xc: np.ndarray, yc: np.ndarray, bbox_proj: tuple[float, float, float, float], pad: int = 2):
    xmin, ymin, xmax, ymax = bbox_proj
    ix = np.where((xc >= xmin) & (xc <= xmax))[0]
    iy = np.where((yc >= ymin) & (yc <= ymax))[0]
    if ix.size == 0 or iy.size == 0:
        return None
    return (slice(max(iy[0] - pad, 0), min(iy[-1] + pad + 1, len(yc))),
            slice(max(ix[0] - pad, 0), min(ix[-1] + pad + 1, len(xc))))


def region_bbox_in(region: Region, epsg: int, margin_m: float = 0.0) -> tuple[float, float, float, float]:
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    w, s, e, n = region.bbox
    xs, ys = tr.transform([w, e, e, w, (w + e) / 2], [s, s, n, n, (s + n) / 2])
    return (min(xs) - margin_m, min(ys) - margin_m, max(xs) + margin_m, max(ys) + margin_m)


def _scalar(h: h5py.File, path: str, default: Any = None) -> Any:
    try:
        v = h[path][()]
        if isinstance(v, bytes):
            return v.decode()
        if isinstance(v, np.ndarray) and v.dtype.kind == "S":
            return [s.decode() for s in v.ravel()]
        return v.item() if hasattr(v, "item") else v
    except KeyError:
        return default


def _open(url: str, retries: int = 3) -> h5py.File:
    last = None
    for a in range(retries):
        try:
            f = https_fs().open(url, block_size=2 * 1024 * 1024, cache_type="blockcache")
            return h5py.File(f, "r")
        except Exception as e:  # transient network/auth issues
            last = e
            time.sleep(3 * (a + 1))
    raise RuntimeError(f"cannot open {url}: {last}")


# ---------------------------------------------------------------------------------------------------------
# Product-specific readers
# ---------------------------------------------------------------------------------------------------------
GUNW_ROOT = "science/LSAR/GUNW"
GUNW_UNW = f"{GUNW_ROOT}/grids/frequencyA/unwrappedInterferogram"
GUNW_CUBE = f"{GUNW_ROOT}/metadata/radarGrid"
GUNW_LAYERS = ("unwrappedPhase", "coherenceMagnitude", "connectedComponents", "ionospherePhaseScreen",
               "ionospherePhaseScreenUncertainty")
GEOMETRY_CUBES = ("losUnitVectorX", "losUnitVectorY", "incidenceAngle", "alongTrackUnitVectorX",
                  "alongTrackUnitVectorY")


def _read_geometry(h: h5py.File, cube_root: str, region: Region, epsg: int, layers: dict, attrs: dict,
                   names: tuple[str, ...] = GEOMETRY_CUBES) -> None:
    """Read acquisition-geometry cubes once per stack (identical for repeat passes), at the two cube heights
    bracketing the AOI's typical elevation range. Each remote read costs ~5 s latency, so keep this minimal."""
    hs = h[f"{cube_root}/heightAboveEllipsoid"][()]
    cx = h[f"{cube_root}/xCoordinates"][()]
    cy = h[f"{cube_root}/yCoordinates"][()]
    cwin = _window(cx, cy, region_bbox_in(region, epsg, margin_m=3000), pad=2)
    lo = int(np.argmin(np.abs(hs - 1000.0)))
    hi = int(np.argmin(np.abs(hs - 5000.0)))
    idx = sorted({lo, hi})
    attrs.update({"heights": hs[idx].tolist(), "cube_x": cx[cwin[1]].tolist(), "cube_y": cy[cwin[0]].tolist()})
    for name in names:
        p = f"{cube_root}/{name}"
        if p in h:
            layers[f"cube_{name}"] = np.stack([h[p][(i,) + cwin] for i in idx]).astype(np.float32)


def _first_pol(h: h5py.File, base: str) -> str:
    for p in ("HH", "VV", "HV", "VH"):
        if f"{base}/{p}" in h:
            return p
    raise KeyError(f"no polarisation group under {base}")


def read_gunw(url: str, granule_id: str, region: Region, geometry: bool = False) -> Subset:
    with _open(url) as h:
        pol = _first_pol(h, GUNW_UNW)
        grp = h[f"{GUNW_UNW}/{pol}"]
        epsg = int(_scalar(h, f"{GUNW_UNW}/{pol}/projection") or _scalar(h, f"{GUNW_UNW}/projection"))
        xc = grp["xCoordinates"][()]
        yc = grp["yCoordinates"][()]
        win = _window(xc, yc, region_bbox_in(region, epsg))
        if win is None:
            raise ValueError("AOI outside granule grid")
        layers = {}
        for name in GUNW_LAYERS:
            if name in grp:
                layers[name] = grp[name][win].astype(np.float32 if name != "connectedComponents" else np.uint16)
        if f"{GUNW_UNW}/mask" in h:
            layers["mask"] = h[f"{GUNW_UNW}/mask"][win]
        attrs: dict[str, Any] = {
            "polarization": pol,
            "pass_direction": _scalar(h, "science/LSAR/identification/orbitPassDirection"),
            "reference_start": _scalar(h, "science/LSAR/identification/referenceZeroDopplerStartTime"),
            "secondary_start": _scalar(h, "science/LSAR/identification/secondaryZeroDopplerStartTime"),
            "x_spacing": float(xc[1] - xc[0]), "y_spacing": float(yc[1] - yc[0]),
            "center_frequency_hz": _scalar(h, f"{GUNW_ROOT}/grids/frequencyA/centerFrequency"),
        }
        if geometry:
            _read_geometry(h, GUNW_CUBE, region, epsg, layers, attrs)
            pi = f"{GUNW_ROOT}/metadata/processingInformation"
            keys = {"algorithms/interferogramFormation/wrappedInterferogramFiltering": "phase_filter",
                    "algorithms/unwrapping/unwrappingAlgorithm": "unwrapper", "algorithms/unwrapping/costMode": "unwrap_cost",
                    "algorithms/ionosphereEstimation/ionosphereAlgorithm": "ionosphere_algorithm",
                    "algorithms/ionosphereEstimation/unwrappingErrorCorrection": "ionosphere_unwrap_correction",
                    "parameters/geocoding/wetTroposphericCorrectionApplied": "wet_troposphere_applied",
                    "parameters/geocoding/hydrostaticTroposphericCorrectionApplied": "hydrostatic_troposphere_applied",
                    "parameters/unwrappedInterferogram/frequencyA/topographicFlatteningApplied": "topographic_flattening",
                    "parameters/unwrappedInterferogram/frequencyA/numberOfAzimuthLooks": "azimuth_looks",
                    "parameters/unwrappedInterferogram/frequencyA/numberOfRangeLooks": "range_looks",
                    "algorithms/coregistration/coregistrationMethod": "coregistration"}
            attrs["processing"] = {v: _scalar(h, f"{pi}/{k}") for k, v in keys.items()}
    return Subset(granule_id, "GUNW", epsg, xc[win[1]], yc[win[0]], layers, attrs)


GCOV_GRID = "science/LSAR/GCOV/grids/frequencyA"
GCOV_TERMS = ("HHHH", "HVHV", "VVVV", "VHVH", "HHHV", "HHVV", "HVVV")
GCOV_AUX = ("numberOfLooks", "rtcGammaToSigmaFactor", "mask")


def read_gcov(url: str, granule_id: str, region: Region, geometry: bool = False,
              terms: tuple[str, ...] = GCOV_TERMS) -> Subset:
    with _open(url) as h:
        g = h[GCOV_GRID]
        epsg = int(_scalar(h, f"{GCOV_GRID}/projection"))
        xc = g["xCoordinates"][()]
        yc = g["yCoordinates"][()]
        win = _window(xc, yc, region_bbox_in(region, epsg))
        if win is None:
            raise ValueError("AOI outside granule grid")
        layers = {}
        listed = _scalar(h, f"{GCOV_GRID}/listOfCovarianceTerms") or []
        listed = [listed] if isinstance(listed, str) else listed
        wanted = [t for t in terms if t in listed] if listed else list(terms)
        for t in wanted + ["mask"] + (["numberOfLooks", "rtcGammaToSigmaFactor"] if geometry else []):
            if t in g and isinstance(g[t], h5py.Dataset) and g[t].ndim == 2:
                arr = g[t][win]
                layers[t] = arr if arr.dtype.kind in "cu" else arr.astype(np.float32)
        attrs = {
            "pass_direction": _scalar(h, "science/LSAR/identification/orbitPassDirection"),
            "start": _scalar(h, "science/LSAR/identification/zeroDopplerStartTime"),
            "x_spacing": float(xc[1] - xc[0]), "y_spacing": float(yc[1] - yc[0]),
            "terms": [t for t in terms if t in layers],
            "center_frequency_hz": _scalar(h, f"{GCOV_GRID}/centerFrequency"),
            "list_of_covariance_terms": _scalar(h, f"{GCOV_GRID}/listOfCovarianceTerms"),
            "look_direction": _scalar(h, "science/LSAR/identification/lookDirection"),
        }
        if geometry:
            _read_geometry(h, "science/LSAR/GCOV/metadata/radarGrid", region, epsg, layers, attrs,
                           names=("losUnitVectorX", "losUnitVectorY", "incidenceAngle"))
            cal = "science/LSAR/GCOV/metadata/calibrationInformation/frequencyA/noiseEquivalentBackscatter"
            if cal in h:
                nx, ny = h[f"{cal}/xCoordinates"][()], h[f"{cal}/yCoordinates"][()]
                nwin = _window(nx, ny, region_bbox_in(region, epsg, margin_m=3000), pad=2)
                for pol in ("HH", "HV"):
                    if f"{cal}/{pol}" in h and nwin is not None:
                        layers[f"nesz_{pol}"] = h[f"{cal}/{pol}"][nwin].astype(np.float32)
                if nwin is not None:
                    attrs["nesz_x"], attrs["nesz_y"] = nx[nwin[1]].tolist(), ny[nwin[0]].tolist()
            pp = "science/LSAR/GCOV/metadata/processingInformation/parameters"
            attrs["processing"] = {k: _scalar(h, f"{pp}/{k}") for k in (
                "noiseCorrectionApplied", "shadowMaskingApplied", "validSamplesSubSwathMaskingApplied",
                "radiometricTerrainCorrectionApplied", "outputBackscatterNormalizationConvention")}
            sw = "science/LSAR/GCOV/metadata/sourceData/swaths/frequencyA"
            attrs["incidence_range_deg"] = [_scalar(h, f"{sw}/nearRangeIncidenceAngle"), _scalar(h, f"{sw}/farRangeIncidenceAngle")]
    return Subset(granule_id, "GCOV", epsg, xc[win[1]], yc[win[0]], layers, attrs)


GOFF_GRID = "science/LSAR/GOFF/grids/frequencyA/pixelOffsets"
GOFF_FIELDS = ("alongTrackOffset", "slantRangeOffset", "alongTrackOffsetVariance", "slantRangeOffsetVariance",
               "correlationSurfacePeak", "snr", "crossOffsetVariance")
GOFF_CUBE = "science/LSAR/GOFF/metadata/radarGrid"


def read_goff(url: str, granule_id: str, region: Region, geometry: bool = False,
              layer: str | None = None) -> Subset:
    with _open(url) as h:
        base = h[GOFF_GRID]
        pol = _first_pol(h, GOFF_GRID)
        pg = base[pol]
        layer_names = sorted(k for k in pg.keys() if k.startswith("layer"))
        layer = layer or (layer_names[-1] if layer_names else None)  # largest window = most robust on ice
        src = pg[layer] if layer else pg
        epsg = int(_scalar(h, f"{GOFF_GRID}/projection") or _scalar(h, f"{GOFF_GRID}/{pol}/projection"))
        xc = (base["xCoordinates"] if "xCoordinates" in base else pg["xCoordinates"])[()]
        yc = (base["yCoordinates"] if "yCoordinates" in base else pg["yCoordinates"])[()]
        win = _window(xc, yc, region_bbox_in(region, epsg))
        if win is None:
            raise ValueError("AOI outside granule grid")
        layers = {f: src[f][win].astype(np.float32) for f in GOFF_FIELDS if f in src}
        if "mask" in base:
            layers["mask"] = base["mask"][win]
        u = src["slantRangeOffset"].attrs.get("units", b"meters") if "slantRangeOffset" in src else b"meters"
        offset_units = u.decode() if isinstance(u, bytes) else str(u)
        ref = "science/LSAR/GOFF/metadata/processingInformation/parameters/reference/frequencyA"
        attrs = {
            "polarization": pol, "layer": layer, "layers_available": layer_names, "offset_units": offset_units,
            "slant_range_spacing": _scalar(h, f"{ref}/slantRangeSpacing"),
            "zero_doppler_time_spacing": _scalar(h, f"{ref}/zeroDopplerTimeSpacing"),
            "pass_direction": _scalar(h, "science/LSAR/identification/orbitPassDirection"),
            "reference_start": _scalar(h, "science/LSAR/identification/referenceZeroDopplerStartTime"),
            "secondary_start": _scalar(h, "science/LSAR/identification/secondaryZeroDopplerStartTime"),
            "x_spacing": float(xc[1] - xc[0]), "y_spacing": float(yc[1] - yc[0]),
            "center_frequency_hz": _scalar(h, "science/LSAR/GOFF/grids/frequencyA/centerFrequency"),
        }
        if geometry:
            _read_geometry(h, GOFF_CUBE, region, epsg, layers, attrs)
    return Subset(granule_id, "GOFF", epsg, xc[win[1]], yc[win[0]], layers, attrs)


SME2_GRID = "science/LSAR/SME2/grids"


def read_sme2(url: str, granule_id: str, region: Region, geometry: bool = False) -> Subset:
    """NISAR L3 soil moisture (EASE-Grid 2.0, EPSG:6933, 200 m): soilMoisture, uncertainty and quality flags."""
    with _open(url) as h:
        g = h[SME2_GRID]
        epsg = int(_scalar(h, f"{SME2_GRID}/projection") or 6933)
        xc = g["xCoordinates"][()]
        yc = g["yCoordinates"][()]
        win = _window(xc, yc, region_bbox_in(region, epsg))
        if win is None:
            raise ValueError("AOI outside granule grid")
        layers = {k: g[k][win] for k in ("soilMoisture", "soilMoistureUncertainty", "retrievalQualityFlag",
                                         "surfaceQualityFlag") if k in g}
        attrs = {"pass_direction": _scalar(h, "science/LSAR/identification/orbitPassDirection"),
                 "look_direction": _scalar(h, "science/LSAR/identification/lookDirection"),
                 "start": _scalar(h, "science/LSAR/identification/zeroDopplerStartTime"),
                 "product_version": _scalar(h, "science/LSAR/identification/productVersion"),
                 "doi": _scalar(h, "science/LSAR/identification/productDoi")}
    return Subset(granule_id, "SME2", epsg, xc[win[1]], yc[win[0]], layers, attrs)


READERS = {"GUNW": read_gunw, "GCOV": read_gcov, "GOFF": read_goff, "SME2": read_sme2}


def cache_path(region_id: str, product: str, granule_id: str, geometry: bool = False) -> Path:
    return CACHE_DIR / region_id / product / f"{granule_id}{'.geom' if geometry else ''}.npz"


def fetch(region: Region, product: str, url: str, granule_id: str, refresh: bool = False,
          geometry: bool = False) -> Subset:
    path = cache_path(region.id, product, granule_id, geometry)
    if path.exists() and not refresh:
        return Subset.load(path)
    t = time.time()
    sub = READERS[product](url, granule_id, region, geometry=geometry)
    sub.save(path)
    log.info("fetched %s %s %s in %.0fs", product, granule_id[:60], sub.shape, time.time() - t)
    return sub


def fetch_many(region: Region, granules: list[dict[str, Any]], workers: int = 12) -> list[Subset]:
    """Parallel fetch in separate *processes*: h5py serialises all HDF5 calls behind a global lock, so threads
    cannot overlap the ~5 s per-request latency. Each process has its own HDF5 library + HTTP session.
    Already-cached granules are loaded locally without spawning work. Failures are logged and skipped."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    todo = [g for g in granules if not cache_path(region.id, g["product"], g["granule_id"]).exists()]
    if todo:
        log.info("fetching %d/%d granules with %d processes", len(todo), len(granules), workers)
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=ctx) as ex:
            futs = {ex.submit(_fetch_worker, region.to_json(), g["product"], g["url"], g["granule_id"]): g
                    for g in todo}
            done = 0
            for fu in as_completed(futs):
                g = futs[fu]
                done += 1
                try:
                    fu.result()
                    log.info("  [%d/%d] %s", done, len(todo), g["granule_id"][:72])
                except Exception as e:
                    log.warning("  [%d/%d] skip %s: %s", done, len(todo), g["granule_id"][:60], e)
    out = []
    for g in granules:
        p = cache_path(region.id, g["product"], g["granule_id"])
        if p.exists():
            out.append(Subset.load(p))
    return out


def _fetch_worker(region_json: dict[str, Any], product: str, url: str, granule_id: str) -> str:
    from shapely.geometry import shape

    region = Region(region_json["id"], region_json["name"], shape(region_json["geometry"]))
    fetch(region, product, url, granule_id)
    return granule_id


def interp_cube(cube: np.ndarray, heights: list[float], cube_x: list[float], cube_y: list[float],
                x: np.ndarray, y: np.ndarray, dem: np.ndarray | float) -> np.ndarray:
    """Interpolate a (height, cy, cx) metadata cube onto the data grid at per-pixel DEM height."""
    from scipy.interpolate import RegularGridInterpolator

    hs = np.asarray(heights, float)
    cy = np.asarray(cube_y, float)
    cx = np.asarray(cube_x, float)
    yo = np.argsort(cy)
    f = RegularGridInterpolator((hs, cy[yo], cx), cube[:, yo, :], bounds_error=False, fill_value=None)
    X, Y = np.meshgrid(x, y)
    Z = np.broadcast_to(np.asarray(dem, float), X.shape)
    Z = np.clip(np.where(np.isfinite(Z), Z, np.nanmedian(Z) if np.ndim(dem) else dem), hs.min(), hs.max())
    return f(np.stack([Z, Y, X], -1)).astype(np.float32)
