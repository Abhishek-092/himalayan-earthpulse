"""Open ancillary layers, reprojected onto any target grid (EPSG + pixel-centre coordinates).

* Copernicus DEM GLO-30 (AWS open data, COG)           -> elevation, slope, aspect, layover/shadow helpers
* ESA WorldCover 2021 v200 10 m (AWS open data, COG)    -> forest (class 10), snow/ice (70), built-up (50), water (80)
* JRC Global Surface Water occurrence 1984-2021         -> permanent-water mask
* RGI 7.0 glacier outlines (NSIDC, Earthdata auth)      -> glacier mask (falls back to WorldCover snow/ice)

No API keys are required except RGI (uses the same Earthdata credentials).
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from earthpulse.config import CACHE_DIR

log = logging.getLogger(__name__)
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

WORLDCOVER_CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
                      60: "bare / sparse vegetation", 70: "snow and ice", 80: "permanent water", 90: "wetland",
                      95: "mangroves", 100: "moss and lichen"}


def grid_transform(x: np.ndarray, y: np.ndarray):
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])  # negative for north-up
    return from_origin(float(x[0]) - dx / 2, float(y[0]) - dy / 2, dx, -dy)


def _lonlat_bounds(epsg: int, x: np.ndarray, y: np.ndarray):
    from pyproj import Transformer

    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    X, Y = np.meshgrid([x.min(), x.max()], [y.min(), y.max()])
    lon, lat = tr.transform(X.ravel(), Y.ravel())
    return min(lon), min(lat), max(lon), max(lat)


def _warp(urls: list[str], epsg: int, x: np.ndarray, y: np.ndarray, resampling: Resampling,
          dtype=np.float32, nodata=np.nan) -> np.ndarray:
    dst = np.full((len(y), len(x)), nodata, dtype=dtype)
    for url in urls:
        try:
            with rasterio.open(url) as src:
                tmp = np.full_like(dst, nodata)
                reproject(rasterio.band(src, 1), tmp, src_nodata=src.nodata, dst_transform=grid_transform(x, y),
                          dst_crs=f"EPSG:{epsg}", dst_nodata=nodata, resampling=resampling)
                fill = np.isnan(dst) if np.issubdtype(dtype, np.floating) else (dst == nodata)
                dst[fill] = tmp[fill]
        except Exception as e:  # tile may not exist (ocean) - fine
            log.debug("ancillary tile unavailable %s: %s", url, e)
    return dst


def _cached(name: str, epsg: int, x: np.ndarray, y: np.ndarray, fn):
    key = f"{name}_{epsg}_{x[0]:.0f}_{y[0]:.0f}_{len(x)}x{len(y)}_{abs(x[1]-x[0]):.0f}.npy"
    p = CACHE_DIR / "ancillary" / key
    if p.exists():
        return np.load(p)
    arr = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)
    return arr


def dem(epsg: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    def run():
        w, s, e, n = _lonlat_bounds(epsg, x, y)
        urls = []
        for lat in range(math.floor(s), math.floor(n) + 1):
            for lon in range(math.floor(w), math.floor(e) + 1):
                ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
                ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
                name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
                urls.append(f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif")
        return _warp(urls, epsg, x, y, Resampling.bilinear)
    return _cached("dem", epsg, x, y, run)


def slope_aspect(elev: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (deg) and aspect (deg clockwise from north, direction the slope faces)."""
    gy, gx = np.gradient(elev, abs(dy), abs(dx))  # gy: d/drow (row increases southward if dy<0)
    dz_dn = -gy if dy < 0 else gy
    dz_de = gx
    slope = np.degrees(np.arctan(np.hypot(dz_de, dz_dn)))
    aspect = (np.degrees(np.arctan2(-dz_de, -dz_dn)) + 360) % 360
    return slope.astype(np.float32), aspect.astype(np.float32)


def worldcover(epsg: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    def run():
        w, s, e, n = _lonlat_bounds(epsg, x, y)
        urls = []
        for lat in range(3 * math.floor(s / 3), 3 * math.floor(n / 3) + 1, 3):
            for lon in range(3 * math.floor(w / 3), 3 * math.floor(e / 3) + 1, 3):
                ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
                ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
                urls.append("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                            f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map.tif")
        return _warp(urls, epsg, x, y, Resampling.mode, dtype=np.uint8, nodata=0)
    return _cached("worldcover", epsg, x, y, run)


def water_occurrence(epsg: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    def run():
        w, s, e, n = _lonlat_bounds(epsg, x, y)
        urls = []
        for latn in range(10 * math.ceil(s / 10), 10 * math.ceil(n / 10) + 1, 10):
            for lonw in range(10 * math.floor(w / 10), 10 * math.floor(e / 10) + 1, 10):
                ew = f"{lonw}E" if lonw >= 0 else f"{-lonw}W"
                ns = f"{latn}N" if latn >= 0 else f"{-latn}S"
                urls.append("/vsicurl/https://storage.googleapis.com/global-surface-water/downloads2021/"
                            f"occurrence/occurrence_{ew}_{ns}v1_4_2021.tif")
        return _warp(urls, epsg, x, y, Resampling.average)
    return _cached("water", epsg, x, y, run)


# --- RGI 7 glacier outlines ------------------------------------------------------------------------------
RGI_BASE = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-G"
RGI_REGIONS = {  # (name, lon_min, lat_min, lon_max, lat_max) - coarse selector, RGI first-order regions
    "13_central_asia": (65, 26, 105, 47),
    "14_south_asia_west": (66, 27.5, 81, 38),
    "15_south_asia_east": (79, 26, 100, 32),
}


def _rgi_file(code: str) -> Path | None:
    d = CACHE_DIR / "rgi"
    if os.environ.get("EARTHPULSE_NO_RGI", "1") == "1":  # NSIDC download needs separate app authorisation
        shp = list(d.glob(f"*{code}*/*.shp"))
        return shp[0] if shp else None
    shp = list(d.glob(f"RGI2000-v7.0-G-{code}*/*.shp")) + list(d.glob(f"RGI2000-v7.0-G-{code}*.shp"))
    if shp:
        return shp[0]
    import zipfile

    import earthaccess

    d.mkdir(parents=True, exist_ok=True)
    zname = f"RGI2000-v7.0-G-{code}.zip"
    zpath = d / zname
    try:
        earthaccess.login(strategy="environment")
        sess = earthaccess.get_requests_https_session()
        with sess.get(f"{RGI_BASE}/{zname}", stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(d / zname[:-4])
        shp = list((d / zname[:-4]).rglob("*.shp"))
        return shp[0] if shp else None
    except Exception as e:
        log.warning("RGI download failed (%s); falling back to WorldCover snow/ice", e)
        return None


def glacier_outlines(lonlat_bbox: tuple[float, float, float, float]):
    """Return list of (rgi_id, name, shapely geometry in EPSG:4326) intersecting the bbox, or [] if unavailable."""
    from shapely.geometry import box

    import fiona
    from shapely.geometry import shape

    w, s, e, n = lonlat_bbox
    aoi = box(w, s, e, n)
    out = []
    for code, (x0, y0, x1, y1) in RGI_REGIONS.items():
        if not aoi.intersects(box(x0, y0, x1, y1)):
            continue
        shp = _rgi_file(code)
        if shp is None:
            continue
        with fiona.open(shp) as src:
            for feat in src.filter(bbox=(w, s, e, n)):
                g = shape(feat["geometry"])
                if g.intersects(aoi):
                    props = feat["properties"]
                    out.append((props.get("rgi_id"), props.get("glac_name") or "", g))
    return out


def rasterize_geoms(geoms, epsg: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.ops import transform

    if not geoms:
        return np.zeros((len(y), len(x)), bool)
    tr = Transformer.from_crs(4326, epsg, always_xy=True).transform
    shapes = [(transform(tr, g), 1) for g in geoms]
    return rasterize(shapes, out_shape=(len(y), len(x)), transform=grid_transform(x, y), fill=0,
                     dtype=np.uint8).astype(bool)
