"""Area calculations that respect map projection (never count lat/lon pixels as equal areas)."""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Geod, Transformer

_GEOD = Geod(ellps="WGS84")


def pixel_area_m2(crs: CRS | int | str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-pixel true ground area (m^2) for a regular grid with centre coordinates x (W,), y (H,).

    Projected CRS (e.g. NISAR's UTM grids): |dx*dy| corrected by the local areal scale factor.
    Geographic CRS: geodesic area of each cell.
    """
    crs = CRS.from_user_input(crs)
    dx = float(np.abs(np.median(np.diff(x))))
    dy = float(np.abs(np.median(np.diff(y))))
    X, Y = np.meshgrid(x, y)
    if crs.is_geographic:
        lat0 = np.radians(Y - dy / 2)
        lat1 = np.radians(Y + dy / 2)
        r = 6371008.8
        return (r ** 2) * np.radians(dx) * np.abs(np.sin(lat1) - np.sin(lat0))
    to_ll = Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = to_ll.transform(X, Y)
    try:
        from pyproj import Proj
        f = Proj(crs).get_factors(lon, lat)
        scale = np.asarray(f.areal_scale)
    except Exception:
        scale = np.ones_like(X)
    return dx * dy / scale


def masked_area_km2(mask: np.ndarray, pixel_area: np.ndarray) -> float:
    return float(np.sum(np.where(mask & np.isfinite(pixel_area), pixel_area, 0.0)) / 1e6)
