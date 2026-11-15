"""NASA ITS_LIVE glacier velocity (independent optical + Sentinel-1 feature tracking), used to cross-check CDM.

Static v2 mosaics (2014-2022 composite, 120 m) are public COGs on AWS - read only the AOI window.
Comparison is a *climatology vs. 2025-26 NISAR* check: agreement in pattern and magnitude validates the method;
differences may also reflect real interannual change, which is stated in the app.
"""
from __future__ import annotations

import numpy as np
from rasterio.enums import Resampling

from earthpulse.geo.ancillary import _cached, _warp

BASE = "/vsicurl/https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/static/cog"


def _rgi_code(lon: float, lat: float) -> str:
    if lon < 79 or (lat > 32 and lon < 81):
        return "RGI14A"
    if lon < 100 and lat < 32.5:
        return "RGI15A"
    return "RGI13A"


def velocity(epsg: int, x: np.ndarray, y: np.ndarray, lonlat: tuple[float, float]) -> dict[str, np.ndarray]:
    first = _rgi_code(*lonlat)
    out = {}
    for code in [first] + [c for c in ("RGI14A", "RGI15A", "RGI13A") if c != first]:
        out = {}
        for comp in ("v", "vx", "vy", "v_error"):
            url = f"{BASE}/ITS_LIVE_velocity_120m_{code}_0000_v02_{comp}.tif"
            out[comp] = _cached(f"itslive_{code}_{comp}", epsg, x, y,
                                lambda u=url: _warp([u], epsg, x, y, Resampling.bilinear))
        if np.isfinite(out["v"]).mean() > 0.05:
            break
    out["source"] = f"ITS_LIVE v2 static mosaic {code} (2014-2022), 120 m"  # type: ignore[assignment]
    return out


def landice(epsg: int, x: np.ndarray, y: np.ndarray, lonlat: tuple[float, float]) -> np.ndarray | None:
    for code in dict.fromkeys([_rgi_code(*lonlat), "RGI14A", "RGI15A"]):
        url = f"{BASE}/ITS_LIVE_velocity_120m_{code}_0000_v02_landice.tif"
        a = _cached(f"itslive_{code}_landice", epsg, x, y, lambda u=url: _warp([u], epsg, x, y, Resampling.nearest))
        m = np.nan_to_num(a) > 0.5
        if m.sum() > 50:
            return m
    return None
