"""Sentinel-2 L2A (Microsoft Planetary Computer STAC, anonymous) - independent optical evidence.

* dNBR (pre - post Normalised Burn Ratio) -> burned / canopy-loss reference for BDM
* true-colour composites for the app's "what a camera saw" context layer
Cloud masking uses the SCL band (classes 3, 8, 9, 10 = shadow, cloud medium/high, cirrus).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from rasterio.enums import Resampling

from earthpulse.geo.ancillary import _cached, _lonlat_bounds, _warp

log = logging.getLogger(__name__)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _search(lonlat_bbox, start: str, end: str, max_cloud: float = 40, limit: int = 30):
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    items = cat.search(collections=["sentinel-2-l2a"], bbox=lonlat_bbox, datetime=f"{start}/{end}",
                       query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=limit).item_collection()
    return sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))


def composite(epsg: int, x: np.ndarray, y: np.ndarray, start: str, end: str, bands=("B04", "B03", "B02", "B08", "B12"),
              n_scenes: int = 4) -> dict[str, Any] | None:
    """Cloud-masked median composite of the clearest scenes in a window, on the target grid."""
    bbox = _lonlat_bounds(epsg, x, y)
    key = f"s2_{start}_{end}_{'-'.join(bands)}"

    def run():
        items = _search(bbox, start, end)[:n_scenes]
        if not items:
            return np.full((len(bands) + 1, len(y), len(x)), np.nan, np.float32)
        stacks = []
        for it in items:
            scl = _warp(["/vsicurl/" + it.assets["SCL"].href], epsg, x, y, Resampling.nearest)
            bad = np.isin(scl, [0, 1, 3, 8, 9, 10]) | ~np.isfinite(scl)
            arrs = []
            for b in bands:
                a = _warp(["/vsicurl/" + it.assets[b].href], epsg, x, y, Resampling.bilinear)
                arrs.append(np.where(bad, np.nan, a / 10000.0))
            stacks.append(np.stack(arrs))
        med = np.nanmedian(np.stack(stacks), axis=0).astype(np.float32)
        count = np.sum(np.isfinite(np.stack(stacks)[:, 0]), axis=0).astype(np.float32)
        return np.concatenate([med, count[None]])
    arr = _cached(key, epsg, x, y, run)
    if not np.isfinite(arr[0]).any():
        return None
    return {b: arr[i] for i, b in enumerate(bands)} | {"count": arr[-1], "window": [start, end]}


def nbr(c: dict[str, Any]) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c["B08"] - c["B12"]) / (c["B08"] + c["B12"])


def dnbr(epsg: int, x, y, pre: tuple[str, str], post: tuple[str, str]) -> dict[str, Any] | None:
    a = composite(epsg, x, y, *pre)
    b = composite(epsg, x, y, *post)
    if a is None or b is None:
        return None
    return {"dnbr": nbr(a) - nbr(b), "pre": a, "post": b}
