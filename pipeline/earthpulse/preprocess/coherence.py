"""Interferometric coherence time series from cached NISAR GUNW subsets, resampled onto another model's grid.

Coherence (0-1) measures how similar the radar echoes stayed between two passes. Forest disturbance, snow, moisture
and fast-moving ice all lower it, so it is an independent indicator next to backscatter (BDM) and offsets (CDM).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from earthpulse.catalog.discover import parse_granule_name
from earthpulse.catalog.stacks import region_granules, select_stacks
from earthpulse.io.nisar import Subset, cache_path


def _nearest_index(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    order = np.argsort(src)
    pos = np.clip(np.searchsorted(src[order], dst), 1, len(src) - 1)
    left, right = src[order][pos - 1], src[order][pos]
    pick = np.where(np.abs(dst - left) <= np.abs(dst - right), pos - 1, pos)
    idx = order[pick]
    step = np.median(np.abs(np.diff(np.sort(src))))
    out_of_range = (dst < src.min() - step) | (dst > src.max() + step)
    return np.where(out_of_range, -1, idx)


def coherence_stack(region, x: np.ndarray, y: np.ndarray, epsg: int) -> dict[str, Any] | None:
    """All cached GUNW pairs of the best-covered stack, coherence resampled (nearest) to the (x, y) grid."""
    stacks = select_stacks(region_granules(region), "GUNW")
    best = None
    for key, members in stacks.items():
        cached = [g for g in members if cache_path(region.id, "GUNW", g.granule_id).exists()]
        if cached and (best is None or len(cached) > len(best[1])):
            best = (key, cached)
    if best is None:
        return None
    key, members = best
    ix = iy = None
    cohs, mids, pairs, ids = [], [], [], []
    for g in members:
        s = Subset.load(cache_path(region.id, "GUNW", g.granule_id))
        if s.epsg != epsg or "coherenceMagnitude" not in s.layers:
            continue
        if ix is None:
            ix, iy = _nearest_index(s.x, x), _nearest_index(s.y, y)
            sx, sy = s.shape[1], s.shape[0]
        if s.shape != (sy, sx):
            continue
        c = s.layers["coherenceMagnitude"].astype(np.float32)
        grid = np.full((len(y), len(x)), np.nan, np.float32)
        okx, oky = ix >= 0, iy >= 0
        grid[np.ix_(oky, okx)] = c[np.ix_(iy[oky], ix[okx])]
        info = parse_granule_name(g.granule_id)
        t0, t1 = np.datetime64(info["ref_time"][:10]), np.datetime64(info["sec_time"][:10])
        cohs.append(grid)
        mids.append(t0 + (t1 - t0) // 2)
        pairs.append([str(t0), str(t1)])
        ids.append(g.granule_id)
    if len(cohs) < 3:
        return None
    o = np.argsort(np.array(mids))
    return {"stack": key, "coherence": np.stack(cohs)[o], "mid_dates": np.array(mids)[o], "pairs": [pairs[i] for i in o],
            "granules": [ids[i] for i in o], "temporal_baseline_days": [int((np.datetime64(pairs[i][1]) - np.datetime64(pairs[i][0])).astype(int)) for i in o]}


def masked_series(C: np.ndarray, mask: np.ndarray) -> list[float | None]:
    return [float(np.nanmedian(c[mask])) if np.isfinite(c[mask]).any() else None for c in C]
