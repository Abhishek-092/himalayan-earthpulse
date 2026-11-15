"""Automatic stack (track/frame) selection - no hard-coded tracks.

For each product, choose the best-covered ascending and descending stacks: highest mean AOI coverage first
(>= min_coverage), then most acquisitions. Using one stack per geometry keeps every pixel on one identical
map grid within a time series, and two geometries enable vertical/east-west decomposition (GDM).
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from earthpulse.catalog.discover import Granule, search_region
from earthpulse.config import CATALOG_FILE, Region


def select_stacks(granules: list[Granule], product: str, min_coverage: float = 0.9,
                  per_direction: int = 1) -> dict[str, list[Granule]]:
    stacks: dict[str, list[Granule]] = defaultdict(list)
    for g in granules:
        if g.product == product:
            stacks[g.stack_key].append(g)
    ranked = []
    for k, gs in stacks.items():
        cov = sum(g.coverage for g in gs) / len(gs)
        if cov >= min_coverage:
            ranked.append((gs[0].direction, -round(cov, 2), -len(gs), k))
    out: dict[str, list[Granule]] = {}
    for direction in ("A", "D"):
        for _, _, _, k in sorted(r for r in ranked if r[0] == direction)[:per_direction]:
            out[k] = sorted(stacks[k], key=lambda g: (g.ref_time, g.sec_time or ""))
    return out


def load_catalog() -> dict[str, Any]:
    if CATALOG_FILE.exists():
        return json.loads(CATALOG_FILE.read_text())
    return {"regions": {}}


def region_granules(region: Region, refresh: bool = False) -> list[Granule]:
    """Cached discovery per region (catalog.json)."""
    cat = load_catalog()
    entry = cat["regions"].get(region.id)
    if entry and not refresh:
        return [Granule(**{k: v for k, v in g.items() if k != "stack_key"}) for g in entry["granules"]]
    gs = search_region(region)
    cat["regions"][region.id] = {"granules": [g.to_json() for g in gs]}
    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_FILE.write_text(json.dumps(cat, indent=1))
    return gs
