"""Discover NISAR L2 granules for any region via the ASF Search API (no authentication needed).

Produces provenance-rich records (granule id, product, track/frame, direction, dates, size, AOI coverage)
that every downstream step references, so each map in the app can name the exact NISAR products it used.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable

import requests
from shapely.geometry import shape

from earthpulse.config import PRODUCTS, Region

ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"
_TS = re.compile(r"(\d{8}T\d{6})")


@dataclass
class Granule:
    granule_id: str
    product: str
    url: str
    track: int
    frame: int
    direction: str  # "A" or "D"
    ref_time: str  # ISO UTC
    sec_time: str | None  # ISO UTC for pair products (GUNW/GOFF)
    temporal_baseline_days: float | None
    size_mb: float
    coverage: float  # fraction of AOI inside footprint
    collection: str
    polarization: str
    mode: str
    crid: str
    browse: str | None
    orbit_type: str = ""
    pge_version: str = ""

    @property
    def stack_key(self) -> str:
        return f"{self.direction}{self.track:03d}_F{self.frame:03d}"

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["stack_key"] = self.stack_key
        return d


def _iso(ts: str) -> str:
    return datetime.strptime(ts, "%Y%m%dT%H%M%S").strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_granule_name(name: str) -> dict[str, Any]:
    """Parse the NISAR file naming convention (L2 products).

    GCOV: NISAR_L2_PR_GCOV_<cyc>_<trk>_<dir>_<frm>_<mode>_<pol>_<x>_<start>_<end>_<crid>_...
    GUNW/GOFF: NISAR_L2_PR_<prod>_<refcyc>_<trk>_<dir>_<frm>_<seccyc>_<mode>_<pol>_<rs>_<re>_<ss>_<se>_<crid>_...
    """
    stem = name[:-3] if name.endswith(".h5") else name
    parts = stem.split("_")
    product = parts[3]
    times = _TS.findall(stem)
    out: dict[str, Any] = {
        "product": product,
        "track": int(parts[5]),
        "direction": parts[6],
        "frame": int(parts[7]),
        "ref_time": _iso(times[0]) if times else None,
        "sec_time": None,
        "crid": next((p for p in parts if re.fullmatch(r"P\d{5}", p)), ""),
    }
    if product in ("GUNW", "GOFF", "RUNW", "RIFG", "ROFF") and len(times) >= 4:
        out["sec_time"] = _iso(times[2])
        out["mode"], out["polarization"] = parts[9], parts[10]
    else:
        out["mode"], out["polarization"] = parts[8], parts[9]
    return out


def _get(params: dict[str, Any], retries: int = 4) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(ASF_SEARCH_URL, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network hiccups: back off and retry
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ASF search failed after {retries} attempts: {last}")


def search_region(region: Region, products: Iterable[str] = PRODUCTS, start: str | None = None,
                  end: str | None = None, min_coverage: float = 0.5) -> list[Granule]:
    """Return all NISAR granules of the given products whose footprint covers >= min_coverage of the AOI."""
    start = start or region.overrides.get("start")
    end = end or region.overrides.get("end")
    aoi = region.geometry
    granules: list[Granule] = []
    for product in products:
        params: dict[str, Any] = {
            "platform": "NISAR",
            "processingLevel": product,
            "intersectsWith": aoi.wkt,
            "maxResults": 5000,
            "output": "geojson",
        }
        if start:
            params["start"] = f"{start}T00:00:00Z"
        if end:
            params["end"] = f"{end}T23:59:59Z"
        for feat in _get(params).get("features", []):
            p = feat["properties"]
            name = p["fileName"]
            try:
                info = parse_granule_name(name)
            except (IndexError, ValueError):
                continue
            try:
                cov = aoi.intersection(shape(feat["geometry"])).area / aoi.area
            except Exception:
                cov = 0.0
            if cov < min_coverage:
                continue
            size = p.get("bytes", {})
            size_b = size.get(name, {}).get("bytes", 0) if isinstance(size, dict) else (size or 0)
            tb = None
            if info["sec_time"]:
                t0 = datetime.fromisoformat(info["ref_time"].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(info["sec_time"].replace("Z", "+00:00"))
                tb = round((t1 - t0).total_seconds() / 86400.0, 3)
            browse = p.get("browse")
            granules.append(Granule(
                granule_id=p.get("sceneName") or name[:-3],
                product=product,
                url=p["url"],
                track=int(p.get("pathNumber") or info["track"]),
                frame=int(p.get("frameNumber") or info["frame"]),
                direction=(p.get("flightDirection") or info["direction"])[0],
                ref_time=info["ref_time"],
                sec_time=info["sec_time"],
                temporal_baseline_days=tb,
                size_mb=round(size_b / 1e6, 1),
                coverage=round(cov, 3),
                collection=p.get("collectionName", ""),
                polarization=info.get("polarization", ""),
                mode=info.get("mode", ""),
                crid=info["crid"],
                browse=browse[0] if isinstance(browse, list) and browse else None,
                orbit_type=str(p.get("orbitType") or ""),
                pge_version=str(p.get("pgeVersion") or ""),
            ))
    # De-duplicate reprocessed versions: keep the latest CRID per (product, track, frame, ref, sec)
    best: dict[tuple, Granule] = {}
    for g in granules:
        k = (g.product, g.track, g.frame, g.ref_time, g.sec_time)
        if k not in best or g.crid > best[k].crid:
            best[k] = g
    return sorted(best.values(), key=lambda g: (g.product, g.stack_key, g.ref_time, g.sec_time or ""))


def summarize(granules: list[Granule]) -> dict[str, Any]:
    """Counts and date ranges per product and per stack (track/frame/direction)."""
    out: dict[str, Any] = {}
    by_prod: dict[str, list[Granule]] = defaultdict(list)
    for g in granules:
        by_prod[g.product].append(g)
    for prod, gs in by_prod.items():
        stacks: dict[str, list[Granule]] = defaultdict(list)
        for g in gs:
            stacks[g.stack_key].append(g)
        out[prod] = {
            "count": len(gs),
            "first": min(g.ref_time for g in gs),
            "last": max((g.sec_time or g.ref_time) for g in gs),
            "stacks": {
                k: {
                    "count": len(v),
                    "first": min(g.ref_time for g in v),
                    "last": max((g.sec_time or g.ref_time) for g in v),
                    "mean_coverage": round(sum(g.coverage for g in v) / len(v), 3),
                    "direction": v[0].direction,
                    "track": v[0].track,
                    "frame": v[0].frame,
                }
                for k, v in sorted(stacks.items(), key=lambda kv: -len(kv[1]))
            },
        }
    return out
