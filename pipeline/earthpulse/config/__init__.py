"""Configuration: paths, physical constants and region loading.

Regions are defined only by id/name/footprint in ``sites.yaml``. Everything else is auto-detected
(see :mod:`earthpulse.sites.autodetect`). No region-specific logic lives in code.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PACKAGE_DIR.parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a private .env (gitignored) without overriding existing env vars."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        os.environ.setdefault(k.strip(), v)


_load_dotenv(PIPELINE_DIR / ".env")
REPO_DIR = PIPELINE_DIR.parent
SITES_FILE = Path(os.environ.get("EARTHPULSE_SITES", PACKAGE_DIR / "config" / "sites.yaml"))
DATA_DIR = Path(os.environ.get("EARTHPULSE_DATA", PIPELINE_DIR / "data"))
CACHE_DIR = DATA_DIR / "cache"
WORK_DIR = DATA_DIR / "work"
WEB_DATA_DIR = Path(os.environ.get("EARTHPULSE_WEB_DATA", REPO_DIR / "web" / "public" / "data"))
RESOLVED_FILE = DATA_DIR / "sites.resolved.json"
CATALOG_FILE = DATA_DIR / "catalog.json"

# --- NISAR physical constants -------------------------------------------------------------------
SPEED_OF_LIGHT = 299_792_458.0
NISAR_L_CENTER_FREQ_HZ = 1.2575e9  # L-band (NASA radar), 1257.5 MHz centre frequency
NISAR_L_WAVELENGTH_M = SPEED_OF_LIGHT / NISAR_L_CENTER_FREQ_HZ  # ~0.2384 m
NISAR_S_WAVELENGTH_M = SPEED_OF_LIGHT / 3.2e9  # S-band (ISRO radar) ~0.094 m
NISAR_REPEAT_DAYS = 12

# NISAR L2 product families used by the three models (+ SME2 discovered for Phase 2).
PRODUCTS = ("GCOV", "GUNW", "GOFF", "SME2")


@dataclass
class Region:
    id: str
    name: str
    geometry: BaseGeometry
    story: str = ""
    priority: str = "catalog"
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.geometry.bounds)  # type: ignore[return-value]

    @property
    def wkt(self) -> str:
        return self.geometry.wkt

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "story": self.story,
            "priority": self.priority,
            "bbox": list(self.bbox),
            "geometry": mapping(self.geometry),
            "overrides": self.overrides,
        }


def _region_from_dict(d: dict[str, Any]) -> Region:
    if "bbox" in d:
        w, s, e, n = d["bbox"]
        if not (w < e and s < n):
            raise ValueError(f"region {d.get('id')}: bbox must be [W,S,E,N] with W<E and S<N")
        geom = box(w, s, e, n)
    elif "geometry" in d:
        geom = shape(d["geometry"])
    else:
        raise ValueError(f"region {d.get('id')}: needs 'bbox' or 'geometry'")
    if not geom.is_valid or geom.is_empty:
        raise ValueError(f"region {d.get('id')}: invalid footprint")
    return Region(
        id=str(d["id"]),
        name=str(d.get("name", d["id"])),
        geometry=geom,
        story=str(d.get("story", "")),
        priority=str(d.get("priority", "catalog")),
        overrides=dict(d.get("overrides") or {}),
    )


def load_regions(path: Path = SITES_FILE) -> list[Region]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    regions = [_region_from_dict(r) for r in cfg.get("regions", [])]
    ids = [r.id for r in regions]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate region ids: {sorted(dup)}")
    return regions


def get_region(region_id: str, path: Path = SITES_FILE) -> Region:
    for r in load_regions(path):
        if r.id == region_id:
            return r
    raise KeyError(f"unknown region '{region_id}' (see {path})")


def add_region(region_id: str, name: str, bbox: tuple[float, float, float, float], story: str = "",
               priority: str = "catalog", path: Path = SITES_FILE) -> Region:
    """Append a region to sites.yaml (validates first). Keeps the file human-editable."""
    region = _region_from_dict({"id": region_id, "name": name, "bbox": list(bbox), "story": story,
                                "priority": priority})
    existing = {r.id for r in load_regions(path)}
    if region_id in existing:
        raise ValueError(f"region '{region_id}' already exists")
    lines = [
        "",
        f"  - id: {region_id}",
        f"    name: {json.dumps(name)}",
        f"    bbox: [{', '.join(f'{v:.5f}' for v in bbox)}]",
    ]
    if story:
        lines.append(f"    story: {json.dumps(story)}")
    lines.append(f"    priority: {priority}")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
    load_regions(path)  # re-validate the whole file
    return region


def load_resolved() -> dict[str, Any]:
    if RESOLVED_FILE.exists():
        return json.loads(RESOLVED_FILE.read_text())
    return {"regions": {}}


def save_resolved(data: dict[str, Any]) -> None:
    RESOLVED_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESOLVED_FILE.write_text(json.dumps(data, indent=2, default=str))


def ensure_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, WORK_DIR, WEB_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
