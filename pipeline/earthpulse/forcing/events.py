"""Documented hazard events in Garhwal (Uttarakhand) used as labels and validation targets.

The NASA Global Landslide Catalog could not be downloaded at build time (data.nasa.gov / NCCS endpoints offline),
so this is a small list of events whose sources were verified at build time. With so few verified rain-triggered events,
the statistical trigger model is trained on an objective hydrological label instead (GloFAS discharge above the
2-year flood level) and these events are used as hindcast checks. `rain_triggered` separates events a rainfall-driven model
*should* catch from events it *should not* (rock-ice avalanche, winter subsidence) - both are reported honestly.
The NASA Global Landslide Catalog export (data/external/glc.csv) is used by hazard/landslide.py to train and test the
landslide trigger model; this short list remains for the event hindcast table.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Event:
    date: str
    kind: str  # landslide | flood | debris flow | avalanche | subsidence
    place: str
    distance_km: float  # approx. distance from Joshimath
    rain_triggered: bool | None  # None = cause debated
    note: str
    source: str


EVENTS: list[Event] = [
    Event("2013-06-16", "flood + landslides", "Kedarnath / Alaknanda & Mandakini valleys", 45, True,
          "Extreme June 2013 rainfall; floods and landslides across Uttarakhand. Main flood day 16 June 2013.",
          "https://en.wikipedia.org/wiki/2013_North_India_floods"),
    Event("2025-08-05", "flash flood / debris flow", "Uttarkashi (Bhagirathi valley)", 110, None,
          "Flash flood on 5 August 2025; trigger debated (heavy rain vs glacial source) and outside our river basin.",
          "https://en.wikipedia.org/wiki/2025_Uttarakhand_flash_flood"),
    Event("2021-02-07", "rock-ice avalanche flood", "Rishiganga / Dhauliganga (Chamoli)", 25, False,
          "Caused by a large rock and ice avalanche in winter - not rainfall-driven; a rain model should miss it.",
          "https://en.wikipedia.org/wiki/2021_Uttarakhand_flood"),
    Event("2023-01-02", "land subsidence", "Joshimath town", 0, False,
          "Ground-cracking crisis in mid-winter (January 2023) - not a rain trigger.",
          "NASA Space Apps 2026 challenge brief (Dancing with the SARs)"),
]


def training_events() -> list[Event]:
    return [e for e in EVENTS if e.rain_triggered is True]


def load_glc_csv(path: Path, bbox=(78.0, 29.5, 81.5, 31.5)) -> list[Event]:
    """NASA Global Landslide Catalog export (event_date, latitude, longitude, landslide_trigger, ...)."""
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
            except (KeyError, ValueError):
                continue
            if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                continue
            trig = (row.get("landslide_trigger") or "").lower()
            out.append(Event(row.get("event_date", "")[:10], "landslide", row.get("event_title", ""), float("nan"),
                             any(k in trig for k in ("rain", "monsoon", "downpour", "cloud")), trig, "NASA GLC"))
    return out
