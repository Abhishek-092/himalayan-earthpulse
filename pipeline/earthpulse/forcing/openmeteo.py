"""Meteorological and hydrological forcing from Open-Meteo (no API key; CORS-enabled so the web app can fetch live).

* archive: ERA5 / ERA5-Land reanalysis, daily, 1991 -> ~5 days ago
* forecast: best-match NWP, 16 days, plus the last 92 days of analysed weather for antecedent conditions
* flood: GloFAS v4 river discharge (reanalysis 1984 -> present + 30-day forecast)
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import requests

from earthpulse.config import CACHE_DIR

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
FLOOD = "https://flood-api.open-meteo.com/v1/flood"
DAILY = "precipitation_sum,snowfall_sum,temperature_2m_mean,soil_moisture_0_to_7cm_mean,soil_moisture_7_to_28cm_mean"


def _get(url: str, params: dict[str, Any], retries: int = 6) -> dict[str, Any]:
    last = None
    for a in range(retries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 429:  # rate limited: wait for the per-minute window to reset
                time.sleep(65)
                last = RuntimeError("429 Too Many Requests")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(2 ** a)
    raise RuntimeError(f"Open-Meteo request failed: {last}")


def _cache(name: str) -> Path:
    p = CACHE_DIR / "forcing" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _to_arrays(daily: dict[str, list]) -> dict[str, np.ndarray]:
    out = {"time": np.array(daily["time"], dtype="datetime64[D]")}
    for k, v in daily.items():
        if k != "time":
            out[k] = np.array([np.nan if x is None else x for x in v], float)
    return out


def archive(lat: float, lon: float, start: str = "1991-01-01", end: str | None = None, refresh: bool = False,
            daily: str = DAILY):
    end = end or str(date.today() - timedelta(days=6))
    tag = "" if daily == DAILY else "_" + str(abs(hash(daily)) % 10000)
    p = _cache(f"archive_{lat:.3f}_{lon:.3f}_{start}_{end}{tag}.json")
    if p.exists() and not refresh:
        d = json.loads(p.read_text())
    else:
        d = _get(ARCHIVE, {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "daily": daily,
                           "timezone": "Asia/Kolkata"})
        p.write_text(json.dumps(d))
    return _to_arrays(d["daily"]), {"source": "Open-Meteo archive (ERA5/ERA5-Land)", "grid_lat": d["latitude"],
                                    "grid_lon": d["longitude"], "elevation_m": d.get("elevation")}


def forecast(lat: float, lon: float, past_days: int = 92) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    d = _get(FORECAST, {"latitude": lat, "longitude": lon, "daily": "precipitation_sum,snowfall_sum,temperature_2m_mean",
                        "hourly": "soil_moisture_0_to_1cm,soil_moisture_9_to_27cm", "past_days": past_days,
                        "forecast_days": 16, "timezone": "Asia/Kolkata"})
    arr = _to_arrays(d["daily"])
    # hourly soil moisture -> daily mean (deep layer is the slope-wetness proxy)
    hs = np.array([np.nan if x is None else x for x in d["hourly"]["soil_moisture_9_to_27cm"]], float)
    n = len(arr["time"])
    arr["soil_moisture_deep"] = np.nanmean(hs[: n * 24].reshape(n, 24), axis=1) if hs.size >= n * 24 else np.full(n, np.nan)
    issued = d.get("generationtime_ms")
    return arr, {"source": "Open-Meteo forecast (best-match NWP)", "issued_utc": str(np.datetime64("now", "m")),
                 "generation_ms": issued, "today": str(date.today())}


def flood(lat: float, lon: float, start: str = "1984-01-01", end: str | None = None, refresh: bool = False):
    end = end or str(date.today())
    p = _cache(f"glofas_{lat:.3f}_{lon:.3f}_{start}_{end}.json")
    if p.exists() and not refresh:
        hist = json.loads(p.read_text())
    else:
        hist = _get(FLOOD, {"latitude": lat, "longitude": lon, "daily": "river_discharge", "start_date": start,
                            "end_date": end})
        p.write_text(json.dumps(hist))
    fc = _get(FLOOD, {"latitude": lat, "longitude": lon, "daily": "river_discharge,river_discharge_max,river_discharge_min",
                      "forecast_days": 30, "past_days": 30})
    return _to_arrays(hist["daily"]), _to_arrays(fc["daily"]), {
        "source": "GloFAS v4 via Open-Meteo flood API", "grid_lat": hist["latitude"], "grid_lon": hist["longitude"]}


# ---------------------------------------------------------------- derived rainfall indices
def indices(rain: np.ndarray, snow_cm: np.ndarray | None = None, temp: np.ndarray | None = None, k: float = 0.9):
    """Antecedent precipitation index, 3/7/15-day sums and a simple snowmelt proxy (mm water equivalent)."""
    r = np.nan_to_num(rain)
    api = np.zeros_like(r)
    for i in range(len(r)):
        api[i] = r[i] + (k * api[i - 1] if i else 0.0)
    def roll(n):
        c = np.cumsum(np.concatenate([[0.0], r]))
        out = c[n:] - c[:-n]
        return np.concatenate([np.cumsum(r)[: n - 1], out])
    melt = np.zeros_like(r)
    if snow_cm is not None and temp is not None:
        swe = 0.0
        for i in range(len(r)):
            swe += np.nan_to_num(snow_cm[i]) * 10 * 0.1  # cm fresh snow -> mm SWE (10:1 ratio)
            m = min(swe, max(0.0, np.nan_to_num(temp[i])) * 3.0)  # degree-day factor 3 mm/degC/day
            swe -= m
            melt[i] = m
    t = np.nan_to_num(temp) if temp is not None else np.zeros_like(r)
    t7 = np.array([t[max(0, i - 6):i + 1].mean() for i in range(len(t))])  # 7-day mean temperature (melt driver)
    return {"api": api, "r1": r, "r3": roll(3), "r7": roll(7), "r15": roll(15), "melt": melt, "t7": t7}


def ecdf_table(x: np.ndarray, n: int = 101) -> list[float]:
    """Quantile table (0..100th percentile) for percentile conversion in Python and in the browser."""
    v = x[np.isfinite(x)]
    return [float(q) for q in np.percentile(v, np.linspace(0, 100, n))]


def to_pct(x: np.ndarray, table: list[float]) -> np.ndarray:
    t = np.asarray(table)
    return np.interp(x, t, np.linspace(0, 100, len(t)), left=0, right=100)


def best_river_cell(lat: float, lon: float, radius: float = 0.1, step: float = 0.05) -> tuple[float, float, float]:
    """GloFAS snaps to the nearest 0.05 deg cell, which in steep valleys is often a tiny side stream. Search nearby
    cells and return the one with the largest mean monsoon discharge (the main river)."""
    p = _cache(f"glofas_cell_{lat:.3f}_{lon:.3f}.json")
    if p.exists():
        return tuple(json.loads(p.read_text()))  # type: ignore[return-value]
    best = (lat, lon, -1.0)
    offs = np.arange(-radius, radius + 1e-9, step)
    for dy in offs:
        for dx in offs:
            try:
                d = _get(FLOOD, {"latitude": lat + dy, "longitude": lon + dx, "daily": "river_discharge",
                                 "start_date": "2020-06-01", "end_date": "2020-09-30"}, retries=2)
            except Exception:
                continue
            q = np.array([x for x in d["daily"]["river_discharge"] if x is not None], float)
            if q.size and q.mean() > best[2]:
                best = (d["latitude"], d["longitude"], float(q.mean()))
    p.write_text(json.dumps(best))
    return best


def snow_depth_daily(lat: float, lon: float, start: str, end: str) -> dict[str, np.ndarray]:
    """ERA5-Land snow depth (m), daily mean from hourly, for per-date snow masking."""
    p = _cache(f"snowdepth_{lat:.3f}_{lon:.3f}_{start}_{end}.json")
    if p.exists():
        d = json.loads(p.read_text())
    else:
        d = _get(ARCHIVE, {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                           "hourly": "snow_depth", "timezone": "Asia/Kolkata"})
        p.write_text(json.dumps(d))
    t = np.array(d["hourly"]["time"], dtype="datetime64[h]").astype("datetime64[D]")
    v = np.array([np.nan if x is None else x for x in d["hourly"]["snow_depth"]], float)
    days = np.unique(t)
    return {"time": days, "snow_depth_m": np.array([np.nanmean(v[t == dd]) for dd in days]),
            "grid_elevation_m": d.get("elevation")}


def hourly_temperature(lat: float, lon: float, start: str, end: str) -> dict[str, Any]:
    p = _cache(f"t2m_hourly_{lat:.3f}_{lon:.3f}_{start}_{end}.json")
    if p.exists():
        d = json.loads(p.read_text())
    else:
        d = _get(ARCHIVE, {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                           "hourly": "temperature_2m", "timezone": "GMT"})
        p.write_text(json.dumps(d))
    return {"time": np.array(d["hourly"]["time"], dtype="datetime64[h]"),
            "t2m": np.array([np.nan if x is None else x for x in d["hourly"]["temperature_2m"]], float),
            "grid_elevation_m": float(d.get("elevation") or 0.0)}
