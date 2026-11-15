"""Flood forecasting layer: GloFAS river discharge return levels (Gumbel on annual maxima) and alert thresholds."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import gumbel_r


def annual_maxima(time: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    years = time.astype("datetime64[Y]").astype(int) + 1970
    ys, mx, dates = [], [], []
    for y in np.unique(years):
        m = (years == y) & np.isfinite(q)
        if m.sum() < 300:  # incomplete year
            continue
        i = np.argmax(np.where(m, q, -np.inf))
        ys.append(y)
        mx.append(q[i])
        dates.append(str(time[i]))
    return np.array(ys), np.array(mx), dates


def return_levels(maxima: np.ndarray, periods=(2, 5, 20, 100)) -> dict[str, float]:
    loc, scale = gumbel_r.fit(maxima)
    return {f"Q{T}": float(gumbel_r.ppf(1 - 1 / T, loc, scale)) for T in periods} | {"loc": float(loc), "scale": float(scale)}


def level_for(q: float, rl: dict[str, float]) -> int:
    """0 normal, 1 >= 2-yr, 2 >= 5-yr, 3 >= 20-yr."""
    if not np.isfinite(q):
        return 0
    return 3 if q >= rl["Q20"] else 2 if q >= rl["Q5"] else 1 if q >= rl["Q2"] else 0


def event_check(time: np.ndarray, q: np.ndarray, rl: dict[str, float], events) -> list[dict[str, Any]]:
    out = []
    for e in events:
        d = np.datetime64(e.date)
        m = (time >= d - np.timedelta64(3, "D")) & (time <= d + np.timedelta64(3, "D"))
        qmax = float(np.nanmax(q[m])) if m.any() else float("nan")
        lvl = level_for(qmax, rl)
        expected = bool(e.rain_triggered) if e.rain_triggered is not None else None
        out.append({"date": e.date, "event": e.kind, "place": e.place, "q_max_m3s": round(qmax, 1) if np.isfinite(qmax) else None,
                    "level": lvl, "detected": lvl >= 1, "expected_detect": expected,
                    "outcome": ("hit" if lvl >= 1 else "miss") if expected else
                               ("correctly quiet" if lvl == 0 else "flagged") if expected is False else "n/a (debated)",
                    "note": e.note, "source": e.source})
    return out


def detect_break(time: np.ndarray, q: np.ndarray, min_years: int = 5) -> dict[str, Any]:
    """Find the year with the largest step change in mean monsoon (JJAS) flow - flags product splices in the record."""
    yr = time.astype("datetime64[Y]").astype(int) + 1970
    mon = (time.astype("datetime64[M]").astype(int) % 12) + 1
    jjas = np.isin(mon, [6, 7, 8, 9]) & np.isfinite(q)
    years = np.unique(yr[jjas])
    best = None
    for b in years[min_years:-min_years + 1] if len(years) > 2 * min_years else []:
        before = q[jjas & (yr < b)]
        after = q[jjas & (yr >= b)]
        ratio = float(np.mean(after) / max(np.mean(before), 1e-9))
        if best is None or abs(np.log(ratio)) > abs(np.log(best[1])):
            best = (int(b), ratio)
    if best is None:
        return {"break_year": None}
    return {"break_year": best[0], "jjas_mean_ratio_after_before": round(best[1], 2),
            "significant": abs(np.log(best[1])) > np.log(1.3)}
