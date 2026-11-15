"""Advisory levels (not official warnings). Bands are provisional: the trigger model is calibrated on 1991-2025 data, but
the slope-failure probability relies on assumed soil parameters, so levels are conservative and every reason is shown."""
from __future__ import annotations

LEVELS = [
    {"level": 0, "name": "GREEN", "label": "Normal", "color": "#16a34a",
     "action": "Conditions within the expected range. Continue routine monitoring."},
    {"level": 1, "name": "YELLOW", "label": "Watch", "color": "#eab308",
     "action": "Conditions are changing. Increase monitoring; notify the monitoring team."},
    {"level": 2, "name": "ORANGE", "label": "Warning", "color": "#f97316",
     "action": "Several indicators elevated. Verify on the ground; preparedness by responsible authorities."},
    {"level": 3, "name": "RED", "label": "Critical", "color": "#dc2626",
     "action": "Severe conditions. Escalate to authorities under an approved protocol (only they issue warnings)."},
]


def slope_level(p72: float) -> int:
    """River high-flow probability bands (kept name for backwards compatibility)."""
    return 3 if p72 >= 0.6 else 2 if p72 >= 0.35 else 1 if p72 >= 0.15 else 0


def lhasa_level(r7: float, p90: float, p95: float, p99: float) -> int:
    """NASA LHASA-style rainfall trigger: 7-day rain vs the site's own climatological percentiles."""
    return 3 if r7 >= p99 else 2 if r7 >= p95 else 1 if r7 >= p90 else 0


def combine(slope_lvl: int, river_lvl: int, flood_level: int, slope_reason: str, river_reason: str, flood_reason: str) -> dict:
    lvl = max(slope_lvl, river_lvl, flood_level)
    reasons = []
    if slope_lvl:
        reasons.append(slope_reason)
    if river_lvl:
        reasons.append(river_reason)
    if flood_level:
        reasons.append(flood_reason)
    if not reasons:
        reasons = ["Rainfall, soil moisture and river discharge are all within their normal ranges."]
    return {**LEVELS[lvl], "slope_level": slope_lvl, "river_level": river_lvl, "flood_level": flood_level, "reasons": reasons,
            "disclaimer": "Model-generated advisory for demonstration. Not an official warning; only authorised "
                          "agencies (e.g. IMD, NDMA, USDMA) issue warnings."}
