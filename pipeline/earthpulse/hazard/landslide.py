"""Landslide trigger model trained on the NASA Global Landslide Catalog (GLC, 2007-2017) for Uttarakhand.

Design (follows NASA LHASA):
* each rain-triggered GLC landslide is assigned to a 0.5 deg analysis cell (rain sampled at the cell centre); per cell we fetch daily rain, soil moisture and
  temperature (Open-Meteo archive) and express every index as a percentile of that cell's own 1998-2017 climatology
* sample = (cell, day) over the GLC period; label 1 if a reported landslide occurred in the cell that day or the next
* models: (a) LHASA rule - 7-day rain above the cell's 95th / 99th percentile; (b) calibrated logistic regression
* validation: time split (train 2007-2012, test 2013-2017) + leave-one-year-out; metrics: ROC-AUC, Brier skill vs climatology,
  probability of detection, false-alarm ratio, critical success index
Caveat stated in the app: GLC is a media-based inventory - unreported landslides count as "no landslide", so absolute
probabilities are lower bounds and remote areas are under-represented.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from earthpulse.forcing import openmeteo as om
from earthpulse.hazard import slope

log = logging.getLogger(__name__)
RAIN_TRIGGERS = {"downpour", "rain", "continuous_rain", "monsoon"}
FEATURES = ("r1", "r3", "r7", "api", "sm", "t7")
BAD_ACCURACY = {"50km", "100km", "250km", "unknown"}


def load_events(path: Path, bbox=(77.5, 29.0, 81.1, 31.6)) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                lat, lon = float(r["latitude"]), float(r["longitude"])
                d = datetime.strptime(r["event_date"][:10], "%m/%d/%Y").date()
            except (ValueError, KeyError):
                continue
            if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                continue
            out.append({"date": str(d), "lat": lat, "lon": lon, "trigger": r.get("landslide_trigger", ""),
                        "size": r.get("landslide_size", ""), "accuracy": r.get("location_accuracy", ""),
                        "title": r.get("event_title", ""), "fatalities": r.get("fatality_count", ""),
                        "source": r.get("source_link", "")})
    return out


CELL_DEG = 0.5  # analysis cell; ERA5 is 0.25 deg - 0.5 deg halves API calls (rate limits) at modest smoothing cost


def cell_of(lat: float, lon: float) -> tuple[float, float]:
    return round(lat / CELL_DEG) * CELL_DEG, round(lon / CELL_DEG) * CELL_DEG


def build_dataset(events: list[dict[str, Any]]):
    rain_ev = [e for e in events if e["trigger"] in RAIN_TRIGGERS and e["accuracy"] not in BAD_ACCURACY]
    by_cell: dict[tuple[float, float], set[str]] = defaultdict(set)
    for e in rain_ev:
        by_cell[cell_of(e["lat"], e["lon"])].add(e["date"])
    start = min(e["date"] for e in events)
    end = max(e["date"] for e in events)
    Xs, ys, years, cells, tables_by_cell = [], [], [], [], {}
    for cell, dates in sorted(by_cell.items()):
        arc, _ = om.archive(cell[0], cell[1], start="1998-01-01", end="2017-12-31",
                            daily="precipitation_sum,temperature_2m_mean,soil_moisture_7_to_28cm_mean")
        ix = om.indices(arc["precipitation_sum"], None, arc["temperature_2m_mean"])
        sm = arc["soil_moisture_7_to_28cm_mean"]
        feats = {"sm": sm, **{k: ix[k] for k in ("r1", "r3", "r7", "api", "t7")}}
        tables = {k: om.ecdf_table(feats[k]) for k in FEATURES}
        tables_by_cell[cell] = tables
        t = arc["time"]
        sel = (t >= np.datetime64(start)) & (t <= np.datetime64(end)) & np.isfinite(sm)
        ev = np.zeros(len(t))
        tidx = {str(d): i for i, d in enumerate(t)}
        for d in dates:
            i = tidx.get(d)
            if i is not None:
                ev[i] = 1
                if i > 0:
                    ev[i - 1] = 1  # reported date may lag the trigger by a day
        X = np.column_stack([om.to_pct(feats[k], tables[k]) for k in FEATURES])
        Xs.append(X[sel])
        ys.append(ev[sel])
        years.append(t[sel].astype("datetime64[Y]").astype(int) + 1970)
        cells += [cell] * int(sel.sum())
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(years), cells, len(rain_ev), len(by_cell)


def _contingency(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return {"pod": tp / max(tp + fn, 1), "far": fp / max(tp + fp, 1), "csi": tp / max(tp + fp + fn, 1),
            "alarm_days_fraction": float(pred.mean())}


def train_evaluate(events: list[dict[str, Any]], train_end: int = 2010, cal_end: int = 2012) -> dict[str, Any]:
    """Three-way time split: train 2007-2010, calibrate 2011-2012, test 2013-2017 (unseen)."""
    X, y, yr, cells, n_ev, n_cells = build_dataset(events)
    cm = slope.calibrated_model(X, y, yr, train_end, cal_end, l2=5.0, gbm=True, mode="oof")
    hold = slope.calibrated_model(X, y, yr, train_end, cal_end, l2=5.0, gbm=False, mode="holdout")
    te = cm.pop("test_mask")
    p_te = cm.pop("p_test")
    r7p = X[:, FEATURES.index("r7")]
    lhasa_score = r7p[te] / 100.0
    res = {
        "events_used": n_ev, "cells": n_cells, "samples": int(len(y)), "positives": int(y.sum()),
        "train_years": f"{int(yr.min())}-{cal_end}", "cal_years": "leave-one-year-out within training years",
        "holdout_variant": {"train": f"{int(yr.min())}-{train_end}", "calibrate": f"{train_end + 1}-{cal_end}",
                            **{k: hold["test"][k] for k in ("roc_auc", "brier_skill_score", "ece", "mean_predicted", "observed_rate")}},
        "test_years": f"{cal_end + 1}-{int(yr.max())}",
        "logistic": {**cm["test"], "at_p95_equiv": _contingency((p_te >= np.quantile(p_te, 0.95)).astype(int), y[te])},
        "logistic_before_fix": cm["test_before_fix"], "calibrator": cm["calibrator"], "brier_cal": cm["brier_cal"],
        "gradient_boosting": cm.get("gradient_boosting"), "splits": cm["splits"],
        "lhasa_r7": {"roc_auc": slope.roc_auc(lhasa_score, y[te]),
                     "p95": _contingency((r7p[te] >= 95).astype(int), y[te]),
                     "p99": _contingency((r7p[te] >= 99).astype(int), y[te])},
        "roc_logistic": cm["roc"], "roc_lhasa": dict(zip(("fpr", "tpr"), slope.roc_curve(lhasa_score, y[te]))),
        "reliability": cm["reliability"],
        "coefficients": {f: float(c) for f, c in zip(FEATURES, cm["deploy"]["weights"][1:])},
    }
    # leave-one-year-out AUC (robustness across years)
    loyo = []
    for yy in np.unique(yr):
        m_te = yr == yy
        if y[m_te].sum() < 3:
            continue
        Xa, mua, sda = slope.standardize(X[~m_te])
        wa = slope.fit_logistic(Xa, y[~m_te], l2=5.0)
        loyo.append({"year": int(yy), "events": int(y[m_te].sum()), "auc": slope.roc_auc(slope.predict(wa, (X[m_te] - mua) / sda), y[m_te]),
                     "auc_lhasa": slope.roc_auc(r7p[m_te], y[m_te])})
    res["leave_one_year_out"] = loyo
    # deployed model = exactly the validated one (train-only weights + calibration-year calibrator)
    res["deploy"] = {"features": list(FEATURES), **cm["deploy"],
                     "unit": "probability of a reported rain-triggered landslide in a 0.5 deg (~50 km) cell per day"}
    res["caveat"] = ("The GLC is compiled from news and reports: unreported landslides count as 'none', so probabilities are "
                     "lower bounds and remote valleys are under-represented. Location accuracy 5-25 km; analysis cells are 0.5 deg (~50 km).")
    return res
