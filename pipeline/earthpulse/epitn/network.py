"""EPITN - Earth Process Interaction and Transition Network.

Aligns event layers from BDM/GDM/CDM on one grid and quantifies *where* and *when* processes co-occur:
  * spatial co-occurrence: Jaccard, conditional probability P(B|A), lift = P(A&B)/(P(A)P(B))
  * significance: toroidal-shift permutation test (preserves each layer's spatial autocorrelation, so clustered
    patterns do not produce spurious significance like a naive pixel shuffle would)
  * temporal ordering: onset lag distribution (B - A) in co-located cells + sign test
  * transition network: nodes = process states, directed edges A->B when A's onset precedes B's within a window

Results describe association, never causation - this is stated in every exported summary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import binomtest


@dataclass
class EventLayer:
    name: str  # e.g. "forest disturbance (BDM)"
    model: str  # bdm | gdm | cdm
    event: np.ndarray  # (H, W) bool
    onset: np.ndarray | None = None  # (H, W) decimal year, NaN where no event
    valid: np.ndarray | None = None  # (H, W) bool - where this model could observe at all


def aggregate(layer: EventLayer, factor: int) -> EventLayer:
    """Block-aggregate to a coarser analysis grid (cell = event if any pixel has an event; onset = earliest)."""
    H, W = layer.event.shape
    h, w = H // factor, W // factor
    ev = layer.event[: h * factor, : w * factor].reshape(h, factor, w, factor)
    event = ev.any(axis=(1, 3))
    onset = None
    if layer.onset is not None:
        on = layer.onset[: h * factor, : w * factor].reshape(h, factor, w, factor)
        with np.errstate(all="ignore"):
            onset = np.nanmin(np.where(ev, on, np.nan), axis=(1, 3))
    valid = None
    if layer.valid is not None:
        valid = layer.valid[: h * factor, : w * factor].reshape(h, factor, w, factor).mean(axis=(1, 3)) > 0.5
    return EventLayer(layer.name, layer.model, event, onset, valid)


def _stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    na, nb, nab, n = a.sum(), b.sum(), (a & b).sum(), a.size
    union = (a | b).sum()
    jacc = nab / union if union else 0.0
    cond = nab / na if na else 0.0
    lift = (nab / n) / ((na / n) * (nb / n)) if na and nb else 0.0
    return float(jacc), float(cond), float(lift)


def cooccurrence(a: EventLayer, b: EventLayer, n_perm: int = 999, seed: int = 0) -> dict[str, Any]:
    common = np.ones_like(a.event, bool)
    for l in (a, b):
        if l.valid is not None:
            common &= l.valid
    ea = a.event & common
    eb = b.event & common
    jacc, cond, lift = _stats(ea, eb)
    rng = np.random.default_rng(seed)
    H, W = ea.shape
    exceed = 0
    null_lifts = []
    for _ in range(n_perm):
        sy, sx = rng.integers(1, H), rng.integers(1, W)
        bs = np.roll(np.roll(b.event, sy, 0), sx, 1) & common
        _, _, l2 = _stats(ea, bs)
        null_lifts.append(l2)
        exceed += l2 >= lift
    p = (exceed + 1) / (n_perm + 1)
    out: dict[str, Any] = {
        "a": a.name, "b": b.name, "cells_a": int(ea.sum()), "cells_b": int(eb.sum()),
        "cells_both": int((ea & eb).sum()), "cells_observed": int(common.sum()),
        "jaccard": round(jacc, 4), "p_b_given_a": round(cond, 4), "lift": round(lift, 3),
        "null_lift_median": round(float(np.median(null_lifts)), 3), "p_value": round(float(p), 4),
        "significant": bool(p < 0.05 and lift > 1),
    }
    if a.onset is not None and b.onset is not None:
        both = ea & eb & np.isfinite(a.onset) & np.isfinite(b.onset)
        lag_days = (b.onset[both] - a.onset[both]) * 365.25
        if lag_days.size:
            pos, neg = int((lag_days > 0).sum()), int((lag_days < 0).sum())
            sign_p = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg > 0 else 1.0
            out["lag"] = {
                "n": int(lag_days.size), "median_days": round(float(np.median(lag_days)), 1),
                "iqr_days": [round(float(np.percentile(lag_days, 25)), 1), round(float(np.percentile(lag_days, 75)), 1)],
                "a_first": pos, "b_first": neg, "simultaneous": int((lag_days == 0).sum()),
                "sign_test_p": round(float(sign_p), 4),
                "histogram": np.histogram(lag_days, bins=12)[0].tolist(),
                "histogram_edges": np.round(np.histogram(lag_days, bins=12)[1], 1).tolist(),
            }
    return out


def build_network(layers: list[EventLayer], n_perm: int = 499, window_days: float = 120.0) -> dict[str, Any]:
    nodes = [{"id": l.name, "model": l.model, "cells": int(l.event.sum())} for l in layers]
    pairs, edges = [], []
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            r = cooccurrence(layers[i], layers[j], n_perm=n_perm, seed=i * 31 + j)
            pairs.append(r)
            lag = r.get("lag")
            if r["cells_both"] == 0:
                continue
            # direction by dominant onset order within the window
            if lag and lag["n"] >= 5 and abs(lag["median_days"]) <= window_days and lag["sign_test_p"] < 0.1:
                src, dst = (r["a"], r["b"]) if lag["median_days"] > 0 else (r["b"], r["a"])
                directed = True
            else:
                src, dst, directed = r["a"], r["b"], False
            edges.append({"source": src, "target": dst, "directed": directed, "weight": r["cells_both"],
                          "lift": r["lift"], "p_value": r["p_value"], "significant": r["significant"]})
    return {
        "nodes": nodes, "edges": edges, "pairs": pairs,
        "disclaimer": "Spatial or temporal co-occurrence shows association, not causation. Significance is "
                      "tested against toroidal-shift permutations that preserve each layer's spatial clustering.",
    }
