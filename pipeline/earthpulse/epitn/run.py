"""Run EPITN for a region: bring BDM/GDM/CDM event layers onto one common grid and build the network."""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from earthpulse.config import WORK_DIR, Region
from earthpulse.epitn.network import EventLayer, aggregate, build_network
from earthpulse.temporal.framework import to_decimal_years

log = logging.getLogger(__name__)
CELL_M = 160.0  # common analysis cell


def _resample(src_x, src_y, arr, dst_x, dst_y, order=0):
    """Nearest-neighbour resample between regular grids in the same CRS."""
    fx = (dst_x - src_x[0]) / (src_x[1] - src_x[0])
    fy = (dst_y - src_y[0]) / (src_y[1] - src_y[0])
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    a = np.asarray(arr, float)
    out = map_coordinates(np.nan_to_num(a, nan=-9999), [FY, FX], order=order, mode="constant", cval=-9999)
    return np.where(out == -9999, np.nan, out)


def run_epitn(region: Region) -> dict[str, Any]:
    wd = WORK_DIR / region.id
    srcs: list[tuple[str, Any]] = []
    if (wd / "bdm" / "bdm.npz").exists():
        srcs.append(("bdm", np.load(wd / "bdm" / "bdm.npz")))
    gdm_stacks = sorted((wd / "gdm").glob("stack_D*.npz")) + sorted((wd / "gdm").glob("stack_A*.npz"))
    if gdm_stacks:
        srcs.append(("gdm", np.load(gdm_stacks[0])))
    if (wd / "cdm" / "goff_velocity.npz").exists():
        srcs.append(("cdm", np.load(wd / "cdm" / "goff_velocity.npz")))
    if len(srcs) < 2:
        raise RuntimeError(f"EPITN needs >= 2 model outputs for {region.id}, found {[s[0] for s in srcs]}")
    epsg = int(srcs[0][1]["epsg"])
    xmin = max(float(s["x"].min()) for _, s in srcs)
    xmax = min(float(s["x"].max()) for _, s in srcs)
    ymin = max(float(s["y"].min()) for _, s in srcs)
    ymax = min(float(s["y"].max()) for _, s in srcs)
    gx = np.arange(xmin + CELL_M / 2, xmax, CELL_M)
    gy = np.arange(ymax - CELL_M / 2, ymin, -CELL_M)
    fine = 4  # build at CELL/4 then aggregate (any-event, earliest onset)
    fx = np.arange(xmin + CELL_M / fine / 2, xmax, CELL_M / fine)
    fy = np.arange(ymax - CELL_M / fine / 2, ymin, -CELL_M / fine)
    layers: list[EventLayer] = []
    for name, z in srcs:
        if name == "bdm":
            dates = np.array(z["dates"]).astype("datetime64[D]")
            dy = to_decimal_years(dates)
            fd = z["first_dark"].astype(float)
            onset = np.where(fd >= 0, dy[np.clip(fd.astype(int), 0, len(dy) - 1)], np.nan)
            ev = _resample(z["x"], z["y"], z["disturbance"].astype(float), fx, fy) > 0.5
            on = _resample(z["x"], z["y"], np.where(z["disturbance"], onset, np.nan), fx, fy)
            valid = np.isfinite(_resample(z["x"], z["y"], np.where(z["forest"], 1.0, np.nan), fx, fy))
            layers.append(EventLayer("forest disturbance (BDM)", "bdm", ev, on, valid))
        elif name == "gdm":
            v = z["p_rate_per_yr"]
            se = z["p_rate_se_eff"] if "p_rate_se_eff" in z.files else z["p_rate_se"]
            rel = (z["p_reliable"] > 0.5) if "p_reliable" in z.files else np.isfinite(v)
            sig = rel & np.isfinite(v) & (np.abs(v) > np.maximum(2 * se, 0.02))
            ev = _resample(z["x"], z["y"], sig.astype(float), fx, fy) > 0.5
            on = _resample(z["x"], z["y"], np.where(sig, z["p_onset_year"], np.nan), fx, fy)
            valid = np.isfinite(_resample(z["x"], z["y"], np.where(rel, v, np.nan), fx, fy))
            layers.append(EventLayer("ground movement (GDM)", "gdm", ev, on, valid))
        elif name == "cdm":
            sp = z["speed"]
            fast = z["glacier"] & np.isfinite(sp) & (sp > max(5.0, float(np.nanpercentile(sp[z["glacier"]], 50))
                                                           if np.isfinite(sp[z["glacier"]]).any() else 5.0))
            ev = _resample(z["x"], z["y"], fast.astype(float), fx, fy) > 0.5
            valid = np.isfinite(_resample(z["x"], z["y"], np.where(np.isfinite(sp), 1.0, np.nan), fx, fy))
            layers.append(EventLayer("active glacier flow (CDM)", "cdm", ev, None, valid))
    coarse = [aggregate(l, fine) for l in layers]
    net = build_network(coarse, n_perm=499)
    code = np.zeros(coarse[0].event.shape, np.uint8)
    bit = {"bdm": 1, "gdm": 2, "cdm": 4}
    for l in coarse:
        code |= (l.event * bit[l.model]).astype(np.uint8)
    out = wd / "epitn"
    out.mkdir(parents=True, exist_ok=True)
    gx2, gy2 = gx[: code.shape[1]], gy[: code.shape[0]]
    np.savez_compressed(out / "layers.npz", x=gx2, y=gy2, epsg=epsg, overlap_code=np.where(code > 0, code, 0)
                        .astype(float) * np.where(code > 0, 1, np.nan))
    net["cell_m"] = CELL_M
    net["models"] = [l.model for l in coarse]
    (out / "network.json").write_text(json.dumps(net, indent=1, default=str))
    log.info("EPITN %s: %s", region.id, [(p["a"], p["b"], p["lift"], p["p_value"]) for p in net["pairs"]])
    return net
