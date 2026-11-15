"""Radar-geometry quality masks: layover, radar shadow (local + cast) and local incidence angle.

NISAR's GCOV metadata reports shadowMaskingApplied = False, so these are computed here from the Copernicus DEM and the
product's own line-of-sight unit vectors / incidence angles (metadata cubes interpolated at DEM height).

Range-plane approximation: with h = horizontal LOS direction (ground -> satellite) and alpha = terrain slope angle
measured along h (positive = slope faces the radar):
  local incidence theta_loc = theta - alpha
  layover        if alpha > theta                (slope facing the radar steeper than the incidence angle)
  local shadow   if -alpha > 90 deg - theta      (slope facing away steeper than the grazing angle)
  cast shadow    if terrain between the pixel and the radar rises above the radar ray (ray marching along h)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def layover_shadow(elev: np.ndarray, dx: float, dy: float, los_e: np.ndarray, los_n: np.ndarray, inc_deg: np.ndarray,
                   max_dist_m: float = 4000.0) -> dict[str, np.ndarray]:
    z = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    gy, gx = np.gradient(z, abs(dy), abs(dx))
    dz_de = gx
    dz_dn = -gy if dy < 0 else gy
    hn = np.hypot(los_e, los_n) + 1e-12
    he, hnn = los_e / hn, los_n / hn  # horizontal unit vector towards the satellite
    grad_h = dz_de * he + dz_dn * hnn  # elevation change per metre moving towards the satellite
    alpha = np.degrees(np.arctan(-grad_h))  # >0: terrain descends towards the radar = slope faces the radar
    theta = np.asarray(inc_deg, float)
    theta_loc = theta - alpha
    layover = alpha > theta
    local_shadow = (-alpha) > (90.0 - theta)
    # cast shadow: march towards the satellite; shadowed if terrain rises above the ray with grazing angle (90-theta)
    H, W = z.shape
    rows, cols = np.mgrid[0:H, 0:W].astype(float)
    step = min(abs(dx), abs(dy))
    tan_graze = np.tan(np.radians(90.0 - theta))
    cast = np.zeros(z.shape, bool)
    # unit steps in pixel space (row increases southward when dy < 0)
    dr = -hnn * step / abs(dy) if dy < 0 else hnn * step / abs(dy)
    dc = he * step / abs(dx)
    for k in range(1, int(max_dist_m / step) + 1):
        rr = rows + k * dr
        cc = cols + k * dc
        zk = map_coordinates(z, [rr, cc], order=1, mode="nearest")
        cast |= (zk - z) > k * step * tan_graze
    valid = np.isfinite(elev)
    return {"layover": layover & valid, "shadow": (local_shadow | cast) & valid, "local_incidence_deg": theta_loc,
            "alpha_deg": alpha}


def cube_to_grid(cube: np.ndarray, attrs: dict, x: np.ndarray, y: np.ndarray, elev: np.ndarray) -> np.ndarray:
    from earthpulse.io.nisar import interp_cube

    return interp_cube(cube, attrs["heights"], attrs["cube_x"], attrs["cube_y"], x, y, elev)
