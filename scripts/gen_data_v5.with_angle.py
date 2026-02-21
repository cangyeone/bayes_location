#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import skfmm
from pyproj import CRS, Transformer
from concurrent.futures import ProcessPoolExecutor, as_completed


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    meta_json: str = "run_fm3d/data/xyz_vp_vs_meta.json"

    # event sampling
    xy_step_events: int = 40
    event_depths_km: Tuple[float, ...] = (5, 15, 25, 35)
    events_per_xy_range: Tuple[int, int] = (3, 4)

    # station distribution (INSIDE inset region)
    station_z_km: float = 0.0
    station_inset_km: float = 0.0   # x in [xmin+inset, xmax-inset], y in [ymin+inset, ymax-inset]

    # station & picking
    max_epi_km: float = 300.0
    dist_decay_km: float = 150.0
    p_count_range: Tuple[int, int] = (4, 25)
    s_count_range: Tuple[int, int] = (3, 25)

    # noise + outliers
    # noise + outliers
    sigma_p: float = 1e-3
    sigma_s: float = 1e-3

    # wrong picks: ratio is fraction of picks (0.1 = 10%)
    wrong_pick_ratio: float = 0.0

    # wrong picks have large STD (seconds)
    wrong_sigma_p: float = 8.0
    wrong_sigma_s: float = 8.0

    # optional: enforce |error| >= this for wrong picks (set 0 to disable)
    wrong_min_abs: float = 0.0



    # output
    out_arrivals: str = "run_fm3d/data/angle.synth_arrivals_skfmm_noise.new.txt"
    out_stations: str = "run_fm3d/data/angle.synth_stations.new.txt"
    out_events_vel: str = "run_fm3d/data/angle.synth_events_vel.new.txt"
    out_stations_vel: str = "run_fm3d/data/angle.synth_stations_vel.new.txt"

    # misc
    seed: int = 1234
    n_workers: int = max(1, (os.cpu_count() or 4) - 1)


# =============================================================================
# Utilities
# =============================================================================

def nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def build_inverse_aeqd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(crs_aeqd, CRS.from_epsg(4326), always_xy=True)


def station_code_LB(lon: float, lat: float) -> str:
    L = int(round(lon * 100))
    B = int(round(lat * 100))
    return f"L{L:05d}B{B:05d}"


def station_line_format(lon: float, lat: float, elev_km: float = -0.0) -> str:
    code = station_code_LB(lon, lat)
    return f"SC {code} {lon:8.4f} {lat:8.4f} {elev_km:7.3f}"


# =============================================================================
# Load model
# =============================================================================

def load_velocity_model(meta_json: str):
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    axes = np.load(meta["axes_file"])
    x_km = axes["x_km"]
    y_km = axes["y_km"]
    z_km = axes["z_km"]

    arr = np.load(meta["npy_file"], mmap_mode="r")
    Nz, Ny, Nx = meta["shape"]
    if arr.shape[0] != Nz * Ny * Nx:
        raise RuntimeError(f"npy rows mismatch: got {arr.shape[0]} expected {Nz*Ny*Nx}")

    VP = arr[:, 3].reshape(Nz, Ny, Nx).astype(np.float32)
    VS = arr[:, 4].reshape(Nz, Ny, Nx).astype(np.float32)

    dx = float(meta["spacing_km"])
    return meta, x_km, y_km, z_km, VP, VS, dx


# =============================================================================
# Build events & stations
# =============================================================================

def build_events(x_km, y_km, z_km, cfg: Config, rng: random.Random) -> List[Dict]:
    ix_list = list(range(0, len(x_km), cfg.xy_step_events))
    iy_list = list(range(0, len(y_km), cfg.xy_step_events))
    iz_list = [nearest_index(z_km, z) for z in cfg.event_depths_km]

    events = []
    eid = 0
    for iy in iy_list:
        for ix in ix_list:
            ne = rng.randint(cfg.events_per_xy_range[0], cfg.events_per_xy_range[1])
            for _ in range(ne):
                iz = rng.choice(iz_list)
                events.append(dict(
                    event_id=eid,
                    ix=int(ix), iy=int(iy), iz=int(iz),
                    x_km=float(x_km[ix]),
                    y_km=float(y_km[iy]),
                    z_km=float(z_km[iz]),
                ))
                eid += 1
    return events


def build_station_grid_inside_inset(
    x_km: np.ndarray,
    y_km: np.ndarray,
    z_km: np.ndarray,
    z_sta_km: float,
    inset_km: float,
):
    iz_sta = nearest_index(z_km, z_sta_km)

    x_min, x_max = float(x_km.min()), float(x_km.max())
    y_min, y_max = float(y_km.min()), float(y_km.max())

    ix_mask = (x_km >= x_min + inset_km) & (x_km <= x_max - inset_km)
    iy_mask = (y_km >= y_min + inset_km) & (y_km <= y_max - inset_km)

    ix_valid = np.where(ix_mask)[0]
    iy_valid = np.where(iy_mask)[0]
    if ix_valid.size == 0 or iy_valid.size == 0:
        raise RuntimeError(
            f"No station grid points inside inset region. inset_km={inset_km}\n"
            f"Model box x=[{x_min},{x_max}] y=[{y_min},{y_max}] -> "
            f"inset box x=[{x_min+inset_km},{x_max-inset_km}] "
            f"y=[{y_min+inset_km},{y_max-inset_km}]"
        )

    xx, yy = np.meshgrid(x_km[ix_valid], y_km[iy_valid], indexing="xy")
    sta_xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    iy_full = np.repeat(iy_valid, ix_valid.size)
    ix_full = np.tile(ix_valid, iy_valid.size)
    sta_ij = np.column_stack([iy_full, ix_full]).astype(np.int32)

    return iz_sta, sta_xy, sta_ij


# =============================================================================
# Sampling picks per event
# =============================================================================

def sample_p_s(ev, sta_xy, cfg: Config, rng: random.Random):
    dx = sta_xy[:, 0] - ev["x_km"]
    dy = sta_xy[:, 1] - ev["y_km"]
    dist = np.sqrt(dx * dx + dy * dy)

    cand = np.where(dist <= cfg.max_epi_km)[0]
    if cand.size == 0:
        return None

    w = np.exp(-dist[cand] / cfg.dist_decay_km)
    wsum = float(w.sum())
    if wsum <= 0:
        return None
    w /= wsum

    kp = min(rng.randint(*cfg.p_count_range), cand.size)
    ks = min(rng.randint(*cfg.s_count_range), cand.size)

    gen = np.random.default_rng(rng.getrandbits(32))
    p_sel = gen.choice(cand, kp, replace=False, p=w)

    # ensure at least 1 station has both P and S
    overlap = int(gen.choice(p_sel))
    rest = cand[cand != overlap]
    ks2 = min(max(ks - 1, 0), rest.size)

    if ks2 > 0:
        s_sel = np.concatenate([[overlap], gen.choice(rest, ks2, replace=False)])
    else:
        s_sel = np.array([overlap], dtype=int)

    return p_sel.astype(int), s_sel.astype(int), dist.astype(np.float32)


# =============================================================================
# Noise
# =============================================================================

def add_pick_noise(
    t_true: float,
    sigma_ok: float,
    rng: np.random.Generator,
    is_wrong: bool,
    wrong_sigma: float,
    wrong_min_abs: float = 0.0,
) -> float:
    """
    - Normal pick:  t_true + N(0, sigma_ok)
    - Wrong pick:   t_true + N(0, wrong_sigma) (optionally enforce |err| >= wrong_min_abs)
    """
    if not is_wrong:
        return t_true + float(rng.normal(0.0, sigma_ok))

    # wrong pick: large-STD error
    err = float(rng.normal(0.0, wrong_sigma))
    if wrong_min_abs > 0.0:
        # resample a few times to avoid tiny errors
        for _ in range(10):
            if abs(err) >= wrong_min_abs:
                break
            err = float(rng.normal(0.0, wrong_sigma))
        # last resort: force sign if still too small
        if abs(err) < wrong_min_abs:
            err = (1.0 if rng.random() < 0.5 else -1.0) * wrong_min_abs

    return t_true + err


def build_events(x_km, y_km, z_km, cfg: Config, rng: random.Random) -> List[Dict]:
    ix_list = list(range(0, len(x_km), cfg.xy_step_events))
    iy_list = list(range(0, len(y_km), cfg.xy_step_events))
    iz_list = [nearest_index(z_km, z) for z in cfg.event_depths_km]

    events = []
    eid = 0
    for iy in iy_list:
        for ix in ix_list:
            ne = rng.randint(cfg.events_per_xy_range[0], cfg.events_per_xy_range[1])
            for _ in range(ne):
                iz = int(rng.choice(iz_list))  # allow repeats -> enables reuse
                events.append(dict(
                    event_id=eid,
                    ix=int(ix), iy=int(iy), iz=int(iz),
                    x_km=float(x_km[ix]),
                    y_km=float(y_km[iy]),
                    z_km=float(z_km[iz]),
                ))
                eid += 1
    return events


def group_events_by_source(events: List[Dict]) -> Dict[Tuple[int, int, int], List[Dict]]:
    groups: Dict[Tuple[int, int, int], List[Dict]] = {}
    for ev in events:
        key = (ev["ix"], ev["iy"], ev["iz"])
        groups.setdefault(key, []).append(ev)
    return groups

# =============================================================================
# Worker: FMM per event
# =============================================================================

def solve_event(
    task: Tuple[Dict, np.ndarray, np.ndarray, np.ndarray],
    VP: np.ndarray,
    VS: np.ndarray,
    dx: float,
    iz_sta: int,
    sta_ij: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    lon0: float,
    lat0: float,
    cfg: Config,
) -> Tuple[int, str]:
    ev, p_sel, s_sel, dist_all = task

    seed = (cfg.seed * 1000003 + ev["event_id"] * 9176) & 0xFFFFFFFF
    rng_np = np.random.default_rng(seed)

    Nz, Ny, Nx = VP.shape
    phi = np.ones((Nz, Ny, Nx), dtype=np.float32)
    phi[ev["iz"], ev["iy"], ev["ix"]] = -1.0

    Tp = skfmm.travel_time(phi, speed=VP, dx=dx)
    Ts = skfmm.travel_time(phi, speed=VS, dx=dx)

    transformer = build_inverse_aeqd(lon0, lat0)

    lon_e, lat_e = transformer.transform(ev["x_km"] * 1000.0, ev["y_km"] * 1000.0)
    t0 = 600.0 + ev["event_id"] * 10.0
    vp_src = float(VP[ev["iz"], ev["iy"], ev["ix"]])  # km/s at source

    lines: List[str] = []
    lines.append(
        f"{ev['event_id']:4d}      2022 09 05 00:10:{ev['event_id']%60:02d}.000"
        f"       {t0:9.3f}   0.0000 {lat_e:8.4f} {lon_e:9.4f}"
        f"   {ev['z_km']:5.2f}   3.00    0.0   0.0    5  3   8   3   {vp_src:6.3f}"
    )

    nP = int(len(p_sel))
    nS = int(len(s_sel))
    total = nP + nS
    n_wrong = int(round(cfg.wrong_pick_ratio * total))
    wrong_mask = np.zeros(total, dtype=bool)
    if n_wrong > 0:
        wrong_idx = rng_np.choice(np.arange(total), size=min(n_wrong, total), replace=False)
        wrong_mask[wrong_idx] = True

    # P
    for i_pick, idx in enumerate(p_sel):
        iy, ix = sta_ij[idx]
        lon_s, lat_s = transformer.transform(float(x_km[ix]) * 1000.0, float(y_km[iy]) * 1000.0)
        sta_code = station_code_LB(lon_s, lat_s)

        t_true = float(Tp[iz_sta, iy, ix])
        t_obs = add_pick_noise(
            t_true=t_true,
            sigma_ok=cfg.sigma_p,
            rng=rng_np,
            is_wrong=bool(wrong_mask[i_pick]),
            wrong_sigma=cfg.wrong_sigma_p,
            wrong_min_abs=cfg.wrong_min_abs,
        )
        at_abs = t0 + t_obs
        vp_sta = float(VP[iz_sta, iy, ix])  # km/s at station (P)
        lines.append(
            f"   SC {sta_code:<10s}  P     {at_abs:10.4f}   {t_obs:8.4f} "
            f"0.00e+00   {vp_sta:8.4f}    0.300   {float(dist_all[idx]):7.4f}   "
            f"{lon_s:9.4f}   {lat_s:8.4f}"
        )

    # S
    for j_pick, idx in enumerate(s_sel):
        iy, ix = sta_ij[idx]
        lon_s, lat_s = transformer.transform(float(x_km[ix]) * 1000.0, float(y_km[iy]) * 1000.0)
        sta_code = station_code_LB(lon_s, lat_s)

        t_true = float(Ts[iz_sta, iy, ix])
        
        t_obs = add_pick_noise(
            t_true=t_true,
            sigma_ok=cfg.sigma_s,
            rng=rng_np,
            is_wrong=bool(wrong_mask[nP + j_pick]),
            wrong_sigma=cfg.wrong_sigma_s,
            wrong_min_abs=cfg.wrong_min_abs,
        )
        at_abs = t0 + t_obs
        vs_sta = float(VS[iz_sta, iy, ix])  # km/s at station (S)

        lines.append(
            f"   SC {sta_code:<10s}  S     {at_abs:10.4f}   {t_obs:8.4f} "
            f"0.00e+00   {vs_sta:8.4f}    0.400   {float(dist_all[idx]):7.4f}   "
            f"{lon_s:9.4f}   {lat_s:8.4f}"
        )

    return ev["event_id"], "\n".join(lines)

def solve_source_group(
    key_and_events,
    VP: np.ndarray,
    VS: np.ndarray,
    dx: float,
    iz_sta: int,
    sta_xy: np.ndarray,
    sta_ij: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    lon0: float,
    lat0: float,
    cfg: Config,
) -> List[Tuple[int, str]]:
    (ix_src, iy_src, iz_src), ev_list = key_and_events

    # 1) FMM once per (ix,iy,iz)
    Nz, Ny, Nx = VP.shape
    phi = np.ones((Nz, Ny, Nx), dtype=np.float32)
    phi[iz_src, iy_src, ix_src] = -1.0

    Tp = skfmm.travel_time(phi, speed=VP, dx=dx)
    Ts = skfmm.travel_time(phi, speed=VS, dx=dx)

    dTp_dz, dTp_dy, dTp_dx = np.gradient(Tp, dx, dx, dx)
    dTs_dz, dTs_dy, dTs_dx = np.gradient(Ts, dx, dx, dx)

    def _takeoff_from_source_deg(
        dT_dx, dT_dy, dT_dz,
        ix_src, iy_src, iz_src,
        ix_sta, iy_sta, iz_sta,
        dx: float,
        nstep_max: int = 20000,
        step_frac: float = 0.5,
        r_stop_cell: float = 1.5,
        eps: float = 1e-12,
    ) -> float:
        """
        Back-trace from station to source along -∇T, then estimate takeoff direction at source.

        Angle definition (IMPORTANT):
        - assume z axis is POSITIVE DOWN (depth increases with z index)
        - takeoff angle: DOWN=0°, HORIZONTAL=90°, UP=180°
            => angle = arccos(vz_hat) in degrees

        Returns NaN if tracing fails.
        """
        # start from station center
        x = float(ix_sta) + 0.5
        y = float(iy_sta) + 0.5
        z = float(iz_sta) + 0.5

        xs = float(ix_src) + 0.5
        ys = float(iy_src) + 0.5
        zs = float(iz_src) + 0.5

        step = float(step_frac)  # in cell units (dx cancels out since grid uniform)

        last_dir_to_source = None  # direction FROM source (we will infer near source)

        for _ in range(nstep_max):
            # stop if close to source
            dxs = x - xs
            dys = y - ys
            dzs = z - zs
            r = (dxs*dxs + dys*dys + dzs*dzs) ** 0.5
            if r <= r_stop_cell:
                break

            # sample gradient at nearest grid point
            ix = int(np.clip(np.floor(x), 0, dT_dx.shape[2] - 1))
            iy = int(np.clip(np.floor(y), 0, dT_dx.shape[1] - 1))
            iz = int(np.clip(np.floor(z), 0, dT_dx.shape[0] - 1))

            gx = float(dT_dx[iz, iy, ix])
            gy = float(dT_dy[iz, iy, ix])
            gz = float(dT_dz[iz, iy, ix])

            n = (gx*gx + gy*gy + gz*gz) ** 0.5
            if (not np.isfinite(n)) or n <= eps:
                return float("nan")

            # direction of increasing T is +∇T; to go toward source we go along -∇T
            ux = -gx / n
            uy = -gy / n
            uz = -gz / n

            # update position (toward source)
            x_new = x + step * ux
            y_new = y + step * uy
            z_new = z + step * uz

            # if we overshoot / get stuck, bail
            if (abs(x_new - x) + abs(y_new - y) + abs(z_new - z)) < 1e-6:
                return float("nan")

            x, y, z = x_new, y_new, z_new

            # store the direction that points from current point toward source (which is our step direction);
            # near source, the takeoff direction FROM source to station is the opposite of this.
            last_dir_to_source = (ux, uy, uz)

        if last_dir_to_source is None:
            return float("nan")

        # Direction FROM source to station (takeoff direction) is opposite of "to_source"
        ux, uy, uz = last_dir_to_source
        vx = -ux
        vy = -uy
        vz = -uz

        # normalize
        nn = (vx*vx + vy*vy + vz*vz) ** 0.5
        if nn <= eps:
            return float("nan")
        vx /= nn; vy /= nn; vz /= nn

        # takeoff angle: DOWN=0°, UP=180° (z positive down)
        ang = float(np.degrees(np.arccos(np.clip(vz, -1.0, 1.0))))
        return ang
    # 2) precompute distance/candidates/weights for this (ix,iy) (same for all events in group)
    dxs = sta_xy[:, 0] - float(x_km[ix_src])
    dys = sta_xy[:, 1] - float(y_km[iy_src])
    dist_all = np.sqrt(dxs * dxs + dys * dys).astype(np.float32)

    cand = np.where(dist_all <= cfg.max_epi_km)[0]
    if cand.size == 0:
        return []

    w = np.exp(-dist_all[cand] / cfg.dist_decay_km).astype(np.float64)
    wsum = float(w.sum())
    if wsum <= 0:
        return []
    w /= wsum

    transformer = build_inverse_aeqd(lon0, lat0)

    out: List[Tuple[int, str]] = []
    vp_src = float(VP[iz_src, iy_src, ix_src])  # 源点 P 波速度
    vs_src = float(VS[iz_src, iy_src, ix_src])  # 源点 S 波速度
    # 3) per-event generation (fast)
    for ev in ev_list:
        # deterministic RNG per event
        seed = (cfg.seed * 1000003 + ev["event_id"] * 9176) & 0xFFFFFFFF
        rng_np = np.random.default_rng(seed)

        # sample stations (different per event)
        kp = min(int(rng_np.integers(cfg.p_count_range[0], cfg.p_count_range[1] + 1)), cand.size)
        ks = min(int(rng_np.integers(cfg.s_count_range[0], cfg.s_count_range[1] + 1)), cand.size)

        p_sel = rng_np.choice(cand, kp, replace=False, p=w)

        # ensure overlap station for P+S
        overlap = int(rng_np.choice(p_sel))
        rest = cand[cand != overlap]
        ks2 = min(max(ks - 1, 0), rest.size)
        if ks2 > 0:
            s_sel = np.concatenate([[overlap], rng_np.choice(rest, ks2, replace=False)])
        else:
            s_sel = np.array([overlap], dtype=int)

        # event lon/lat + header
        lon_e, lat_e = transformer.transform(ev["x_km"] * 1000.0, ev["y_km"] * 1000.0)
        t0 = 600.0 + ev["event_id"] * 10.0

        lines = []
        lines.append(
            f"{ev['event_id']:4d}      2022 09 05 00:10:{ev['event_id']%60:02d}.000"
            f"       {t0:9.3f}   0.0000 {lat_e:8.4f} {lon_e:9.4f}"
            f"   {ev['z_km']:5.2f}   3.00    0.0   0.0    5  3   8   3   {vp_src:.3f}/{vs_src:.3f}"
        )

        # wrong-pick mask (10% of all picks in this event)
        total = len(p_sel) + len(s_sel)
        n_wrong = int(round(cfg.wrong_pick_ratio * total))
        wrong_mask = np.zeros(total, dtype=bool)
        if n_wrong > 0:
            wrong_idx = rng_np.choice(np.arange(total), size=min(n_wrong, total), replace=False)
            wrong_mask[wrong_idx] = True

        # P picks
        for i_pick, idx in enumerate(p_sel):
            iy, ix = sta_ij[idx]
            lon_s, lat_s = transformer.transform(float(x_km[ix]) * 1000.0, float(y_km[iy]) * 1000.0)
            sta_code = station_code_LB(lon_s, lat_s)
            takeoff_p = _takeoff_from_source_deg(
                dTp_dx, dTp_dy, dTp_dz,
                ix_src, iy_src, iz_src,
                ix, iy, iz_sta,
                dx=dx
            )

            t_true = float(Tp[iz_sta, iy, ix])
            t_obs = add_pick_noise(
                t_true=t_true,
                sigma_ok=cfg.sigma_p,
                rng=rng_np,
                is_wrong=bool(wrong_mask[i_pick]),
                wrong_sigma=cfg.wrong_sigma_p,
                wrong_min_abs=cfg.wrong_min_abs,
            )
            at_abs = t0 + t_obs
            vp_sta = float(VP[iz_sta, iy, ix])
            vs_sta = float(VS[iz_sta, iy, ix]) 
            lines.append(
                f"   SC {sta_code:<10s}  P     {at_abs:10.4f}   {t_obs:8.4f} "
                f"{takeoff_p:8.3f}   {vp_sta:.3f}/{vs_sta:.3f}    0.300   {float(dist_all[idx]):7.4f}   "
                f"{lon_s:9.4f}   {lat_s:8.4f}"
            )

        # S picks
        offset = len(p_sel)
        for j_pick, idx in enumerate(s_sel):
            iy, ix = sta_ij[idx]
            lon_s, lat_s = transformer.transform(float(x_km[ix]) * 1000.0, float(y_km[iy]) * 1000.0)
            sta_code = station_code_LB(lon_s, lat_s)
            takeoff_s = _takeoff_from_source_deg(
                dTs_dx, dTs_dy, dTs_dz,
                ix_src, iy_src, iz_src,
                ix, iy, iz_sta,
                dx=dx,
            )

            t_true = float(Ts[iz_sta, iy, ix])
            t_obs = add_pick_noise(
                t_true=t_true,
                sigma_ok=cfg.sigma_s,
                rng=rng_np,
                is_wrong=bool(wrong_mask[offset + j_pick]),
                wrong_sigma=cfg.wrong_sigma_s,
                wrong_min_abs=cfg.wrong_min_abs,
            )
            at_abs = t0 + t_obs
            vp_sta = float(VP[iz_sta, iy, ix])
            vs_sta = float(VS[iz_sta, iy, ix]) 
            lines.append(
                f"   SC {sta_code:<10s}  S     {at_abs:10.4f}   {t_obs:8.4f} "
                f"{takeoff_s:8.3f}   {vp_sta:.3f}/{vs_sta:.3f}   0.400   {float(dist_all[idx]):7.4f}   "
                f"{lon_s:9.4f}   {lat_s:8.4f}"
            )

        out.append((ev["event_id"], "\n".join(lines)))

    return out

# =============================================================================
# Main
# =============================================================================

def main():
    cfg = Config()
    rng = random.Random(cfg.seed)

    meta, x_km, y_km, z_km, VP, VS, dx = load_velocity_model(cfg.meta_json)
    lon0 = float(meta["lon0"])
    lat0 = float(meta["lat0"])

    # stations
    iz_sta, sta_xy, sta_ij = build_station_grid_inside_inset(
        x_km, y_km, z_km,
        z_sta_km=cfg.station_z_km,
        inset_km=cfg.station_inset_km,
    )

    # write stations file
    transformer = build_inverse_aeqd(lon0, lat0)
    sta_lines = []
    for (iy, ix) in sta_ij:
        lon_s, lat_s = transformer.transform(float(x_km[ix]) * 1000.0, float(y_km[iy]) * 1000.0)
        sta_lines.append(station_line_format(lon_s, lat_s, elev_km=-0.0))

    Path(cfg.out_stations).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_stations).write_text("\n".join(sta_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote stations: {cfg.out_stations}  (N={len(sta_lines)})")

    # events
    events = build_events(x_km, y_km, z_km, cfg, rng)
    print(f"[INFO] events sampled: {len(events)}")

    # tasks
    tasks = []
    skipped = 0
    for ev in events:
        ret = sample_p_s(ev, sta_xy, cfg, rng)
        if ret is None:
            skipped += 1
            continue
        p_sel, s_sel, dist_all = ret
        tasks.append((ev, p_sel, s_sel, dist_all))
    print(f"[INFO] tasks ready: {len(tasks)} (skipped={skipped})")

    # parallel solve
    out_blocks: List[Tuple[int, str]] = []
    groups = group_events_by_source(events)
    group_items = list(groups.items())
    print(f"[INFO] unique sources: {len(group_items)}  (events={len(events)})")

    out_blocks: List[Tuple[int, str]] = []
    with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
        futs = [
            ex.submit(
                solve_source_group,
                item,
                VP, VS, dx,
                iz_sta, sta_xy, sta_ij,
                x_km, y_km,
                lon0, lat0,
                cfg
            )
            for item in group_items
        ]
        for f in as_completed(futs):
            out_blocks.extend(f.result())

    out_blocks.sort(key=lambda t: t[0])
    Path(cfg.out_arrivals).write_text("\n".join(b for _, b in out_blocks) + "\n", encoding="utf-8")


    out_blocks.sort(key=lambda t: t[0])
    out_text = "\n".join(block for _, block in out_blocks) + "\n"

    Path(cfg.out_arrivals).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_arrivals).write_text(out_text, encoding="utf-8")
    print(f"[OK] wrote arrivals: {cfg.out_arrivals}  (events={len(out_blocks)})")


if __name__ == "__main__":
    main()
