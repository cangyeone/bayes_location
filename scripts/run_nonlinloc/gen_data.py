#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import skfmm
from pyproj import CRS, Transformer
from concurrent.futures import ProcessPoolExecutor, as_completed


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    meta_json: str = "run_fm3d/run_demo/data/xyz_vp_vs_meta.json"

    # event sampling
    xy_step_events: int = 10
    event_depths_km: Tuple[float, ...] = (0, 5, 10, 15, 20, 25)

    # station & picking
    station_z_km: float = 0.0
    max_epi_km: float = 200.0
    dist_decay_km: float = 80.0
    p_count_range: Tuple[int, int] = (4, 20)
    s_count_range: Tuple[int, int] = (3, 20)

    # output
    out_txt: str = "run_fm3d/run_demo/data/synth_arrivals_skfmm.txt"

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


def station_code_from_lonlat(lon: float, lat: float) -> str:
    L = int(round(lon * 10))
    B = int(round(lat * 10))
    return f"SC_L{L:04d}B{B:04d}"


# =============================================================================
# Load model
# =============================================================================

def load_velocity_model(meta_json: str):
    meta = json.loads(Path(meta_json).read_text())
    axes = np.load(meta["axes_file"])
    x_km = axes["x_km"]
    y_km = axes["y_km"]
    z_km = axes["z_km"]

    arr = np.load(meta["npy_file"], mmap_mode="r")
    Nz, Ny, Nx = meta["shape"]

    VP = arr[:, 3].reshape(Nz, Ny, Nx).astype(np.float32)
    VS = arr[:, 4].reshape(Nz, Ny, Nx).astype(np.float32)

    dx = float(meta["spacing_km"])
    return meta, x_km, y_km, z_km, VP, VS, dx


# =============================================================================
# Build events & stations
# =============================================================================

def build_events(x_km, y_km, z_km, cfg: Config) -> List[Dict]:
    ix_list = range(0, len(x_km), cfg.xy_step_events)
    iy_list = range(0, len(y_km), cfg.xy_step_events)
    iz_list = [nearest_index(z_km, z) for z in cfg.event_depths_km]

    events = []
    eid = 0
    for iy in iy_list:
        for ix in ix_list:
            for iz in iz_list:
                events.append(dict(
                    event_id=eid,
                    ix=ix, iy=iy, iz=iz,
                    x_km=float(x_km[ix]),
                    y_km=float(y_km[iy]),
                    z_km=float(z_km[iz]),
                ))
                eid += 1
    return events


def build_station_grid(x_km, y_km, z_km, z_sta_km: float):
    iz_sta = nearest_index(z_km, z_sta_km)
    xx, yy = np.meshgrid(x_km, y_km, indexing="xy")
    sta_xy = np.column_stack([xx.ravel(), yy.ravel()])
    iy = np.repeat(np.arange(len(y_km)), len(x_km))
    ix = np.tile(np.arange(len(x_km)), len(y_km))
    sta_ij = np.column_stack([iy, ix])
    return iz_sta, sta_xy, sta_ij


# =============================================================================
# Sampling picks
# =============================================================================

def sample_p_s(ev, sta_xy, cfg: Config, rng: random.Random):
    dx = sta_xy[:, 0] - ev["x_km"]
    dy = sta_xy[:, 1] - ev["y_km"]
    dist = np.sqrt(dx * dx + dy * dy)

    cand = np.where(dist <= cfg.max_epi_km)[0]
    if cand.size == 0:
        return None

    w = np.exp(-dist[cand] / cfg.dist_decay_km)
    w /= w.sum()

    kp = min(rng.randint(*cfg.p_count_range), cand.size)
    ks = min(rng.randint(*cfg.s_count_range), cand.size)

    gen = np.random.default_rng(rng.getrandbits(32))
    p_sel = gen.choice(cand, kp, replace=False, p=w)
    overlap = gen.choice(p_sel)
    rest = cand[cand != overlap]

    ks2 = min(max(ks - 1, 0), rest.size)
    s_sel = np.concatenate([[overlap],
                             gen.choice(rest, ks2, replace=False)]) if ks2 > 0 else np.array([overlap])

    return p_sel.astype(int), s_sel.astype(int), dist


# =============================================================================
# FMM per event
# =============================================================================

def solve_event(ev, p_sel, s_sel, dist_all,
                VP, VS, dx, iz_sta, sta_ij,
                transformer, x_km, y_km):

    Nz, Ny, Nx = VP.shape
    phi = np.ones((Nz, Ny, Nx), np.float32)
    phi[ev["iz"], ev["iy"], ev["ix"]] = -1.0

    Tp = skfmm.travel_time(phi, speed=VP, dx=dx)
    Ts = skfmm.travel_time(phi, speed=VS, dx=dx)

    # event lon/lat
    lon_e, lat_e = transformer.transform(ev["x_km"] * 1000, ev["y_km"] * 1000)

    t0 = 600.0 + ev["event_id"] * 10.0

    lines = []
    lines.append(
        f"{ev['event_id']:4d}      2022 09 05 00:10:{ev['event_id']%60:02d}.000"
        f"       {t0:9.3f}   0.0000 {lat_e:8.4f} {lon_e:9.4f}"
        f"   {ev['z_km']:5.2f}   3.00    0.0   0.0    5  3   8   3   0.00"
    )

    for idx in p_sel:
        iy, ix = sta_ij[idx]
        lon, lat = transformer.transform(x_km[ix]*1000, y_km[iy]*1000)
        sta = station_code_from_lonlat(lon, lat)
        at = Tp[iz_sta, iy, ix]
        lines.append(
            f"   SC {sta:<12s}  P     {t0+at:10.4f}   {at:8.4f} "
            f"0.00e+00   0.0000    0.300   {dist_all[idx]:7.4f}   "
            f"{lon:9.4f}   {lat:8.4f}"
        )


    for idx in s_sel:
        iy, ix = sta_ij[idx]
        lon, lat = transformer.transform(x_km[ix]*1000, y_km[iy]*1000)
        sta = station_code_from_lonlat(lon, lat)
        at = Ts[iz_sta, iy, ix]
        lines.append(
            f"   SC {sta:<12s}  S     {t0+at:10.4f}   {at:8.4f} "
            f"0.00e+00   0.0000    0.400   {dist_all[idx]:7.4f}   "
            f"{lon:9.4f}   {lat:8.4f}"
        )


    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = Config()
    rng = random.Random(cfg.seed)

    meta, x_km, y_km, z_km, VP, VS, dx = load_velocity_model(cfg.meta_json)
    transformer = build_inverse_aeqd(meta["lon0"], meta["lat0"])

    events = build_events(x_km, y_km, z_km, cfg)
    iz_sta, sta_xy, sta_ij = build_station_grid(x_km, y_km, z_km, cfg.station_z_km)

    tasks = []
    for ev in events:
        ret = sample_p_s(ev, sta_xy, cfg, rng)
        if ret is None:
            continue
        p_sel, s_sel, dist_all = ret
        tasks.append((ev, p_sel, s_sel, dist_all))

    out_blocks = []

    with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
        futs = [
            ex.submit(
                solve_event,
                ev, p_sel, s_sel, dist_all,
                VP, VS, dx, iz_sta, sta_ij,
                transformer, x_km, y_km
            )
            for ev, p_sel, s_sel, dist_all in tasks
        ]
        for f in as_completed(futs):
            out_blocks.append(f.result())

    out_blocks.sort(key=lambda s: int(s.split()[0]))
    Path(cfg.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_txt).write_text("\n".join(out_blocks) + "\n")
    print(f"Wrote {cfg.out_txt}")


if __name__ == "__main__":
    main()
