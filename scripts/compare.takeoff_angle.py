#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer


# -------------------------
# 1) AEQD projection (lon,lat -> x,y in km)
# -------------------------
def build_aeqd_fwd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(CRS.from_epsg(4326), crs_aeqd, always_xy=True)  # lon,lat -> x,y (m)


# -------------------------
# 2) Parse your output text
# -------------------------
@dataclass
class PickRec:
    event_id: int
    phase: str              # 'P' or 'S'
    takeoff_fmm: float      # degrees
    sta_lon: float
    sta_lat: float

@dataclass
class EventRec:
    event_id: int
    src_lon: float
    src_lat: float
    src_depth_km: float


def parse_sum_like_text(path: str) -> Tuple[Dict[int, EventRec], List[PickRec]]:
    """
    Parse your lines like:
      <eid> 2022 09 05 ... lat lon depth ...
      SC <sta> P <at_abs> <t_obs> <takeoff> <vp/vs> <...> <lon> <lat>
    """
    # header line starts with event_id + date tokens
    header_re = re.compile(
        r"^\s*(\d+)\s+\d{4}\s+\d{2}\s+\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s+"
        r"([0-9.]+)\s+[0-9.]+\s+([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)"
    )
    # pick line: we just fish numbers by splitting
    # Example:
    # SC L09670B02084  P       620.8436    20.8436   99.797   4.583/2.703    0.300   120.4159     96.6987    20.8361
    pick_re = re.compile(r"^\s*SC\s+(\S+)\s+([PS])\s+")

    events: Dict[int, EventRec] = {}
    picks: List[PickRec] = []

    cur_eid = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            m = header_re.match(line)
            if m:
                eid = int(m.group(1))
                # m.group(2) is origin time (t0) numeric; not used
                lat_e = float(m.group(3))
                lon_e = float(m.group(4))
                dep_e = float(m.group(5))
                events[eid] = EventRec(event_id=eid, src_lon=lon_e, src_lat=lat_e, src_depth_km=dep_e)
                cur_eid = eid
                continue

            m2 = pick_re.match(line)
            if m2 and cur_eid is not None:
                phase = m2.group(2)

                parts = line.split()
                # Expected tokens:
                # 0 SC
                # 1 station
                # 2 phase
                # 3 at_abs
                # 4 t_obs
                # 5 takeoff_angle
                # ...
                # last two are lon lat
                try:
                    takeoff = float(parts[5])
                    sta_lon = float(parts[-2])
                    sta_lat = float(parts[-1])
                except Exception:
                    continue

                picks.append(PickRec(
                    event_id=cur_eid,
                    phase=phase,
                    takeoff_fmm=takeoff,
                    sta_lon=sta_lon,
                    sta_lat=sta_lat
                ))

    return events, picks


# -------------------------
# 3) Your PINN model + takeoff computation
# -------------------------
class PINNTravelTime(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net_merge = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softplus(),
        )

    def forward(self, x, xs):
        # x : (B,3) receiver [xr,yr,zr] in km
        # xs: (B,3) source   [xe,ye,ze] in km
        inp = torch.cat([x, xs.expand_as(x)], dim=-1)  # (B,6)
        inp = inp / 1000.0
        out = self.net_merge(inp)
        out = out * 10.0
        return out


def takeoff_deg_from_pinn(model: nn.Module, x_km: torch.Tensor, xs_km: torch.Tensor) -> torch.Tensor:
    """
    Return takeoff angle at source in degrees, with definition:
      DOWN=0°, UP=180° (z is depth positive downward)
    Uses grad wrt xs of Tp.
    """
    # ensure 2d
    if x_km.dim() == 1:
        x_km = x_km[None, :]
    if xs_km.dim() == 1:
        xs_km = xs_km[None, :]

    xs_req = xs_km.clone().detach().requires_grad_(True)
    out = model(x_km, xs_req)
    Tp = out[:, 0]

    grad = torch.autograd.grad(Tp.sum(), xs_req, create_graph=False, retain_graph=False)[0]  # (B,3)

    # ray direction from source -> receiver is -grad (per your convention)
    dir_vec = -grad
    dir_unit = dir_vec / (dir_vec.norm(dim=-1, keepdim=True) + 1e-12)

    vx, vy, vz = dir_unit[:, 0], dir_unit[:, 1], dir_unit[:, 2]  # z is depth positive down
    horiz = torch.sqrt(vx * vx + vy * vy)

    # angle from vertical down, range [0, 180]
    ang = torch.atan2(horiz, vz) * (180.0 / math.pi)
    return ang.detach()


# -------------------------
# 4) Main: compute NN takeoff per pick and plot vs longitude
# -------------------------
def main(
    in_txt: str,
    ckpt_path: str,
    lon0: float = 102.5,
    lat0: float = 27.5,
    device: str = "mps",        # "cuda" / "cpu" / "mps"
    batch_size: int = 4096,
    out_prefix: str = "takeoff_vs_lon",
):
    events, picks = parse_sum_like_text(in_txt)
    if not picks:
        raise RuntimeError("No picks parsed. Check input format / indices.")

    tf = build_aeqd_fwd(lon0, lat0)

    # prepare arrays for NN computation
    # receiver z: your station is on surface -> set zr=0 km (or iz_sta depth if you want)
    rec_list = []
    src_list = []
    meta = []  # (phase, sta_lon, takeoff_fmm)
    for p in picks:
        ev = events.get(p.event_id, None)
        if ev is None:
            continue
        xs_m, ys_m = tf.transform(ev.src_lon, ev.src_lat)
        xr_m, yr_m = tf.transform(p.sta_lon, p.sta_lat)

        xs_km = xs_m / 1000.0
        ys_km = ys_m / 1000.0
        xr_km = xr_m / 1000.0
        yr_km = yr_m / 1000.0

        rec_list.append([xr_km, yr_km, 0.0])
        src_list.append([xs_km, ys_km, float(ev.src_depth_km)])  # depth positive down
        dist = np.sqrt((xr_km - xs_km) ** 2 + (yr_km - ys_km) ** 2)
        meta.append((p.phase, p.sta_lon, p.takeoff_fmm, dist))

    rec = torch.tensor(rec_list, dtype=torch.float32, device=device)
    src = torch.tensor(src_list, dtype=torch.float32, device=device)

    # load model
    model = PINNTravelTime()
    sd = torch.load(ckpt_path, map_location="cpu")["model_state"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(device=device, dtype=torch.float32)

    # batch compute NN angles
    nn_angles = np.empty((rec.shape[0],), dtype=np.float32)
    with torch.enable_grad():
        for i0 in range(0, rec.shape[0], batch_size):
            i1 = min(i0 + batch_size, rec.shape[0])
            ang = takeoff_deg_from_pinn(model, rec[i0:i1], src[i0:i1])
            nn_angles[i0:i1] = ang.detach().cpu().numpy()

    # split by phase and plot: takeoff vs station longitude
    def _plot_phase(phase: str):
        lon = []
        fmm = []
        nn  = []
        ds = []
        for i, (ph, sta_lon, takeoff_fmm, dist) in enumerate(meta):
            if ph != phase:
                continue
            lon.append(sta_lon)
            fmm.append(takeoff_fmm)
            nn.append(float(nn_angles[i]))
            ds.append(dist)

        lon = np.asarray(lon, dtype=np.float32)
        fmm = np.asarray(fmm, dtype=np.float32)
        nn  = np.asarray(nn,  dtype=np.float32)
        ds = np.asarray(ds,  dtype=np.float32) 

        plt.figure()
        plt.scatter(ds, fmm, s=12, label="NN-vs-FMM")
        plt.scatter(ds, nn,  s=12, label="NN (from PINN grad)")
        plt.xlabel("NN")
        plt.ylabel("FMM  [down=0, up=180]")
        plt.title(f"Takeoff angle vs longitude ({phase}-phase)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_png = f"run_fm3d/data/figs/{out_prefix}_{phase}.png"
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[OK] saved {out_png}")

    _plot_phase("P")
    _plot_phase("S")


if __name__ == "__main__":
    # ---- EDIT THESE ----
    IN_TXT = "run_fm3d/data/angle.synth_arrivals_skfmm_noise.new.txt"          # 你的输出文件路径（含 takeoff 列）
    CKPT   = "ckpt/time.v1.0.pt"    # 你的 PINN checkpoint
    main(
        in_txt=IN_TXT,
        ckpt_path=CKPT,
        lon0=102.5,
        lat0=27.5,
        device="mps",       # "cuda" / "cpu" / "mps"
        batch_size=2048,
        out_prefix="takeoff_vs_lon",
    )
