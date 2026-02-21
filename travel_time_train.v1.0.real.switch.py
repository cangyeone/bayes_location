#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pinn_from_real.py

- Parse REAL-like file (real.txt)
- Build (receiver, source) -> (tp, ts) dataset with NaN for missing phases
- Train PINNTravelTime with masked MSE loss
- Save / Load checkpoints (model + optimizer + epoch + meta)

Run examples:
  # train from scratch, save to ckpt.pt
  python train_pinn_from_real.py \
      --real_txt run_fm3d/run_demo/data/real.txt \
      --meta_json run_fm3d/run_demo/data/xyz_vp_vs_meta.json \
      --epochs 20 --batch_size 8192 --lr 1e-3 \
      --save_ckpt run_fm3d/run_demo/data/pinn_tt.ckpt.pt

  # resume training
  python train_pinn_from_real.py \
      --real_txt run_fm3d/run_demo/data/real.txt \
      --meta_json run_fm3d/run_demo/data/xyz_vp_vs_meta.json \
      --epochs 20 --batch_size 8192 --lr 1e-3 \
      --save_ckpt run_fm3d/run_demo/data/pinn_tt.ckpt.pt \
      --resume run_fm3d/run_demo/data/pinn_tt.ckpt.pt

  # load only for inference / evaluation (no training)
  python train_pinn_from_real.py \
      --real_txt run_fm3d/run_demo/data/real.txt \
      --meta_json run_fm3d/run_demo/data/xyz_vp_vs_meta.json \
      --eval_only \
      --resume run_fm3d/run_demo/data/pinn_tt.ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyproj import CRS, Transformer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Model (as provided)
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
            nn.Linear(hidden_dim, 4),      # <- 2 改 4
            nn.Softplus(),
        )

    def forward(self, x, xs):
        inp = torch.cat([x, xs.expand_as(x)], dim=-1)
        inp = inp / 1000.0
        out = self.net_merge(inp)
        out = out * 10.0
        return out




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime

from pyproj import CRS, Transformer


# -------------------------
# Projection (AEQD)
# -------------------------
def build_aeqd_fwd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(CRS.from_epsg(4326), crs_aeqd, always_xy=True)  # lon,lat -> x,y (m)


# -------------------------
# Station table: china.loc
# -------------------------
def read_china_loc(sta_loc_path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Parse lines like:
      AH A0004 40 117.3518  31.7213    -7 AH.A0004.40 # ...
    Returns: { "AH.A0004": (lon, lat, elev_km) }  elev_km optional (if parse fails -> 0)
    """
    out: Dict[str, Tuple[float, float, float]] = {}
    with open(sta_loc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            net = parts[0]
            sta = parts[1]
            try:
                lon = float(parts[3])
                lat = float(parts[4])
            except Exception:
                continue
            elev_km = 0.0
            try:
                elev_maybe = float(parts[5])
                # 你这个文件里看起来像米或直接是“高程/深度”数值；这里统一按 km 存
                # 如果它本来就是 km，你可以把 /1000 去掉
                elev_km = elev_maybe / 1000.0
            except Exception:
                elev_km = 0.0
            out[f"{net}.{sta}"] = (lon, lat, elev_km)
    return out


# -------------------------
# Parsed structures
# -------------------------
@dataclass
class Pick:
    net: str
    sta: str
    phase: str   # 'P' or 'S'
    rel_t: float # seconds


@dataclass
class EventBlock:
    evid: str
    ev_lon: float
    ev_lat: float
    ev_dep_km: float
    origin_dt: datetime
    picks: List[Pick]


# -------------------------
# PHA parsing helpers
# -------------------------
def _to_dt(y: int, mo: int, d: int, hh: int, mm: int, ss: int, usec: int) -> datetime:
    # 不强行加 timezone；相对差值不受影响（同一文件内一致）
    return datetime(y, mo, d, hh, mm, ss, usec)


def _phase_to_PS(phase_token: str) -> Optional[str]:
    """
    Map Pg/Pn/P* -> P ; Sg/Sn/S* -> S
    """
    p = phase_token.strip()
    if not p:
        return None
    p_up = p.upper()
    if p_up.startswith("P"):
        return "P"
    if p_up.startswith("S"):
        return "S"
    return None


def parse_pha_file(
    pha_path: str,
    station_table: Dict[str, Tuple[float, float, float]],
    rel_t_max: float = 50.0
) -> List[EventBlock]:
    """
    Parse your #EVENT / PHASE format.
    - Only uses TRAVTIME
    - rel_t = pick_abs_time - origin_time
    - keep only P/S after mapping
    - drop rel_t > rel_t_max
    """
    blocks: List[EventBlock] = []
    cur: Optional[EventBlock] = None

    with open(pha_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("#EVENT"):
                # flush previous
                if cur is not None:
                    blocks.append(cur)

                parts = s.split()
                # Example:
                # #EVENT SC.202001312355.0001 eq 2020 01 31 031 15 55 35 550000 LOC  104.519   29.474 DEP        4 MAG   ML  0.5
                if len(parts) < 18:
                    cur = None
                    continue
                evid = parts[1]

                try:
                    y = int(parts[3]); mo = int(parts[4]); d = int(parts[5])
                    hh = int(parts[7]); mm = int(parts[8]); ss = int(parts[9]); usec = int(parts[10])
                    origin_dt = _to_dt(y, mo, d, hh, mm, ss, usec)
                except Exception:
                    cur = None
                    continue

                # find LOC lon lat ; DEP dep
                ev_lon = ev_lat = None
                ev_dep_km = None
                try:
                    i_loc = parts.index("LOC")
                    ev_lon = float(parts[i_loc + 1])
                    ev_lat = float(parts[i_loc + 2])
                except Exception:
                    pass
                try:
                    i_dep = parts.index("DEP")
                    ev_dep_km = float(parts[i_dep + 1])
                except Exception:
                    pass

                if ev_lon is None or ev_lat is None or ev_dep_km is None:
                    cur = None
                    continue

                cur = EventBlock(
                    evid=evid,
                    ev_lon=ev_lon,
                    ev_lat=ev_lat,
                    ev_dep_km=float(ev_dep_km),
                    origin_dt=origin_dt,
                    picks=[]
                )
                continue

            if cur is None:
                continue

            if not s.startswith("PHASE"):
                continue

            parts = s.split()
            # Example:
            # PHASE  SC.202001... eq TRAVTIME SC   HMS 00 BHZ     Pg ttime  2.81  V 2020 01 31 031 15 55 38 360000 ...
            if len(parts) < 20:
                continue

            kind = parts[3]
            if kind != "TRAVTIME":
                continue

            net = parts[4]
            sta = parts[5]
            phase_raw = parts[8]
            phase = _phase_to_PS(phase_raw)
            if phase is None:
                continue

            # pick abs time
            try:
                y = int(parts[12]); mo = int(parts[13]); d = int(parts[14])
                hh = int(parts[16]); mm = int(parts[17]); ss = int(parts[18]); usec = int(parts[19])
                pick_dt = _to_dt(y, mo, d, hh, mm, ss, usec)
            except Exception:
                continue

            rel_t = (pick_dt - cur.origin_dt).total_seconds()
            if rel_t > rel_t_max:
                continue
            # 通常 rel_t 不应为负；如果你希望保留很小的负值可改阈值
            if rel_t <= 0:
                continue

            # station exists?
            sta_key = f"{net}.{sta}"
            if sta_key not in station_table:
                # 如果你希望允许缺站，改成 continue/或者用 PHASE 行内 DISTAZ 反推不现实，所以这里直接跳过
                continue

            cur.picks.append(Pick(net=net, sta=sta, phase=phase, rel_t=float(rel_t)))

    if cur is not None:
        blocks.append(cur)

    return blocks


def old_parse_pha_files(
    pha_paths: List[str],
    station_table: Dict[str, Tuple[float, float, float]],
    rel_t_max: float = 50.0
) -> List[EventBlock]:
    blocks: List[EventBlock] = []
    for p in pha_paths:
        blocks.extend(parse_pha_file(p, station_table=station_table, rel_t_max=rel_t_max))
    return blocks


# -------------------------
# Build samples (with projection window)
# -------------------------
def blocks_to_samples_from_station_table(
    blocks: List[EventBlock],
    meta_json: str,
    station_table: Dict[str, Tuple[float, float, float]],
    xy_km_limit: float = 1000.0,
    station_z_mode: str = "zero",   # "zero" or "elev"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Returns:
      Xr: (N,3) station xyz in km
      Xs: (N,3) event   xyz in km
      Y : (N,2) [tp, ts] in sec, -1 for missing
    Filtering:
      keep only samples where BOTH station and event projected x,y within [-xy_km_limit, xy_km_limit]
    """
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    lon0 = float(meta["lon0"])
    lat0 = float(meta["lat0"])
    fwd = build_aeqd_fwd(lon0, lat0)

    Xr_list, Xs_list, Y_list = [], [], []

    for ev in blocks:
        ex_m, ey_m = fwd.transform(ev.ev_lon, ev.ev_lat)
        ex_km, ey_km = ex_m / 1000.0, ey_m / 1000.0
        ez_km = float(ev.ev_dep_km)

        if abs(ex_km) > xy_km_limit or abs(ey_km) > xy_km_limit:
            continue  # event outside window

        # 聚合：同一事件-台站可能有多条 P/S；这里保留最小 rel_t（通常对应最早到时）
        # 你也可以改成取平均或只取第一条
        by_sta: Dict[str, Dict[str, float]] = {}
        for pk in ev.picks:
            sta_key = f"{pk.net}.{pk.sta}"
            d = by_sta.get(sta_key, {})
            old = d.get(pk.phase, None)
            if old is None or pk.rel_t < old:
                d[pk.phase] = pk.rel_t
            by_sta[sta_key] = d

        for sta_key, ps in by_sta.items():
            lon_s, lat_s, elev_km = station_table[sta_key]
            sx_m, sy_m = fwd.transform(lon_s, lat_s)
            sx_km, sy_km = sx_m / 1000.0, sy_m / 1000.0

            if abs(sx_km) > xy_km_limit or abs(sy_km) > xy_km_limit:
                continue  # station outside window

            if station_z_mode == "elev":
                sz_km = float(elev_km)
            else:
                sz_km = 0.0

            tp = float(ps["P"]) if "P" in ps else -1.0
            ts = float(ps["S"]) if "S" in ps else -1.0

            Xr_list.append([sx_km, sy_km, sz_km])
            Xs_list.append([ex_km, ey_km, ez_km])
            Y_list.append([tp, ts])

    Xr = np.asarray(Xr_list, dtype=np.float32)
    Xs = np.asarray(Xs_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)

    if Xr.shape[0] == 0:
        raise RuntimeError("No samples built after filtering. Check station_table / pha parsing / xy window.")

    proj_meta = dict(
        projection=str(meta.get("projection", "AEQD")),
        lon0=lon0,
        lat0=lat0,
        xy_km_limit=float(xy_km_limit),
        station_z_mode=station_z_mode,
        rel_t_max="<=50s (applied in parsing)",
    )
    return Xr, Xs, Y, proj_meta

from datetime import datetime

# -------------------------
# Station table: china.loc (units: km)
# -------------------------
def read_china_loc_km(sta_loc_path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Your example:
      AH A0004 40 117.3518  31.7213    -7 AH.A0004.40 # ...
    Here: coordinates are degrees; the last numeric is elevation in km (per your requirement).
    Returns: { "NET.STA": (lon_deg, lat_deg, elev_km) }
    """
    out: Dict[str, Tuple[float, float, float]] = {}
    with open(sta_loc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            net = parts[0]
            sta = parts[1]
            try:
                lon = float(parts[3])
                lat = float(parts[4])
                elev_km = float(parts[5]) / 1000.0  # already km (per your request)
            except Exception:
                continue
            out[f"{net}.{sta}"] = (lon, lat, elev_km)
    return out


# -------------------------
# PHA parsing helpers
# -------------------------
def _to_dt(y: int, mo: int, d: int, hh: int, mm: int, ss: int, usec: int) -> datetime:
    return datetime(y, mo, d, hh, mm, ss, usec)

def _phase_family(token: str) -> Optional[str]:
    """
    Keep only Pg/Pn/Sg/Sn (case-insensitive). Return one of: 'Pg','Pn','Sg','Sn'
    """
    t = token.strip()
    if not t:
        return None
    u = t.upper()
    if u == "PG":
        return "Pg"
    if u == "PN":
        return "Pn"
    if u == "SG":
        return "Sg"
    if u == "SN":
        return "Sn"
    return None


@dataclass
class PhaPick:
    net: str
    sta: str
    phase4: str   # Pg/Pn/Sg/Sn
    rel_t: float  # seconds

@dataclass
class PhaEvent:
    evid: str
    ev_lon: float
    ev_lat: float
    ev_dep_km: float
    origin_dt: datetime
    picks: List[PhaPick]


def parse_pha_files(
    pha_paths: List[str],
    station_table: Dict[str, Tuple[float, float, float]],
    rel_t_max: float = 100.0,
) -> List[PhaEvent]:
    """
    Parse multiple *.pha files:
    - #EVENT ... LOC lon lat DEP dep ... origin time in header
    - PHASE ... TRAVTIME NET STA ... (chan) Pg/Pn/Sg/Sn ... absolute pick time
    - rel_t = pick_dt - origin_dt
    - drop rel_t <= 0 or > rel_t_max
    - drop stations not found in station_table
    """
    events: List[PhaEvent] = []
    cur: Optional[PhaEvent] = None

    def flush():
        nonlocal cur
        if cur is not None:
            events.append(cur)
            cur = None

    for pha_path in pha_paths:
        with open(pha_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue

                if s.startswith("#EVENT"):
                    flush()
                    parts = s.split()
                    # robust indices by keywords
                    # origin time fields (based on your example):
                    # #EVENT <evid> eq YYYY MM DD JDAY HH MM SS USEC LOC lon lat DEP dep ...
                    if len(parts) < 18:
                        cur = None
                        continue

                    evid = parts[1]
                    try:
                        y = int(parts[3]); mo = int(parts[4]); d = int(parts[5])
                        hh = int(parts[7]); mm = int(parts[8]); ss = int(parts[9]); usec = int(parts[10])
                        origin_dt = _to_dt(y, mo, d, hh, mm, ss, usec)
                    except Exception:
                        cur = None
                        continue

                    try:
                        i_loc = parts.index("LOC")
                        ev_lon = float(parts[i_loc + 1])
                        ev_lat = float(parts[i_loc + 2])
                    except Exception:
                        cur = None
                        continue

                    try:
                        i_dep = parts.index("DEP")
                        ev_dep_km = float(parts[i_dep + 1])
                    except Exception:
                        cur = None
                        continue

                    cur = PhaEvent(
                        evid=evid,
                        ev_lon=ev_lon,
                        ev_lat=ev_lat,
                        ev_dep_km=float(ev_dep_km),
                        origin_dt=origin_dt,
                        picks=[]
                    )
                    continue

                if cur is None:
                    continue

                if not s.startswith("PHASE"):
                    continue

                parts = s.split()
                # Must be TRAVTIME
                if len(parts) < 20 or parts[3] != "TRAVTIME":
                    continue

                net = parts[4]
                sta = parts[5]
                phase4 = _phase_family(parts[8])
                if phase4 is None:
                    continue

                # absolute pick time indices (per your sample): 12..19
                try:
                    y = int(parts[12]); mo = int(parts[13]); d = int(parts[14])
                    hh = int(parts[16]); mm = int(parts[17]); ss = int(parts[18]); usec = int(parts[19])
                    pick_dt = _to_dt(y, mo, d, hh, mm, ss, usec)
                except Exception:
                    continue

                rel_t = (pick_dt - cur.origin_dt).total_seconds()
                if rel_t <= 0.0 or rel_t > rel_t_max:
                    continue

                sta_key = f"{net}.{sta}"
                if sta_key not in station_table:
                    continue
                if phase4 in ["Pg", "Sg"]:
                    if np.random.random()<0.15:
                        cur.picks.append(PhaPick(net=net, sta=sta, phase4=phase4, rel_t=float(rel_t)))
                else:
                    cur.picks.append(PhaPick(net=net, sta=sta, phase4=phase4, rel_t=float(rel_t)))
        flush()

    if not events:
        raise RuntimeError("No events parsed from pha files. Check format / paths.")
    return events


# -------------------------
# Dataset + masked loss
# -------------------------

# -------------------------
# Build samples (km window, Pn/Sn preferred)
# -------------------------
def events_to_samples(
    events: List[PhaEvent],
    meta_json: str,
    station_table: Dict[str, Tuple[float, float, float]],
    xy_km_limit: float = 1000.0,
    use_station_elev_as_z: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Outputs:
      Xr (N,3) station xyz in km
      Xs (N,3) event   xyz in km
      Y  (N,4) travel times in sec: [Pg, Sg, Pn, Sn], missing = -1

    Rules:
      - project lon/lat to AEQD; meters -> km
      - keep only if BOTH station and event in window |x|,|y|<=xy_km_limit
      - for each event-station & each phase in {Pg,Pn,Sg,Sn}, if multiple picks exist: take minimum rel_t
    """
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    lon0 = float(meta["lon0"])
    lat0 = float(meta["lat0"])
    fwd = build_aeqd_fwd(lon0, lat0)

    Xr_list, Xs_list, Y_list = [], [], []

    phase_order = ["Pg", "Sg", "Pn", "Sn"]

    for ev in events:
        ex_m, ey_m = fwd.transform(ev.ev_lon, ev.ev_lat)
        ex_km, ey_km = ex_m / 1000.0, ey_m / 1000.0
        ez_km = float(ev.ev_dep_km)

        if abs(ex_km) > xy_km_limit or abs(ey_km) > xy_km_limit:
            continue

        # aggregate: sta -> phase4 -> min rel_t
        by_sta: Dict[str, Dict[str, float]] = {}
        for pk in ev.picks:
            sta_key = f"{pk.net}.{pk.sta}"
            d = by_sta.get(sta_key, {})
            old = d.get(pk.phase4, None)
            if old is None:
                d[pk.phase4] = pk.rel_t
            by_sta[sta_key] = d

        for sta_key, d in by_sta.items():
            lon_s, lat_s, elev_km = station_table[sta_key]
            sx_m, sy_m = fwd.transform(lon_s, lat_s)
            sx_km, sy_km = sx_m / 1000.0, sy_m / 1000.0
            if abs(sx_km) > xy_km_limit or abs(sy_km) > xy_km_limit:
                continue

            if use_station_elev_as_z:
                zr_km = -float(elev_km)  # 你要求台站高程为负
            else:
                zr_km = 0.0

            y4 = [float(d.get(ph, -1.0)) for ph in phase_order]  # [Pg,Sg,Pn,Sn]
            
            Xr_list.append([sx_km, sy_km, zr_km])
            Xs_list.append([ex_km, ey_km, ez_km])
            Y_list.append(y4)

    Xr = np.asarray(Xr_list, dtype=np.float32)
    Xs = np.asarray(Xs_list, dtype=np.float32)
    Y  = np.asarray(Y_list, dtype=np.float32)

    if Xr.shape[0] == 0:
        raise RuntimeError("No samples built after windowing. Check lon0/lat0, window, station table.")

    proj_meta = dict(
        projection=str(meta.get("projection", "AEQD")),
        lon0=lon0,
        lat0=lat0,
        xy_km_limit=float(xy_km_limit),
        station_z_mode="elev(negative)" if use_station_elev_as_z else "zero",
        y_columns="Pg,Sg,Pn,Sn",
        rel_t_rule="rel_t=pick-origin ; keep 0<rel_t<=rel_t_max",
    )
    return Xr, Xs, Y, proj_meta



# -------------------------
# Dataset
# -------------------------
class TravelTimeDataset(Dataset):
    def __init__(self, Xr: np.ndarray, Xs: np.ndarray, Y: np.ndarray):
        assert Xr.shape[0] == Xs.shape[0] == Y.shape[0]
        self.Xr = torch.from_numpy(Xr)  # (N,3)
        self.Xs = torch.from_numpy(Xs)  # (N,3)
        self.Y = torch.from_numpy(Y)  # (N,4), -1 for missing
    def __len__(self):
        return self.Xr.shape[0]

    def __getitem__(self, idx: int):
        return self.Xr[idx], self.Xs[idx], self.Y[idx]


# -------------------------
# One-shot builder
# -------------------------
def build_dataset_from_pha(
    pha_paths: List[str],
    station_loc_path: str,
    meta_json: str,
    xy_km_limit: float = 1000.0,
    rel_t_max: float = 50.0,
    station_z_mode: str = "zero",
) -> Tuple[TravelTimeDataset, Dict, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    station_table = read_china_loc(station_loc_path)
    blocks = parse_pha_files(pha_paths, station_table=station_table, rel_t_max=rel_t_max)
    Xr, Xs, Y, proj_meta = blocks_to_samples_from_station_table(
        blocks, meta_json=meta_json, station_table=station_table,
        xy_km_limit=xy_km_limit, station_z_mode=station_z_mode
    )
    ds = TravelTimeDataset(Xr, Xs, Y)
    meta = {"proj": proj_meta, "n_events": len(blocks), "n_samples": int(Xr.shape[0])}
    return ds, meta, (Xr, Xs, Y)



def masked_mse_old(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target) & torch.isfinite(pred)
    if mask.sum().item() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    diff2 = (pred - target) ** 2
    return diff2[mask].mean()

def masked_mse(pred: torch.Tensor, target: torch.Tensor, tmax: float = 1000.0) -> torch.Tensor:
    mask = (target >= 0.0) & (target <= tmax) & torch.isfinite(target) & torch.isfinite(pred)
    if mask.sum().item() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    diff2 = (pred - target) ** 2
    return diff2[mask].mean()


# -------------------------
# Checkpoint IO
# -------------------------
def save_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    args_dict: Dict,
    extra_meta: Dict,
):
    ckpt = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "model_class": model.__class__.__name__,
        "model_hidden_dim": getattr(model, "hidden_dim", None),
        "args": args_dict,
        "meta": extra_meta,
    }
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Tuple[int, Dict]:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = int(ckpt.get("epoch", 0))
    meta = {"args": ckpt.get("args", {}), "meta": ckpt.get("meta", {})}
    return start_epoch, meta


# -------------------------
# Training loop
# -------------------------
def train(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    save_ckpt: str = "",
    save_every: int = 1,
    resume_ckpt: str = "",
    run_args: Dict = None,
    meta: Dict = None,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    start_epoch = 0
    if resume_ckpt:
        start_epoch, resume_meta = load_checkpoint(resume_ckpt, model, opt, map_location=device, strict=True)
        print(f"[OK] resumed from {resume_ckpt} (epoch={start_epoch})")
        # keep resume meta for reference
        if meta is not None:
            meta = {**resume_meta.get("meta", {}), **meta}

    run_args = run_args or {}
    meta = meta or {}
    steps = 0
    for ep in range(start_epoch + 1, epochs + 1):
        model.train()
        losses = []
        for xr, xs, y in loader:
            xr = xr.to(device)
            xs = xs.to(device)
            y = y.to(device)
            
            pred = model(xr, xs)
            #print("[EPOCH]", ep, "[STEPS]", steps)
            if not torch.isfinite(pred).all():
                bad = (~torch.isfinite(pred)).nonzero(as_tuple=False)
                print("[FATAL] pred has NaN/inf, first bad index:", bad[0].tolist())
                i, j = bad[0].tolist()
                print("pred[i] =", pred[i].detach().cpu().numpy())
                print("target[i] =", y[i].detach().cpu().numpy())
                print("xr[i] =", xr[i].detach().cpu().numpy())
                print("xs[i] =", xs[i].detach().cpu().numpy())
                raise RuntimeError("pred non-finite")

            loss = masked_mse(pred, y)
            #print(xr, xs, y, pred)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))
            steps += 1
            if steps % 1000 == 0:
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                print(f"[EPOCH {ep:03d}] step={steps} loss={loss:.6f} batches={len(losses)}")

                if save_ckpt and (ep % max(1, save_every) == 0):
                    save_checkpoint(
                        save_ckpt,
                        model=model,
                        optimizer=opt,
                        epoch=ep,
                        args_dict=run_args,
                        extra_meta=meta,
                    )
                    print(f"[OK] saved checkpoint: {save_ckpt}")
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[EPOCH {ep:03d}] loss={mean_loss:.6f} batches={len(losses)}")

        if save_ckpt and (ep % max(1, save_every) == 0):
            save_checkpoint(
                save_ckpt,
                model=model,
                optimizer=opt,
                epoch=ep,
                args_dict=run_args,
                extra_meta=meta,
            )
            print(f"[OK] saved checkpoint: {save_ckpt}")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument("--real_txt", default="run_fm3d/data/train.synth_arrivals_skfmm_noise.new.txt", help="real.txt path")
    ap.add_argument("--meta_json", default="run_fm3d/data/xyz_vp_vs_meta.json", help="meta.json path (contains lon0/lat0)")

    ap.add_argument("--pha_glob", default="data/pha_train/*.pha", help="glob pattern for pha files, e.g. data/2009.pha")
    ap.add_argument("--station_loc", default="data/china.loc", help="station file path (china.loc)")
    ap.add_argument("--xy_km_limit", type=float, default=1000.0)
    ap.add_argument("--rel_t_max", type=float, default=500.0)
    ap.add_argument("--use_station_elev_as_z", action="store_true", help="use station elev (forced negative) as zr")

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--save_ckpt", default="ckpt/time.real.pnsn.switch.v1.0.pt", help="checkpoint path to save (e.g., pinn_tt.ckpt.pt)")
    ap.add_argument("--save_every", type=int, default=1, help="save every N epochs")
    ap.add_argument("--resume", default="", help="checkpoint path to resume/load")
    ap.add_argument("--eval_only", action="store_true", help="load checkpoint and run sanity inference only")
    args = ap.parse_args()
    # -------------------------
    # Build dataset from PHA
    # -------------------------
    args.use_station_elev_as_z = True 
    pha_paths = sorted([str(p) for p in Path(".").glob(args.pha_glob)])
    if not pha_paths:
        raise RuntimeError(f"No pha files matched: {args.pha_glob}")
    print(f"[OK] pha files: {len(pha_paths)}")

    station_table = read_china_loc_km(args.station_loc)
    print(f"[OK] stations loaded: {len(station_table)}")

    events = parse_pha_files(
        pha_paths=pha_paths,
        station_table=station_table,
        rel_t_max=args.rel_t_max,
    )
    print(f"[OK] parsed events: {len(events)}")
    print(events[5999])
    Xr, Xs, Y, proj_meta = events_to_samples(
        events=events,
        meta_json=args.meta_json,
        station_table=station_table,
        xy_km_limit=args.xy_km_limit,
        use_station_elev_as_z=args.use_station_elev_as_z,
    )
    print(f"[OK] samples: {Y.shape}")
    print(
        "[INFO] missing ratio: "
        f"Pg={(Y[:,0] < 0).mean():.3f}, "
        f"Sg={(Y[:,1] < 0).mean():.3f}, "
        f"Pn={(Y[:,2] < 0).mean():.3f}, "
        f"Sn={(Y[:,3] < 0).mean():.3f}"
    )

    
    Xr2 = np.concatenate([Xr, Xs], axis=0)
    Xs2 = np.concatenate([Xs, Xr], axis=0)
    Y2  = np.concatenate([Y, Y], axis=0)
    ds = TravelTimeDataset(Xr2, Xs2, Y2)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=False)

    device = "mps" #if torch.cuda.is_available() else "cpu"
    model = PINNTravelTime(hidden_dim=args.hidden_dim)

    # If eval_only: must have resume
    if args.eval_only:
        if not args.resume:
            raise RuntimeError("--eval_only requires --resume <ckpt_path>")
        _ = load_checkpoint(args.resume, model, optimizer=None, map_location=device, strict=True)
        model.to(device)
        model.eval()
        with torch.no_grad():
            xr, xs, y = next(iter(dl))
            pred = model(xr.to(device), xs.to(device)).cpu().numpy()
            print("[EVAL] pred[0:10] (Pg,Sg,Pn,Sn):\n", pred[:10])
            print("[EVAL] true[0:10] (Pg,Sg,Pn,Sn), -1 means missing:\n", y.numpy()[:10])

        return

    # Train (supports resume)
    run_args = vars(args).copy()
    meta = {"projection_meta": proj_meta}

    train(
        model=model,
        loader=dl,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_ckpt=args.save_ckpt,
        save_every=args.save_every,
        resume_ckpt=args.resume,
        run_args=run_args,
        meta=meta,
    )

    # Final sanity
    model.eval()
    with torch.no_grad():
        xr, xs, y = next(iter(dl))
        pred = model(xr.to(device), xs.to(device)).cpu().numpy()
        print("[SANITY] pred[0:5] (tp, ts):\n", pred[:5])
        print("[SANITY] true[0:5] (tp, ts):\n", y.numpy()[:5])


if __name__ == "__main__":
    main()
