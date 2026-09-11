#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from pyproj import CRS, Transformer


# =========================
# 0) Global config (unchanged)
# =========================
LON0 = 102.5
LAT0 = 27.5

DEPTH_RANGE_KM = (0.0, 80.0)
NUM_PARTICLES = 128
NUM_ITERS = 250
MAX_PAIRS = 6000
STEPSIZE0 = 0.1
STEPSIZE_DECAY = 0.985
PINN_BATCH = 65536


# =========================
# 1) AEQD projection (unchanged)
# =========================
def build_inverse_aeqd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(crs_aeqd, CRS.from_epsg(4326), always_xy=True)

def build_forward_aeqd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(CRS.from_epsg(4326), crs_aeqd, always_xy=True)

proj_fwd = build_forward_aeqd(LON0, LAT0)
proj_inv = build_inverse_aeqd(LON0, LAT0)


# =========================
# 2) Time utils (unchanged)
# =========================
def parse_dt_to_epoch_seconds(dt_str: str) -> float:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()


# =========================
# 3) Station file (unchanged)
# =========================
def read_station_file(path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    SC L09985B02373  99.8537  23.7290  -0.000
    -> station_xyz_km[sta] = (x_km, y_km, z_km)
    """
    station_xyz = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            net, sta, lat, lon, z = line.split()[:5]
            #print(net, sta, lon, lat, z)
            x_m, y_m = proj_fwd.transform(float(lon), float(lat))
            #print(x_m, y_m)
            station_xyz[sta] = (x_m / 1000.0, y_m / 1000.0, float(z))
    return station_xyz


# =========================
# 4) Event JSON (unchanged)
# =========================
def load_events(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_picks(event_obj: Dict[str, Any]):
    picks = event_obj["Picks"]
    out = []
    for k in picks["DT"]:
        sta = picks["Station"][k]
        ph = picks["PhasePick"][k].upper()
        if ph.startswith("P"):
            ph = "P"
        elif ph.startswith("S"):
            ph = "S"
        else:
            continue
        t = parse_dt_to_epoch_seconds(picks["DT"][k])
        sigma = float(picks["PickError"].get(k, 0.3))
        out.append((sta, ph, t, sigma))
    return out

# =========================
# 5) Pair sampling (unchanged)
# =========================
def sample_pairs(M: int, max_pairs: int, rng):
    total = M * (M - 1) // 2
    if total <= max_pairs:
        return np.array([(i, j) for i in range(M) for j in range(i+1, M)], dtype=np.int64)
    pairs = set()
    while len(pairs) < max_pairs:
        i, j = rng.integers(0, M, size=2)
        if i != j:
            pairs.add(tuple(sorted((int(i), int(j)))))
    return np.array(list(pairs), dtype=np.int64)
import re 
def parse_real_arrivals(path: str) -> Dict[int, Dict[str, Any]]:
    """
    Parse REAL-like arrivals file:
      <event header line>
        <pick line>
        ...

    Returns:
      events[eid] = {
         "event_id": eid,
         "header": {"lon":..., "lat":..., "dep_km":...}  # if present
         "picks": [(sta, phase("P"/"S"), t_epoch, sigma), ...]
      }
    """
    events: Dict[int, Dict[str, Any]] = {}
    HEADER_RE = re.compile(
            r"^\s*(\d+)\s+(\d{4})\s+(\d{2})\s+(\d{2})\s+(\d{2}:\d{2}:\d{2}\.\d+)\s+([+-]?\d+(?:\.\d+)?)\s+"
        )
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        current_eid = None
        current_year = current_mon = current_day = 0
        current_hh = current_mm = 0
        current_ss = 0.0

        
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            m = HEADER_RE.match(line)
            # ---- event header ----
            if m:
                parts = line.split()
                if True:
                    eid = int(parts[0])
                    yyyy = int(parts[1]); mo = int(parts[2]); dd = int(parts[3])
                    hms = parts[4]  # "00:10:03.000"
                    hh, mm, ss = hms.split(":")
                    hh = int(hh); mm = int(mm); ss = float(ss)

                    current_eid = eid
                    current_year, current_mon, current_day = yyyy, mo, dd
                    current_hh, current_mm, current_ss = hh, mm, ss

                    if current_eid not in events:
                        events[current_eid] = {
                            "event_id": current_eid,
                            "header": {"lon": None, "lat": None, "dep_km": None},
                            "picks": []
                        }

                    # ---- try parse lon/lat/dep from the header line ----
                    # 现实中不同 REAL 输出列可能不完全一致，这里用“就近扫描”的稳健策略：
                    # 1) 在 header tokens 中找连续 3 个 float，且满足 lon/lat 合理范围、dep>=0
                    # 2) 优先选择 dep<=200km 的组
                    floats = []
                    #print(parts)
                    elon, elat, edep = float(parts[8]), float(parts[7]), float(parts[9])

                    if True:
                        events[current_eid]["header"]["lon"] = float(elon)
                        events[current_eid]["header"]["lat"] = float(elat)
                        events[current_eid]["header"]["dep_km"] = float(edep)

                else:
                    current_eid = None
                    continue

            # ---- pick line ----
            else:
                if current_eid is None:
                    continue

                parts = line.split()
                if len(parts) < 9:
                    continue

                sta = parts[1]
                ph_raw = parts[2].upper()
                if ph_raw.startswith("P"):
                    ph = "P"
                elif ph_raw.startswith("S"):
                    ph = "S"
                else:
                    continue

                # REL time (sec)
                try:
                    rel = float(parts[4])
                except Exception:
                    continue

                # sigma
                sigma = 0.3
                try:
                    sigma = float(parts[7])
                except Exception:
                    for tok in parts[:12]:
                        try:
                            val = float(tok)
                            if 0.001 <= val <= 5.0:
                                sigma = val
                        except Exception:
                            pass

                origin_dt = datetime(
                    current_year, current_mon, current_day,
                    current_hh, current_mm, int(current_ss),
                    int(round((current_ss - int(current_ss)) * 1e6))
                )
                origin_epoch = origin_dt.timestamp()
                pick_epoch = origin_epoch + rel

                events[current_eid]["picks"].append((sta, ph, pick_epoch, sigma))

    # sort and prune
    for eid in list(events.keys()):
        events[eid]["picks"].sort(key=lambda x: x[2])
        if len(events[eid]["picks"]) == 0:
            del events[eid]

    return events

# =========================
# 6) SVGD kernel (unchanged)
# =========================
def svgd_update(xs, grad, stepsize):
    N = xs.shape[0]
    diff = xs[:, None, :] - xs[None, :, :]
    dist2 = (diff ** 2).sum(-1)
    h = torch.median(dist2[dist2 > 0]) / math.log(N + 1.0)
    h = torch.clamp(h, min=1e-6)
    k = torch.exp(-dist2 / h)
    term1 = (k @ grad) / N
    term2 = -(2 / h) * ((k.sum(0)[:, None] * xs - k.T @ xs) / N)
    return xs + stepsize * (term1 + term2)


# =========================
# 7) Log-posterior + grad (unchanged)
# =========================
def logp_and_grad(travel_time, xs, xr, phase_is_p, t_obs, sigma, pairs, device, dtype):

    xs = xs.clone().detach().requires_grad_(True)
    N, M = xs.shape[0], xr.shape[0]

    Tp = torch.empty((N, M), device=device, dtype=dtype)
    Ts = torch.empty((N, M), device=device, dtype=dtype)

    K = N * M
    for k0 in range(0, K, PINN_BATCH):
        k1 = min(K, k0 + PINN_BATCH)
        kk = torch.arange(k0, k1, device=device)
        pi = kk // M
        si = kk % M
        out = travel_time(xr[si], xs[pi])   # receiver=xr, source=xs

        Tp[pi, si] = out[:, 0]
        Ts[pi, si] = out[:, 1]

    tt = torch.where(phase_is_p[None, :], Tp, Ts)


    a, b = pairs[:, 0], pairs[:, 1]
    r = (t_obs[a] - t_obs[b])[None, :] - (tt[:, a] - tt[:, b])
    sig = torch.sqrt(sigma[a]**2 + sigma[b]**2)[None, :]
    ll = -torch.abs(r) / sig - torch.log(sig)
    logp = ll.sum(1)

    grad = torch.autograd.grad(logp.sum(), xs)[0]
    return logp.detach(), grad.detach()


# =========================
# 8) Locate one event (ONLY ADD t0 back-substitution + MAD)
# =========================
def locate_event(travel_time, picks, station_xyz, device, dtype):
    xr_list, ph_list, t_list, s_list = [], [], [], []
    for sta, ph, t, s in picks:
        if sta not in station_xyz:
            continue
        xr_list.append(station_xyz[sta])
        ph_list.append(ph)
        t_list.append(t)
        s_list.append(s)

    if len(xr_list) < 4:
        raise ValueError("too few picks")

    xr = torch.tensor(xr_list, device=device, dtype=dtype)          # [M,3]
    phases_np = np.array(ph_list, dtype=str)  # 原来你就有
    phase_is_p = torch.tensor(phases_np == "P", device=device, dtype=torch.bool)  # [M]

    phases = np.array(phases_np, dtype=str)                         # ensure "P"/"S"
    t_obs = torch.tensor(t_list, device=device, dtype=dtype)        # [M] (epoch seconds)
    t_obs = t_obs - t_obs.min()   # 关键：把 epoch 秒挪到 0 附近
    sigma = torch.tensor(s_list, device=device, dtype=dtype)        # [M]

    rng = np.random.default_rng(0)
    pairs = torch.tensor(sample_pairs(len(xr), MAX_PAIRS, rng), device=device)  # [K,2]

    # prior from station extent (unchanged)
    xmin, xmax = xr[:, 0].min()-50, xr[:, 0].max()+50
    ymin, ymax = xr[:, 1].min()-50, xr[:, 1].max()+50
    zmin, zmax = DEPTH_RANGE_KM
    #print(xr, xmin, xmax, ymin, ymax, zmin, zmax)
    xs = torch.empty((NUM_PARTICLES, 3), device=device, dtype=dtype)
    xs[:, 0] = torch.rand(NUM_PARTICLES, device=device, dtype=dtype) * (xmax-xmin) + xmin
    xs[:, 1] = torch.rand(NUM_PARTICLES, device=device, dtype=dtype) * (ymax-ymin) + ymin
    xs[:, 2] = torch.rand(NUM_PARTICLES, device=device, dtype=dtype) * (zmax-zmin) + zmin
    #print(xs)
    stepsize = STEPSIZE0
    for _ in range(NUM_ITERS):
        _, grad = logp_and_grad(travel_time, xs, xr, phase_is_p, t_obs, sigma, pairs, device, dtype)


        xs = svgd_update(xs, grad, stepsize)
        stepsize *= STEPSIZE_DECAY

    # --- location point estimate + MAD (same style as t0: median + MAD) ---
    x_med_t = xs.median(0).values                          # [3]
    x_mad_t = (xs - x_med_t[None, :]).abs().median(0).values  # [3]

    x_med = x_med_t.detach().cpu().numpy()
    x_mad = x_mad_t.detach().cpu().numpy()

    # convert back to lon/lat
    lon, lat = proj_inv.transform(float(x_med[0]*1000.0), float(x_med[1]*1000.0))
    #print(lon, lat)
    # --- t0 back-substitution + MAD (paper style: median + MAD) ---
    # t0_i = t_obs(i) - tt_pred(i)
    with torch.no_grad():
        M = xr.shape[0]
        xs_rep = x_med_t.view(1, 3).repeat(M, 1)  # [M,3]
        out = travel_time(xr, xs_rep)             # [M,2]
        Tp = out[:, 0]
        Ts = out[:, 1]

        # FIX: condition must be a torch.BoolTensor on the same device
        phase_is_p = torch.tensor((phases_np == "P"), device=device, dtype=torch.bool)  # [M]
        tt = torch.where(phase_is_p, Tp, Ts)

        t0_candidates = t_obs - tt
        t0_med = t0_candidates.median()
        t0_mad = (t0_candidates - t0_med).abs().median()
    t_ref = float(min(t_list))
    return {
        "lon_med": float(lon),
        "lat_med": float(lat),
        "depth_med_km": float(x_med[2]),

        "lonlat_mad_like": None,  # (optional) not defined on sphere; we keep xy/depth MAD below

        "x_med_km": x_med.tolist(),          # [x_km,y_km,z_km] in AEQD km
        "x_mad_km": x_mad.tolist(),          # MAD in km, same coordinate system

        "t0_med_epoch": t_ref + float(t0_med.item()),
        "t0_mad_sec": float(t0_mad.item()),

        "num_picks_used": int(len(xr_list)),
        "num_pairs": int(pairs.shape[0]),
    }

import torch.nn as nn
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
        )

    def forward(self, x, xs):
        # x : (B,3) receiver [xr,yr,zr]  (km)
        # xs: (B,3) source   [xe,ye,ze]  (km)
        inp = torch.cat([x, xs.expand_as(x)], dim=-1)  # (B,6)
        inp = inp / 1000.0
        out = self.net_merge(inp)
        out = out.sigmoid() * 1000.0  # seconds
        return out

# =========================
# 9) Run all events
# =========================
if __name__ == "__main__":
    EVENT_JSON = "run_fm3d/data/svi/event_picks.json"
    STATION_TXT = "run_fm3d/data/svi/stations.csv"
    CKPT = "ckpt/time.v3.1.eikonal.pt"

    dtype = torch.float32
    device = torch.device("mps")  # cpu/cuda/mps

    # ---- load PINN model (as you provided) ----
    travel_time = PINNTravelTime()
    travel_time.eval()
    travel_time.to(device)

    state = torch.load(CKPT, map_location="cpu")["model_state"]
    travel_time.load_state_dict(state)
    travel_time.to(device=device, dtype=dtype)
    travel_time.eval()

    station_xyz = read_station_file(STATION_TXT)
    events = load_events(EVENT_JSON)

    results = {}
    REAL_TXT = "run_fm3d/data/loc.synth_arrivals_skfmm_noise.new.txt"
    events_real = parse_real_arrivals(REAL_TXT)

    for eid, pack in events_real.items():
        picks = pack["picks"]
        hdr = pack["header"]   # {"lon","lat","dep_km"}
        try:
            out = locate_event(travel_time, picks, station_xyz, device, dtype)

            # 保存 event_id + REAL header（方便后处理对比）
            out["event_id"] = int(eid)
            out["real_lon"] = hdr.get("lon", None)
            out["real_lat"] = hdr.get("lat", None)
            out["real_dep_km"] = hdr.get("dep_km", None)

            results[eid] = out

            # 打印对比：REAL vs SVI
            real_str = "REAL lon/lat/dep=NA"
            if (out["real_lon"] is not None) and (out["real_lat"] is not None) and (out["real_dep_km"] is not None):
                real_str = f"REAL lon={out['real_lon']:.4f} lat={out['real_lat']:.4f} dep={out['real_dep_km']:.2f}km"

            print(
                f"{eid}  "
                f"{real_str}  |  "
                f"SVI lon={out['lon_med']:.4f} lat={out['lat_med']:.4f} dep={out['depth_med_km']:.2f}km  "
                f"xyMAD=({out['x_mad_km'][0]:.2f},{out['x_mad_km'][1]:.2f})km zMAD={out['x_mad_km'][2]:.2f}km  "
                f"t0MAD={out['t0_mad_sec']:.3f}s"
            )

        except Exception as e:
            results[eid] = {"event_id": int(eid), "error": str(e)}
            print(eid, "FAILED:", e)


    with open("run_fm3d/data/svi/out/location_results_v1.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
