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
STEPSIZE0 = 0.18
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
def parse_real_arrivals(path: str) -> Dict[int, List[Tuple[str, str, float, float]]]:
    """
    Parse REAL-like arrivals file:
      <event header line>
        <pick line>
        <pick line>
      <event header line>
        ...

    Returns:
      events_picks[eid] = [(sta, phase("P"/"S"), t_epoch, sigma), ...]
    """
    events_picks: Dict[int, List[Tuple[str, str, float, float]]] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        current_eid = None
        current_year = current_mon = current_day = 0
        current_hh = current_mm = 0
        current_ss = 0.0  # seconds with fraction, from header time

        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            # Event header lines typically start with event id in col 0 (not indented)
            # Pick lines often start with spaces then NET
            if line[0].isdigit():
                parts = line.split()
                # example header:
                # 783 2022 09 05 00:10:03.000  8430.000  ... lat lon depth ...
                # robust parse: eid, yyyy, mm, dd, hh:mm:ss.xxx
                try:
                    eid = int(parts[0])
                    yyyy = int(parts[1]); mo = int(parts[2]); dd = int(parts[3])
                    hms = parts[4]  # "00:10:03.000"
                    hh, mm, ss = hms.split(":")
                    hh = int(hh); mm = int(mm); ss = float(ss)

                    current_eid = eid
                    current_year, current_mon, current_day = yyyy, mo, dd
                    current_hh, current_mm, current_ss = hh, mm, ss

                    if current_eid not in events_picks:
                        events_picks[current_eid] = []

                except Exception:
                    # if header parse fails, skip line
                    current_eid = None
                    continue

            else:
                # pick line
                if current_eid is None:
                    continue

                parts = line.split()
                # example pick:
                # SC L09971B02373  P  8461.1334  31.1334 ... 0.300 ... lon lat
                # indices:
                # 0=NET, 1=STA, 2=PHASE, 3=ABS_TIME(sec), 4=REL_TIME(sec), ... , 6=..., 7=..., 8=SIGMA? (here 0.300)
                if len(parts) < 9:
                    continue

                net = parts[0]
                sta = parts[1]
                ph_raw = parts[2].upper()

                if ph_raw.startswith("P"):
                    ph = "P"
                elif ph_raw.startswith("S"):
                    ph = "S"
                else:
                    continue

                # Absolute time is given as "8461.1334" (seconds since some reference)
                # BUT you want epoch seconds; easiest consistent method:
                # Use header time as epoch anchor, and REL_TIME as pick offset (recommended).
                #
                # REL_TIME is parts[4] (e.g., 31.1334) = pick_time - origin_time
                try:
                    rel = float(parts[4])
                except Exception:
                    continue

                # sigma is usually the column with 0.300 / 0.400 in your example.
                # In your shown line, it's the 8th element (index 7?) depending on spacing.
                # For your sample: "... 5.104/2.993    0.300   200.0000 ..."
                # That 0.300 is parts[7].
                sigma = 0.3
                try:
                    sigma = float(parts[7])
                except Exception:
                    # fallback: try last numeric in first 12 tokens
                    for tok in parts[:12]:
                        try:
                            val = float(tok)
                            # heuristic: sigma usually in [0.01, 5]
                            if 0.001 <= val <= 5.0:
                                sigma = val
                        except Exception:
                            pass

                # Build origin epoch from header yyyy-mm-dd hh:mm:ss.xxx
                origin_dt = datetime(
                    current_year, current_mon, current_day,
                    current_hh, current_mm, int(current_ss),
                    int(round((current_ss - int(current_ss)) * 1e6))
                )
                origin_epoch = origin_dt.timestamp()

                pick_epoch = origin_epoch + rel

                events_picks[current_eid].append((sta, ph, pick_epoch, sigma))

    # sort picks by time within each event
    for eid in list(events_picks.keys()):
        events_picks[eid].sort(key=lambda x: x[2])
        if len(events_picks[eid]) == 0:
            del events_picks[eid]

    return events_picks


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
        out = travel_time(xs[pi], xr[si])
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

    if len(xr_list) < 2:
        raise ValueError("too few usable picks after station filtering")

    xr = torch.tensor(xr_list, device=device, dtype=dtype)          # [M,3]
    phases_np = np.array(ph_list, dtype=str)  # 原来你就有
    phase_is_p = torch.tensor(phases_np == "P", device=device, dtype=torch.bool)  # [M]

    phases = np.array(phases_np, dtype=str)                         # ensure "P"/"S"
    t_obs = torch.tensor(t_list, device=device, dtype=dtype)        # [M] (epoch seconds)
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

    return {
        "lon_med": float(lon),
        "lat_med": float(lat),
        "depth_med_km": float(x_med[2]),

        "lonlat_mad_like": None,  # (optional) not defined on sphere; we keep xy/depth MAD below

        "x_med_km": x_med.tolist(),          # [x_km,y_km,z_km] in AEQD km
        "x_mad_km": x_mad.tolist(),          # MAD in km, same coordinate system

        "t0_med_epoch": float(t0_med.item()),
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
    CKPT = "ckpt/time.v3.pt"

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
    events_picks = parse_real_arrivals(REAL_TXT)

    for eid, picks in events_picks.items():
        try:
            #print(picks)
            out = locate_event(travel_time, picks, station_xyz, device, dtype)
            results[eid] = out
            print(
                eid,
                f"lon={out['lon_med']:.4f} lat={out['lat_med']:.4f} dep={out['depth_med_km']:.2f}km",
                f"xyMAD=({out['x_mad_km'][0]:.2f},{out['x_mad_km'][1]:.2f})km zMAD={out['x_mad_km'][2]:.2f}km",
                f"t0MAD={out['t0_mad_sec']:.3f}s"
            )
        except Exception as e:
            results[eid] = {"error": str(e)}
            print(eid, "FAILED:", e)

    with open("location_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
