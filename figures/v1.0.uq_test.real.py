#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from pyproj import CRS, Transformer


# =========================
# User config
# =========================
REAL_TXT  = Path("run_fm3d/data/loc.synth_arrivals_skfmm_noise.new.txt")
NLLOC_CSV = Path("run_fm3d/data/nlloc_sc/all.locfiles.csv")
OURS_TXT  = Path("run_fm3d/data/ours/reloc.real.True.v1.0.txt")

CKPT_PATH = Path("ckpt/time.v1.0.pt")

OUT_DIR = Path("figures/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# region bbox (edit if needed)
LON_RANGE = (97.0, 108.0)
LAT_RANGE = (20.0, 35.0)

# AEQD origin
LON0, LAT0 = 102.5, 27.5

# If err0/err1/err2 in ours are meters -> set True
OURS_ERR_IN_METERS = False  # change to True if your err are meters

# For PPC travel-time parsing: which column is "travel time" in pick lines?
# Based on your earlier example: "... P 8461.1334 31.1334 ..."
# travel time = parts[4] (0-based) => the 5th token.
PICK_TTIME_COL = 4


# =========================
# Utilities
# =========================
def _in_bbox(lon: float, lat: float) -> bool:
    return (LON_RANGE[0] <= lon <= LON_RANGE[1]) and (LAT_RANGE[0] <= lat <= LAT_RANGE[1])

def _wrap360(deg: float) -> float:
    deg = deg % 360.0
    return deg + 360.0 if deg < 0 else deg

def azimuth_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Forward azimuth from (lon1,lat1) to (lon2,lat2), degrees in [0,360)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    az = math.degrees(math.atan2(x, y))
    return _wrap360(az)

def max_azimuth_gap(az_list: List[float]) -> float:
    """Maximum azimuthal gap (deg) for station azimuths around event."""
    if len(az_list) < 2:
        return 360.0
    az = np.sort(np.array([_wrap360(a) for a in az_list], dtype=float))
    gaps = np.diff(az)
    wrap_gap = (az[0] + 360.0) - az[-1]
    return float(np.max(np.append(gaps, wrap_gap)))

def haversine_km(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# =========================
# AEQD projection (lon/lat -> x/y meters)
# =========================
def build_forward_aeqd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(CRS.from_epsg(4326), crs_aeqd, always_xy=True)

TF_LL2XY = build_forward_aeqd(LON0, LAT0)

def lonlat_to_xy_m(lon: float, lat: float) -> Tuple[float, float]:
    x, y = TF_LL2XY.transform(lon, lat)
    return float(x), float(y)


# =========================
# Data models
# =========================
@dataclass
class RealEventObs:
    evid: str
    lon: float
    lat: float
    dep_km: float
    nsta: int
    gap_deg: float
    mean_dist_km: float

@dataclass
class LocSolution:
    evid: str
    lon: float
    lat: float
    dep_km: float
    h_err_km: float
    z_err_km: float
    # optional for better coverage (ours has components)
    ex_km: Optional[float] = None
    ey_km: Optional[float] = None
    ez_km: Optional[float] = None

@dataclass
class TrueLoc:
    lon: float
    lat: float
    dep_km: float

@dataclass
class Pick:
    evid: str
    net: str
    sta: str
    phase: str     # "P" or "S"
    t_obs: float   # observed travel time (sec)
    slon: float
    slat: float


# =========================
# 1) REAL: parse event headers + picks (for nsta/gap/mean distance and PPC)
# =========================
def parse_real_events_and_picks(path: Path) -> Tuple[Dict[str, TrueLoc], List[Pick], Dict[str, Dict[str, Tuple[float, float]]]]:
    """
    Returns:
      - evloc: dict evid -> (lon,lat,dep_km) from REAL header
      - picks: list of Pick (with phase and travel time)
      - stas_by_evid: evid -> {NET.STA: (slon, slat)} for geometry metrics
    """
    evloc: Dict[str, TrueLoc] = {}
    picks: List[Pick] = []
    stas_by_evid: Dict[str, Dict[str, Tuple[float, float]]] = {}

    cur_evid: Optional[str] = None

    def is_event_header(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        tok = s.split()[0]
        return tok.isdigit()

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if is_event_header(line):
                parts = line.split()
                cur_evid = str(parts[0])
                try:
                    lat = float(parts[7])
                    lon = float(parts[8])
                    dep = float(parts[9])
                except Exception:
                    cur_evid = None
                    continue
                if _in_bbox(lon, lat):
                    evloc[cur_evid] = TrueLoc(lon=lon, lat=lat, dep_km=dep)
                    stas_by_evid.setdefault(cur_evid, {})
                continue

            if cur_evid is None or cur_evid not in evloc:
                continue

            parts = line.split()
            if len(parts) < 6:
                continue
            if not parts[0][0].isalpha():
                continue

            net = parts[0]
            sta = parts[1]
            phase = parts[2].upper()
            if phase not in ("P", "S"):
                continue

            # observed travel time
            try:
                t_obs = float(parts[PICK_TTIME_COL])
            except Exception:
                continue

            # station lon/lat (last two cols)
            try:
                slon = float(parts[-2])
                slat = float(parts[-1])
            except Exception:
                continue

            key = f"{net}.{sta}"
            stas_by_evid[cur_evid][key] = (slon, slat)
            picks.append(Pick(evid=cur_evid, net=net, sta=sta, phase=phase, t_obs=t_obs, slon=slon, slat=slat))

    return evloc, picks, stas_by_evid


def compute_real_obs(evloc: Dict[str, TrueLoc],
                     stas_by_evid: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[str, RealEventObs]:
    out: Dict[str, RealEventObs] = {}
    for evid, ev in evloc.items():
        stas = stas_by_evid.get(evid, {})
        nsta = len(stas)

        azs = [azimuth_deg(ev.lon, ev.lat, slon, slat) for (slon, slat) in stas.values()]
        gap = max_azimuth_gap(azs) if azs else 360.0

        dists = [haversine_km(ev.lon, ev.lat, slon, slat) for (slon, slat) in stas.values()]
        mean_dist = float(np.mean(dists)) if dists else float("nan")

        out[evid] = RealEventObs(
            evid=evid, lon=ev.lon, lat=ev.lat, dep_km=ev.dep_km,
            nsta=nsta, gap_deg=gap, mean_dist_km=mean_dist
        )
    return out


# =========================
# 2) NLLoc: read CSV
# =========================
import re 
def read_nlloc_csv(path: Path) -> Dict[str, LocSolution]:
    sols: Dict[str, LocSolution] = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evid = (row.get("event_id") or "").strip()
            m = re.search(r'_(\d+)\.sum$', evid)
            num_str = m.group(1)      # '00004259'
            evid = int(num_str)   # 4259
            if not evid:
                continue
            try:
                lat  = float(row["lat"])
                lon  = float(row["lon"])
                dep  = float(row["dep_km"])
                herr = float(row["h_err_km"])
                zerr = float(row["z_err_km"])
            except Exception:
                continue
            if not _in_bbox(lon, lat):
                continue
            sols[str(evid)] = LocSolution(evid=str(evid), lon=lon, lat=lat, dep_km=dep, h_err_km=herr, z_err_km=zerr)
    return sols


# =========================
# 3) Ours: read reloc.synth.True.txt (also extract true)
# =========================
def read_ours_txt(path: Path) -> Tuple[Dict[str, LocSolution], Dict[str, TrueLoc]]:
    """
    #{evid},{tstr},{lon},{lat},{dep},{x},{y},{z},{xs_std0},{xs_std1},{xs_std2},{err0},{err1},{err2},{orig_lon},{orig_lat},{orig_dep}

    Units:
      - solution lon/lat/dep in deg/km
      - err0/err1/err2 in km OR meters (controlled by OURS_ERR_IN_METERS)
      - true/orig lon/lat/dep in deg/km
    """
    scale = 1000.0 if OURS_ERR_IN_METERS else 1.0  # meters -> km

    sols: Dict[str, LocSolution] = {}
    trues: Dict[str, TrueLoc] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s.startswith("#"):
                continue
            s = s[1:]
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 17:
                continue
            evid = str(parts[0])

            try:
                lon = float(parts[2]); lat = float(parts[3]); dep = float(parts[4])
                err0 = float(parts[8]); err1 = float(parts[9]); err2 = float(parts[10])
                olon = float(parts[14]); olat = float(parts[15]); odep = float(parts[16])
            except Exception:
                continue

            if not _in_bbox(lon, lat):
                continue

            ex = abs(err0) / scale
            ey = abs(err1) / scale
            ez = abs(err2) / scale

            # sigma_h proxy (radial), consistent with your earlier usage:
            h = math.sqrt(ex*ex + ey*ey)

            sols[evid] = LocSolution(evid=evid, lon=lon, lat=lat, dep_km=dep, h_err_km=h, z_err_km=ez,
                                     ex_km=ex, ey_km=ey, ez_km=ez)
            trues[evid] = TrueLoc(lon=olon, lat=olat, dep_km=odep)

    return sols, trues


# =========================
# 4) PINN travel time
# =========================
class PINNTravelTime(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
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
        # x/xs: meters for x,y and meters for z (we use km->m here)
        inp = torch.cat([x, xs.expand_as(x)], dim=-1)  # (B,6)
        inp = inp / 1000.0 / 1000.0  # -> km
        out = self.net_merge(inp)
        out = out * 10.0
        return out  # (B,2) -> [Tp, Ts]


def load_pinn(ckpt_path: Path, device: torch.device, dtype: torch.dtype) -> PINNTravelTime:
    model = PINNTravelTime().to(device).eval()
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device=device, dtype=dtype)
    return model


# =========================
# Plotting: Fig1 geometry consistency
# =========================
def fig1_geometry(real_obs: Dict[str, RealEventObs],
                  nlloc: Dict[str, LocSolution],
                  ours: Dict[str, LocSolution]) -> None:
    """
    JGR-style: 2 rows x 3 cols
      Row 1: Ours
      Row 2: NLLoc
    Grayscale only, no subplot titles, panel labels (a)-(f).
    Use log y-scale to avoid NLLoc being visually squashed.
    """

    # ---- match evids (intersection) ----
    evids_o = sorted(set(real_obs.keys()) & set(ours.keys()))
    evids_n = sorted(set(real_obs.keys()) & set(nlloc.keys()))

    if len(evids_o) == 0 or len(evids_n) == 0:
        print(f"[warn] matched events: ours={len(evids_o)}, nlloc={len(evids_n)}. "
              f"Check bbox filters / event_id consistency.")
        # continue anyway; plot what exists

    def arr(evids, getter):
        return np.array([getter(e) for e in evids], dtype=float) if len(evids) else np.array([], dtype=float)

    # Geometry metrics from REAL (use same x for both methods, per matched evids)
    gap_o  = arr(evids_o, lambda e: real_obs[e].gap_deg)
    nsta_o = arr(evids_o, lambda e: real_obs[e].nsta)
    md_o   = arr(evids_o, lambda e: real_obs[e].mean_dist_km)

    gap_n  = arr(evids_n, lambda e: real_obs[e].gap_deg)
    nsta_n = arr(evids_n, lambda e: real_obs[e].nsta)
    md_n   = arr(evids_n, lambda e: real_obs[e].mean_dist_km)

    sh_o = arr(evids_o, lambda e: ours[e].h_err_km)
    sz_o = arr(evids_o, lambda e: ours[e].z_err_km)

    sh_n = arr(evids_n, lambda e: nlloc[e].h_err_km)
    sz_n = arr(evids_n, lambda e: nlloc[e].z_err_km)
    #print(nlloc[:3])
    # ---- figure canvas ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=False, sharey=False)

    # Grayscale styles (explicit colors as requested)
    # Use different markers; same color family
    ours_kw = dict(marker="x", s=28, linewidths=1.2, color="0.35", alpha=0.55)
    nll_kw  = dict(marker="+", s=28, linewidths=1.2, color="0.0",  alpha=0.75)

    # Helper: consistent axes styling (JGR-ish)
    def style_ax(ax):
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.tick_params(direction="out", length=4, width=1.0, labelsize=10)

    # ---- Row 1: Ours ----
    # (a) sigma_h vs gap
    ax = axes[0, 0]
    ax.scatter(gap_o, sh_o, **ours_kw)
    ax.set_xlabel("Azimuthal gap (deg)", fontsize=11)
    ax.set_ylabel(r"$\sigma_h$ (km)", fontsize=11)
    style_ax(ax)

    # (b) sigma_h vs nsta
    ax = axes[0, 1]
    ax.scatter(nsta_o, sh_o, **ours_kw)
    ax.set_xlabel(r"$n_{\mathrm{sta}}$", fontsize=11)
    ax.set_ylabel(r"$\sigma_h$ (km)", fontsize=11)
    style_ax(ax)

    # (c) sigma_z vs mean distance
    ax = axes[0, 2]
    ax.scatter(md_o, sz_o, **ours_kw)
    ax.set_xlabel("Mean source-station distance (km)", fontsize=11)
    ax.set_ylabel(r"$\sigma_z$ (km)", fontsize=11)
    style_ax(ax)

    # ---- Row 2: NLLoc ----
    # (d) sigma_h vs gap
    ax = axes[1, 0]
    ax.scatter(gap_n, sh_n, **nll_kw)
    ax.set_xlabel("Azimuthal gap (deg)", fontsize=11)
    ax.set_ylabel(r"$\sigma_h$ (km)", fontsize=11)
    style_ax(ax)

    # (e) sigma_h vs nsta
    ax = axes[1, 1]
    ax.scatter(nsta_n, sh_n, **nll_kw)
    ax.set_xlabel(r"$n_{\mathrm{sta}}$", fontsize=11)
    ax.set_ylabel(r"$\sigma_h$ (km)", fontsize=11)
    style_ax(ax)

    # (f) sigma_z vs mean distance
    ax = axes[1, 2]
    ax.scatter(md_n, sz_n, **nll_kw)
    ax.set_xlabel("Mean source-station distance (km)", fontsize=11)
    ax.set_ylabel(r"$\sigma_z$ (km)", fontsize=11)
    style_ax(ax)

    # ---- Make y-scale readable (prevents "no NLLoc visible") ----
    # Log-scale is the most robust choice given your σ can span orders of magnitude
    for r in range(2):
        axes[r, 0].set_yscale("log")
        axes[r, 1].set_yscale("log")
    axes[0, 2].set_yscale("log")
    axes[1, 2].set_yscale("log")
    
    # Avoid log(0) issues: set a small lower bound if needed
    def set_log_ylim(ax, y):
        y = y[np.isfinite(y)]
        y = y[y > 0]
        if len(y) == 0:
            return
        lo = max(np.min(y) * 0.8, 1e-2)
        hi = np.max(y) * 1.25
        ax.set_ylim(1, 5e3)

    # Row-wise limits (so each method's structure is visible)
    set_log_ylim(axes[0, 0], sh_o); set_log_ylim(axes[0, 1], sh_o); set_log_ylim(axes[0, 2], sz_o)
    set_log_ylim(axes[1, 0], sh_n); set_log_ylim(axes[1, 1], sh_n); set_log_ylim(axes[1, 2], sz_n)

    # ---- panel labels (a)-(f) ----
    panel = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    k = 0
    for i in range(2):
        for j in range(3):
            axes[i, j].text(0.02, 0.96, panel[k],
                            transform=axes[i, j].transAxes,
                            ha="left", va="top",
                            fontsize=12, fontweight="bold", color="0.0")
            k += 1

    # ---- compact spacing (JGR-like) ----
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.10, top=0.98, wspace=0.25, hspace=0.22)

    out = OUT_DIR / "fig_geom_consistency_gray_2rows.v1.0.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[save] {out}")



# =========================
# PPC: residual distributions using PINN
# =========================
@torch.no_grad()
def compute_tt_residuals(picks: List[Pick],
                         sol: Dict[str, LocSolution],
                         model: PINNTravelTime,
                         device: torch.device,
                         dtype: torch.dtype,
                         batch: int = 65536) -> Dict[str, np.ndarray]:
    """
    Returns:
      residuals["P"] = t_obs - t_predP
      residuals["S"] = t_obs - t_predS
    """
    xs_list = []
    xr_list = []
    t_list = []
    ph_list = []

    for pk in picks:
        ev = sol.get(pk.evid)
        if ev is None:
            continue

        xe, ye = lonlat_to_xy_m(ev.lon, ev.lat)  # meters
        ze = ev.dep_km * 1000.0                  # meters
        xr, yr = lonlat_to_xy_m(pk.slon, pk.slat)
        zr = 0.0

        xs_list.append((xe, ye, ze))
        xr_list.append((xr, yr, zr))
        t_list.append(pk.t_obs)
        ph_list.append(pk.phase)

    if not xs_list:
        return {"P": np.array([]), "S": np.array([])}

    xs = torch.tensor(xs_list, device=device, dtype=dtype)
    xr = torch.tensor(xr_list, device=device, dtype=dtype)
    t_obs = torch.tensor(t_list, device=device, dtype=dtype)

    resP, resS = [], []
    ph_arr = np.array(ph_list)

    n = xs.shape[0]
    for i in range(0, n, batch):
        j = min(i + batch, n)
        out = model(xr[i:j], xs[i:j])  # (B,2)
        tp = out[:, 0]
        ts = out[:, 1]
        to = t_obs[i:j]
        ph = ph_arr[i:j]

        if np.any(ph == "P"):
            idx = torch.from_numpy(np.where(ph == "P")[0]).to(device)
            resP.append((to[idx] - tp[idx]).detach().cpu().numpy())
        if np.any(ph == "S"):
            idx = torch.from_numpy(np.where(ph == "S")[0]).to(device)
            resS.append((to[idx] - ts[idx]).detach().cpu().numpy())

    return {
        "P": np.concatenate(resP) if resP else np.array([]),
        "S": np.concatenate(resS) if resS else np.array([]),
    }


def fig2_ppc_color(picks: List[Pick],
             nlloc: Dict[str, LocSolution],
             ours: Dict[str, LocSolution],
             device: torch.device,
             dtype: torch.dtype) -> None:

    model = load_pinn(CKPT_PATH, device=device, dtype=dtype)

    res_n = compute_tt_residuals(picks, nlloc, model, device, dtype)
    res_o = compute_tt_residuals(picks, ours,  model, device, dtype)

    # Plot: 2x2 -> P hist, S hist, P ECDF, S ECDF
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def hist(ax, a, b, title):
        ax.hist(a, bins=80, density=True, alpha=0.6, label="NLLoc")
        ax.hist(b, bins=80, density=True, alpha=0.6, label="Ours")
        ax.set_title(title)
        ax.set_xlabel("Residual (s)  (t_obs - t_pred)")
        ax.set_ylabel("Density")
        ax.legend(frameon=True)

    def ecdf(ax, a, b, title):
        def _ecdf(x):
            x = np.sort(x)
            y = np.linspace(0, 1, len(x), endpoint=True)
            return x, y
        xa, ya = _ecdf(a)
        xb, yb = _ecdf(b)
        ax.plot(xa, ya, label="NLLoc")
        ax.plot(xb, yb, label="Ours")
        ax.set_title(title)
        ax.set_xlabel("Residual (s)")
        ax.set_ylabel("ECDF")
        ax.legend(frameon=True)

    # Robust xlim for readability
    def clip(x):
        if len(x) == 0:
            return x
        q = np.quantile(x, [0.01, 0.99])
        return x[(x >= q[0]) & (x <= q[1])]

    p_n = clip(res_n["P"]); p_o = clip(res_o["P"])
    s_n = clip(res_n["S"]); s_o = clip(res_o["S"])

    hist(axes[0, 0], p_n, p_o, "P-phase residuals")
    hist(axes[0, 1], s_n, s_o, "S-phase residuals")
    ecdf(axes[1, 0], p_n, p_o, "P-phase residual ECDF")
    ecdf(axes[1, 1], s_n, s_o, "S-phase residual ECDF")

    fig.tight_layout()
    out = OUT_DIR / "fig_ppc_residuals.v1.0.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[save] {out}")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def fig2_ppc(picks: List[Pick],
             nlloc: Dict[str, LocSolution],
             ours: Dict[str, LocSolution],
             device: torch.device,
             dtype: torch.dtype) -> None:
    print("LOADED PPC")
    model = load_pinn(CKPT_PATH, device=device, dtype=dtype)

    res_n = compute_tt_residuals(picks, nlloc, model, device, dtype)
    res_o = compute_tt_residuals(picks, ours,  model, device, dtype)

    # --------- JGR-ish global style (grayscale-friendly) ----------
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 300,
    })

    # Plot: 2x2 -> P hist, S hist, P ECDF, S ECDF
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), constrained_layout=False)

    # ---------- helpers ----------
    def clip_for_display(x: np.ndarray, qlo=0.01, qhi=0.99) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return x
        lo, hi = np.quantile(x, [qlo, qhi])
        return x[(x >= lo) & (x <= hi)]

    def set_common(ax):
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="0.85")
        ax.minorticks_on()
        ax.tick_params(which="both", top=True, right=True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    def panel_label(ax, s):
        ax.text(0.02, 0.98, s, transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")

    def hist_overlay(ax, x_n, x_o, bins=70):
        # compute shared bins from combined data (display-clipped)
        x_all = np.concatenate([x_n, x_o]) if (x_n.size and x_o.size) else (x_n if x_n.size else x_o)
        if x_all.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            return

        edges = np.histogram_bin_edges(x_all, bins=bins)

        # NLLoc: light gray filled
        ax.hist(x_n, bins=edges, density=True,
                facecolor="0.80", edgecolor="0.60", linewidth=0.6,
                label="NLLoc")

        # Ours: outline-only, black (so it works in grayscale)
        ax.hist(x_o, bins=edges, density=True, histtype="step",
                color="0.10", linewidth=1.2, linestyle="-",
                label="Ours")

        ax.set_ylabel("Density")
        set_common(ax)

    def ecdf_overlay(ax, x_n, x_o):
        def _ecdf(x):
            x = np.sort(x)
            if x.size == 0:
                return x, x
            y = np.arange(1, x.size + 1) / x.size
            return x, y

        xn, yn = _ecdf(x_n)
        xo, yo = _ecdf(x_o)

        if xn.size == 0 and xo.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            return

        # NLLoc: gray dashed
        if xn.size:
            ax.plot(xn, yn, color="0.35", linewidth=1.0, linestyle="--", label="NLLoc")
        # Ours: black solid
        if xo.size:
            ax.plot(xo, yo, color="0.05", linewidth=1.3, linestyle="-", label="Ours")

        ax.set_ylabel("ECDF")
        set_common(ax)

    # --------- data (display-clipped for readability) ----------
    p_n = clip_for_display(res_n.get("P", np.array([])))
    p_o = clip_for_display(res_o.get("P", np.array([])))
    s_n = clip_for_display(res_n.get("S", np.array([])))
    s_o = clip_for_display(res_o.get("S", np.array([])))

    # shared x-lims per phase (so hist + ecdf align visually)
    def xlim_from(a, b):
        x = np.concatenate([a, b]) if (a.size and b.size) else (a if a.size else b)
        if x.size == 0:
            return None
        lo, hi = np.min(x), np.max(x)
        if np.isclose(lo, hi):
            pad = 1.0
            return (lo - pad, hi + pad)
        pad = 0.05 * (hi - lo)
        return (lo - pad, hi + pad)

    xlim_p = xlim_from(p_n, p_o)
    xlim_s = xlim_from(s_n, s_o)

    # --------- draw panels (no titles) ----------
    ax = axes[0, 0]
    hist_overlay(ax, p_n, p_o)
    ax.set_xlabel(r"Residual (s)  ($t_{\mathrm{obs}} - t_{\mathrm{pred}}$)")
    if xlim_p: ax.set_xlim(*xlim_p)
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    hist_overlay(ax, s_n, s_o)
    ax.set_xlabel(r"Residual (s)  ($t_{\mathrm{obs}} - t_{\mathrm{pred}}$)")
    if xlim_s: ax.set_xlim(*xlim_s)
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    ecdf_overlay(ax, p_n, p_o)
    ax.set_xlabel("Residual (s)")
    if xlim_p: ax.set_xlim(*xlim_p)
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    ecdf_overlay(ax, s_n, s_o)
    ax.set_xlabel("Residual (s)")
    if xlim_s: ax.set_xlim(*xlim_s)
    panel_label(ax, "(d)")

    # Legend: put once, keep clean
    # (use upper center inside figure to save space)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, 1.02))

    # tighter spacing suited for journal columns
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.92,
                        wspace=0.25, hspace=0.28)

    out = OUT_DIR / "fig_ppc_residuals.v1.1_grayscale.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")

# =========================
# Coverage / calibration test
# =========================
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

def fig3_coverage(nlloc: Dict[str, "LocSolution"],
                  ours: Dict[str, "LocSolution"],
                  true_loc: Dict[str, "TrueLoc"]) -> None:
    """
    Grayscale / print-friendly (JGR-style): distinguish curves by linestyle + marker,
    avoid relying on color.
    """

    # --- JGR-ish style defaults (local to this figure) ---
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "savefig.dpi": 600,          # JGR print-friendly
        "pdf.fonttype": 42,          # TrueType (editable text)
        "ps.fonttype": 42,
    })

    def xy_km(lon, lat):
        x_m, y_m = lonlat_to_xy_m(lon, lat)
        return x_m / 1000.0, y_m / 1000.0

    def collect(sol: Dict[str, "LocSolution"]) -> Tuple[np.ndarray, np.ndarray]:
        zh, zz = [], []
        for evid, s in sol.items():
            t = true_loc.get(evid)
            if t is None:
                continue

            xs, ys = xy_km(s.lon, s.lat)
            xt, yt = xy_km(t.lon, t.lat)
            dh = math.sqrt((xs - xt) ** 2 + (ys - yt) ** 2)

            dz = abs(s.dep_km - t.dep_km)

            if getattr(s, "h_err_km", 0.0) and s.h_err_km > 0:
                zh.append(dh / s.h_err_km)
            if getattr(s, "z_err_km", 0.0) and s.z_err_km > 0:
                zz.append(dz / s.z_err_km)

        return np.asarray(zh, float), np.asarray(zz, float)

    zh_n, zz_n = collect(nlloc)
    zh_o, zz_o = collect(ours)

    def ecdf(x):
        x = np.sort(x)
        # 注意：len(x)=1 时也正常
        y = np.linspace(0.0, 1.0, len(x), endpoint=True) if len(x) > 0 else np.array([])
        return x, y

    # Theoretical CDFs
    xgrid = np.linspace(0, 5, 400)
    rayleigh_cdf = 1.0 - np.exp(-0.5 * xgrid ** 2)
    halfnorm_cdf = np.array([math.erf(v / math.sqrt(2.0)) for v in xgrid])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))  # 两栏宽常用 ~6.8 in

    # ---- Style map (grayscale + line styles + markers) ----
    # empirical curves
    style_nll = dict(color="0.10", linestyle="-",  marker="o", markersize=3.2,
                     markevery=0.08, markerfacecolor="white", markeredgewidth=0.8)
    style_our = dict(color="0.35", linestyle="--", marker="s", markersize=3.0,
                     markevery=0.08, markerfacecolor="white", markeredgewidth=0.8)
    # theory curve
    style_th  = dict(color="0.65", linestyle="-.", linewidth=1.6)

    # Horizontal
    ax = axes[0]
    if len(zh_n) > 0:
        x, y = ecdf(zh_n)
        ax.plot(x, y, label="NLLoc (empirical)", **style_nll)
    if len(zh_o) > 0:
        x, y = ecdf(zh_o)
        ax.plot(x, y, label="This study (empirical)", **style_our)
    ax.plot(xgrid, rayleigh_cdf, label="Rayleigh (theory)", **style_th)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\Delta_h / \sigma_h$")
    ax.set_ylabel("CDF")
    ax.set_title("Horizontal coverage")
    ax.grid(True, which="both", linewidth=0.4, color="0.85")
    ax.legend(frameon=False, loc="lower right")

    # Depth
    ax = axes[1]
    if len(zz_n) > 0:
        x, y = ecdf(zz_n)
        ax.plot(x, y, label="NLLoc (empirical)", **style_nll)
    if len(zz_o) > 0:
        x, y = ecdf(zz_o)
        ax.plot(x, y, label="This study (empirical)", **style_our)
    ax.plot(xgrid, halfnorm_cdf, label="Half-normal (theory)", **style_th)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$|\Delta_z| / \sigma_z$")
    ax.set_ylabel("CDF")
    ax.set_title("Depth coverage")
    ax.grid(True, which="both", linewidth=0.4, color="0.85")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()

    out_png = OUT_DIR / "fig_coverage_calibration.grayscale.png"
    out_pdf = OUT_DIR / "fig_coverage_calibration.grayscale.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")
    print(f"[save] {out_pdf}")

    # Optional: report a couple of key coverages for text/table
    def report(name, z):
        if len(z) == 0:
            print(f"[coverage] {name}: empty")
            return
        for k in [1.0, 2.0, 3.0]:
            frac = float(np.mean(z <= k))
            print(f"[coverage] {name}: P(z<= {k:.1f}) = {frac:.3f}")

    report("NLLoc horiz", zh_n)
    report("This study horiz", zh_o)
    report("NLLoc depth", zz_n)
    report("This study depth", zz_o)


# =========================
# main
# =========================
def main():
    print("[load] REAL events & picks:", REAL_TXT)
    evloc_real, picks, stas_by_evid = parse_real_events_and_picks(REAL_TXT)
    real_obs = compute_real_obs(evloc_real, stas_by_evid)
    print(f"[real] events={len(real_obs)} picks={len(picks)}")

    print("[load] NLLoc:", NLLOC_CSV)
    nlloc = read_nlloc_csv(NLLOC_CSV)
    print(f"[nlloc] sols={len(nlloc)}")

    print("[load] Ours:", OURS_TXT)
    ours, true_from_ours = read_ours_txt(OURS_TXT)
    print(f"[ours] sols={len(ours)} true={len(true_from_ours)}")

    # For coverage: prefer true from ours (synth True file)
    true_loc = true_from_ours

    # Device for PPC (CPU works; CUDA optional)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Fig1 geometry
    fig1_geometry(real_obs, nlloc, ours)

    # Fig2 PPC residuals
    fig2_ppc(picks, nlloc, ours, device=device, dtype=dtype)

    # Fig3 Coverage / calibration
    fig3_coverage(nlloc, ours, true_loc)

    print("[done] outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
