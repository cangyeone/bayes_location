#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_pinn_tt.py

Read phase picks -> project lon/lat to AEQD x/y -> load PINN travel-time model ->
predict P/S travel times -> compute residuals (pred - obs) -> bin by epicentral distance ->
plot JGR-style curves.

Usage example:
  python compare_pinn_tt.py \
    --phase_file run_fm3d/data/phases.txt \
    --meta_json run_fm3d/data/xyz_vp_vs_meta.json \
    --ckpt ckpt/time.v1.0.pt \
    --out_dir out_tt_compare \
    --batch_size 65536 \
    --device cuda

Outputs (PNG):
  out_tt_compare/tt_err_mae_vs_dist.png
  out_tt_compare/tt_err_rmse_vs_dist.png
  out_tt_compare/tt_err_bias_vs_dist.png
  out_tt_compare/tt_err_summary.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from pyproj import CRS, Transformer


# =========================
# 1) Data structures + parser (from your snippet)
# =========================
@dataclass
class Pick:
    event_id: int
    phase: str
    sta: str
    t_abs: float
    tt: float
    dist_km: float
    sta_lon: float
    sta_lat: float


@dataclass
class Event:
    event_id: int
    ev_lat: float
    ev_lon: float
    ev_dep_km: float


def is_int_token(tok: str) -> bool:
    return tok.isdigit()


def is_event_header(line: str) -> bool:
    s = line.lstrip()
    if not s:
        return False
    first = s.split()[0]
    return is_int_token(first)


def is_pick_line(line: str) -> bool:
    s = line.lstrip()
    if not s:
        return False
    first = s.split()[0]
    return first[0].isalpha()


def parse_event_header_fixed(line: str) -> Event:
    parts = line.strip().split()
    if len(parts) < 10:
        raise ValueError(f"Event header too short (need >=10 tokens):\n{line}")

    event_id = int(parts[0])
    ev_lat = float(parts[7])
    ev_lon = float(parts[8])
    ev_dep = float(parts[9])

    if not (-90.0 <= ev_lat <= 90.0 and -180.0 <= ev_lon <= 180.0 and ev_dep >= 0.0):
        raise ValueError(
            f"Parsed weird event lat/lon/dep from:\n{line}\n"
            f"got lat={ev_lat}, lon={ev_lon}, dep={ev_dep}"
        )

    return Event(event_id=event_id, ev_lat=ev_lat, ev_lon=ev_lon, ev_dep_km=ev_dep)


def parse_pick_line(line: str, current_event_id: int) -> Pick:
    parts = line.strip().split()
    if len(parts) < 9:
        raise ValueError(f"Pick line too short:\n{line}")

    sta = parts[1]
    phase = parts[2].upper()
    t_abs = float(parts[3])
    tt = float(parts[4])

    sta_lon = float(parts[-2])
    sta_lat = float(parts[-1])
    dist_km = float(parts[-3])

    return Pick(
        event_id=current_event_id,
        phase=phase,
        sta=sta,
        t_abs=t_abs,
        tt=tt,
        dist_km=dist_km,
        sta_lon=sta_lon,
        sta_lat=sta_lat,
    )


def read_phase_file(path: Path) -> Tuple[Dict[int, Event], List[Pick]]:
    events: Dict[int, Event] = {}
    picks: List[Pick] = []
    current_event_id: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            if is_event_header(line):
                ev = parse_event_header_fixed(line)
                events[ev.event_id] = ev
                current_event_id = ev.event_id
                continue

            if is_pick_line(line):
                if current_event_id is None:
                    raise ValueError("Encountered pick line before any event header.")
                pk = parse_pick_line(line, current_event_id)
                picks.append(pk)
                continue

    if not events:
        raise ValueError(f"No events parsed from {path}")
    if not picks:
        raise ValueError(f"No picks parsed from {path}")

    return events, picks


# =========================
# 2) Projection utilities (your AEQD)
# =========================
def build_aeqd_fwd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(CRS.from_epsg(4326), crs_aeqd, always_xy=True)  # lon,lat -> x,y (m)


def load_meta_lonlat0(meta_json: Path) -> Tuple[float, float, dict]:
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    if "lon0" not in meta or "lat0" not in meta:
        raise ValueError(f"meta_json missing lon0/lat0: {meta_json}")
    return float(meta["lon0"]), float(meta["lat0"]), meta


# =========================
# 3) PINN model + checkpoint loader (your spec)
# =========================
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
        # x : (B,3) receiver [xr,yr,zr]
        # xs: (B,3) source   [xe,ye,ze]
        inp = torch.cat([x, xs.expand_as(x)], dim=-1)  # (B,6)
        inp = inp / 1000.0
        out = self.net_merge(inp)
        out = out * 10.0
        return out


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


# =========================
# 4) Metrics + binning
# =========================
def robust_stats(residual: np.ndarray) -> dict:
    """
    residual: pred - obs (sec)
    """
    if residual.size == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, bias=np.nan, med=np.nan, p16=np.nan, p84=np.nan)
    absr = np.abs(residual)
    mae = float(np.mean(absr))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    bias = float(np.mean(residual))
    med = float(np.median(residual))
    p16 = float(np.percentile(residual, 16))
    p84 = float(np.percentile(residual, 84))
    return dict(n=int(residual.size), mae=mae, rmse=rmse, bias=bias, med=med, p16=p16, p84=p84)


def bin_by_distance(dist_km: np.ndarray, residual: np.ndarray, bins: List[Tuple[float, float]]) -> dict:
    out = {
        "bin_lo": [],
        "bin_hi": [],
        "bin_center": [],
        "n": [],
        "mae": [],
        "rmse": [],
        "bias": [],
        "med": [],
        "p16": [],
        "p84": [],
    }
    for lo, hi in bins:
        m = (dist_km >= lo) & (dist_km < hi)
        st = robust_stats(residual[m])
        out["bin_lo"].append(lo)
        out["bin_hi"].append(hi)
        out["bin_center"].append(0.5 * (lo + hi) if np.isfinite(hi) else lo)
        for k in ["n", "mae", "rmse", "bias", "med", "p16", "p84"]:
            out[k].append(st[k])
    for k in out:
        out[k] = np.asarray(out[k])
    return out

def read_station_loc(path: Path) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Parse china.loc lines like:
      AH A0001 40 117.2231 31.8175 32 ...
    Return dict[(net, sta)] = (lon, lat)
    """
    sta = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            net = parts[0]
            code = parts[1]
            # parts[2] may be something like "40" (region/quality); ignore
            lon = float(parts[3])
            lat = float(parts[4])
            dep = float(parts[5]) / 1000.0
            sta[(net, code)] = (lon, lat, -dep)
    return sta

@dataclass
class RealPick:
    phase: str          # "P" or "S"
    net: str
    sta: str
    tt: float           # observed travel time (s)
    ev_lon: float
    ev_lat: float
    ev_dep_km: float

from typing import DefaultDict

def parse_pha_file(
    path: Path,
    lon_range=(96.0, 109.0),
    lat_range=(20.0, 35.0),
) -> List[RealPick]:
    """
    Parse 2022.pha:
      #EVENT <evid> ... LOC lon lat DEP dep ...
      PHASE  <evid> ... TRAVTIME NET STA LOC CHAN <phase> ttime <val> ...
    Rules:
      - Keep events within given lon/lat ranges
      - Travel-time phases:
          P-group: prefer Pn/P over Pg if multiple for same (event, station)
          S-group: keep Sg (extendable)
      - Discard tt > 100 s (as you already did)
    """
    # phase mapping + priority within P-group
    # larger is higher priority
    P_PRI = {"Pn": 3, "P": 2, "Pg": 1}
    S_PRI = {"Sn": 3, "S": 2, "Sg": 1}  # extend later if needed: {"Sn":3,"S":2,"Sg":1}

    # store best pick per (event_id, net, sta, group)
    # value: (priority, RealPick)
    best: Dict[Tuple[str, str, str, str], Tuple[int, RealPick]] = {}

    cur_ev: Optional[Tuple[str, float, float, float]] = None
    # (event_id, lon, lat, dep_km)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue

            if s.startswith("#EVENT"):
                parts = s.split()
                # expect: #EVENT <evid> ... LOC lon lat DEP dep ...
                if len(parts) < 3:
                    cur_ev = None
                    continue
                evid = parts[1]

                if "LOC" not in parts or "DEP" not in parts:
                    cur_ev = None
                    continue
                i_loc = parts.index("LOC")
                i_dep = parts.index("DEP")

                try:
                    ev_lon = float(parts[i_loc + 1])
                    ev_lat = float(parts[i_loc + 2])
                    ev_dep = float(parts[i_dep + 1])
                except Exception:
                    cur_ev = None
                    continue

                if not (lon_range[0] <= ev_lon <= lon_range[1] and lat_range[0] <= ev_lat <= lat_range[1]):
                    cur_ev = None
                else:
                    cur_ev = (evid, ev_lon, ev_lat, ev_dep)
                continue

            if cur_ev is None:
                continue
            if "TRAVTIME" not in s:
                continue

            parts = s.split()
            try:
                i_tt = parts.index("TRAVTIME")
            except ValueError:
                continue

            # Expect: TRAVTIME NET STA LOC CHAN <phase> ttime <val> ...
            if i_tt + 6 >= len(parts):
                continue

            net = parts[i_tt + 1]
            sta = parts[i_tt + 2]
            pha = parts[i_tt + 5].strip()  # Pg/Sg/Pn/P/...

            # find ttime value
            if "ttime" not in parts:
                continue
            j = parts.index("ttime")
            if j + 1 >= len(parts):
                continue
            try:
                tt = float(parts[j + 1])
            except Exception:
                continue
            if tt > 100:
                continue

            evid, ev_lon, ev_lat, ev_dep = cur_ev

            # decide group + priority
            if pha in P_PRI:
                group = "P"
                pr = P_PRI[pha]
                phase_out = "P"   # RealPick.phase uses "P"/"S"
            elif pha in S_PRI:
                group = "S"
                pr = S_PRI[pha]
                phase_out = "S"
            else:
                continue  # ignore other phases

            pk = RealPick(
                phase=phase_out,
                net=net,
                sta=sta,
                tt=tt,
                ev_lon=ev_lon,
                ev_lat=ev_lat,
                ev_dep_km=ev_dep,
            )

            key = (evid, net, sta, group)
            if key not in best or pr > best[key][0]:
                best[key] = (pr, pk)
            # if equal priority, keep the first encountered (or you can choose min tt)

    picks = [v[1] for v in best.values()]
    if len(picks) == 0:
        raise ValueError(f"No TRAVTIME picks parsed from {path} after filtering.")
    return picks


def predict_real_picks(
    picks: List[RealPick],
    sta_lonlat: Dict[Tuple[str, str], Tuple[float, float]],
    fwd: Transformer,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 65536,
    receiver_z_km: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      dist_km (N,)
      residual (pred-obs) (N,)
      phase (N,) as "P"/"S"
    """
    # collect only picks with station coords
    kept = []
    for pk in picks:
        if (pk.net, pk.sta) in sta_lonlat:
            kept.append(pk)
    if len(kept) == 0:
        raise ValueError("No real picks matched station list (china.loc).")

    N = len(kept)
    phase = np.empty(N, dtype="U1")
    obs_tt = np.empty(N, dtype=np.float32)

    xr_m = np.empty(N, dtype=np.float32)
    yr_m = np.empty(N, dtype=np.float32)
    zr_m = np.full(N, float(receiver_z_km) * 1000.0, dtype=np.float32)

    xe_m = np.empty(N, dtype=np.float32)
    ye_m = np.empty(N, dtype=np.float32)
    ze_m = np.empty(N, dtype=np.float32)

    dist_km = np.empty(N, dtype=np.float32)

    for i, pk in enumerate(kept):
        slon, slat, dep = sta_lonlat[(pk.net, pk.sta)]
        xrm, yrm = fwd.transform(slon, slat)
        xem, yem = fwd.transform(pk.ev_lon, pk.ev_lat)

        xr_m[i] = xrm / 1000.0
        yr_m[i] = yrm / 1000.0
        zr_m[i] = dep 
        xe_m[i] = xem / 1000.0
        ye_m[i] = yem / 1000.0
        ze_m[i] = float(pk.ev_dep_km)
        #if xrm > 6000000 or yrm > 6000000 or xem > 6000000 or yem > 6000000:
        #    continue 
        #if xrm < -6000000 or yrm < -6000000 or xem < -6000000 or yem < -6000000:
        #    continue
        # epicentral distance (horizontal) in km
        dist_km[i] = float(np.hypot(xrm - xem, yrm - yem)) / 1000.0 

        phase[i] = pk.phase
        obs_tt[i] = float(pk.tt)
    print(xe_m, ze_m, dist_km)
    #plt.scatter(dist_km, obs_tt)
    #plt.show()
    # model inference
    pred_tt = np.empty(N, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        bs = int(batch_size)
        for i0 in range(0, N, bs):
            i1 = min(i0 + bs, N)
            x = torch.from_numpy(np.stack([xr_m[i0:i1], yr_m[i0:i1], zr_m[i0:i1]], axis=1)).to(device)
            xs = torch.from_numpy(np.stack([xe_m[i0:i1], ye_m[i0:i1], ze_m[i0:i1]], axis=1)).to(device)
            #print(x, xs)
            out = model(x, xs).detach().cpu().numpy()  # (B,2)

            ph = phase[i0:i1]
            # convention: out[:,0]=P, out[:,1]=S
            pred_tt[i0:i1] = np.where(ph == "P", out[:, 0], out[:, 1]).astype(np.float32)

    residual = (pred_tt - obs_tt).astype(np.float32)
    print(len(residual), "real picks predicted.")
    return dist_km, residual, phase


# =========================
# 5) Plotting (JGR-ish)
# =========================
def setup_jgr_axes(ax):
    ax.grid(True, which="major", linewidth=0.6, alpha=0.4)
    ax.tick_params(direction="out", length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_metric_curve(out_png: Path, title: str, x: np.ndarray, yP: np.ndarray, yS: np.ndarray,
                      nP: np.ndarray, nS: np.ndarray, ylabel: str):
    fig = plt.figure(figsize=(6.2, 3.8), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    setup_jgr_axes(ax)

    ax.plot(x, yP, marker="o", linewidth=1.2, markersize=4, label="P")
    ax.plot(x, yS, marker="s", linewidth=1.2, markersize=4, label="S")

    ax.set_xlabel("Epicentral distance (km)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True)

    # annotate sample counts lightly
    for xi, yi, ni in zip(x, yP, nP):
        if np.isfinite(yi) and ni > 0:
            ax.annotate(f"{ni}", (xi, yi), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7)
    for xi, yi, ni in zip(x, yS, nS):
        if np.isfinite(yi) and ni > 0:
            ax.annotate(f"{ni}", (xi, yi), textcoords="offset points", xytext=(0, -10),
                        ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _setup_jgr_axes(ax):
    ax.grid(True, which="major", linewidth=0.6, alpha=0.4)
    ax.tick_params(direction="out", length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def _panel_label(ax, s: str):
    # (a)(b)(c)(d) in top-left, inside axes
    ax.text(
        0.02, 0.98, s,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold"
    )

def _plot_two_series_with_n(ax, x, yP, yS, nP, nS, ylabel, title=None, annotate_n=True):
    _setup_jgr_axes(ax)
    ax.plot(x, yP, marker="o", linewidth=1.2, markersize=4, label="P")
    ax.plot(x, yS, marker="s", linewidth=1.2, markersize=4, label="S")
    ax.set_xlabel("Epicentral distance (km)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if annotate_n:
        # annotate n for each point; offset P upward, S downward
        for xi, yi, ni in zip(x, yP, nP):
            if np.isfinite(yi) and int(ni) > 0:
                ax.annotate(
                    f"{int(ni)}", (xi, yi),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7
                )
        for xi, yi, ni in zip(x, yS, nS):
            if np.isfinite(yi) and int(ni) > 0:
                ax.annotate(
                    f"{int(ni)}", (xi, yi),
                    textcoords="offset points", xytext=(0, -10),
                    ha="center", fontsize=7
                )

def plot_4panel_figure(out_png, x, bP, bS):
    """
    2x2 panels:
      (a) MAE
      (b) RMSE
      (c) Bias
      (d) Sample count N
    with per-point n annotations on (a)(b)(c).
    """
    fig = plt.figure(figsize=(7.2, 6.2), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    # (a) MAE
    _plot_two_series_with_n(
        axa, x,
        bP["mae"], bS["mae"],
        bP["n"], bS["n"],
        ylabel="MAE (pred - obs) (s)",
        title="MAE vs distance",
        annotate_n=True
    )
    _panel_label(axa, "(a)")

    # (b) RMSE
    _plot_two_series_with_n(
        axb, x,
        bP["rmse"], bS["rmse"],
        bP["n"], bS["n"],
        ylabel="RMSE (pred - obs) (s)",
        title="RMSE vs distance",
        annotate_n=True
    )
    _panel_label(axb, "(b)")

    # (c) Bias
    _plot_two_series_with_n(
        axc, x,
        bP["bias"], bS["bias"],
        bP["n"], bS["n"],
        ylabel="Mean (pred - obs) (s)",
        title="Bias vs distance",
        annotate_n=True
    )
    _panel_label(axc, "(c)")

    # (d) N vs distance (no need to annotate n here; curve itself is n)
    _setup_jgr_axes(axd)
    axd.plot(x, bP["n"], marker="o", linewidth=1.2, markersize=4, label="P")
    axd.plot(x, bS["n"], marker="s", linewidth=1.2, markersize=4, label="S")
    axd.set_xlabel("Epicentral distance (km)")
    axd.set_ylabel("Count (n)")
    axd.set_title("Sample count vs distance")
    axd.set_yscale("log")  # JGR里很常用；如果你不想log就删掉这行
    axd.legend(frameon=True)
    _panel_label(axd, "(d)")

    # one legend is enough if you prefer: keep per-panel legends or remove some
    # (这里我保留(d)的legend；若想全局legend可改成 fig.legend)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def _setup_jgr_axes(ax):
    ax.grid(True, which="major", linewidth=0.6, alpha=0.4)
    ax.tick_params(direction="out", length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def _panel_label(ax, s: str):
    ax.text(
        0.02, 0.98, s,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold"
    )

def _plot_two_series_with_n_categorical(
    ax, xpos, yP, yS, nP, nS,
    ylabel, title=None, xticklabels=None,
    annotate_n=True, ylim=None
):
    _setup_jgr_axes(ax)
    ax.plot(xpos, yP, marker="o", linewidth=1.2, markersize=4, label="P")
    ax.plot(xpos, yS, marker="s", linewidth=1.2, markersize=4, label="S")
    ax.set_xlabel("Epicentral distance bin (km)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if xticklabels is not None:
        ax.set_xticks(xpos)
        ax.set_xticklabels(xticklabels, rotation=0)

    if ylim is not None:
        ax.set_ylim(ylim)

    if annotate_n:
        for xi, yi, ni in zip(xpos, yP, nP):
            if np.isfinite(yi) and int(ni) > 0:
                ax.annotate(f"{int(ni)}", (xi, yi),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7)
        for xi, yi, ni in zip(xpos, yS, nS):
            if np.isfinite(yi) and int(ni) > 0:
                ax.annotate(f"{int(ni)}", (xi, yi),
                            textcoords="offset points", xytext=(0, -10),
                            ha="center", fontsize=7)

def plot_4panel_mae_bias_resid_hist(
    out_png,
    dist_bins,
    bP, bS,
    resid_P, resid_S,
    hist_bins=40
):
    """
    2x2 panels:
      (a) MAE vs distance bins (y fixed 0–0.5)
      (b) P residual distribution (hist of pred-obs)
      (c) |Bias| vs distance bins (y fixed 0–0.5)
      (d) S residual distribution (hist of pred-obs)

    x-axis for (a)(c): categorical bin labels "0-50", ...
    hist: shared symmetric range for fair comparison.
    """
    nbin = len(dist_bins)
    xpos = np.arange(nbin)
    xticklabels = [f"{int(lo)}-{int(hi)}" for (lo, hi) in dist_bins]

    # --- shared symmetric histogram edges based on robust percentile
    r_all = np.concatenate([resid_P, resid_S]).astype(float)
    if r_all.size == 0:
        R = 1.0
    else:
        R = float(np.percentile(np.abs(r_all), 99.5))
        R = max(R, 1e-3)
    edges = np.linspace(-R, R, hist_bins + 1)

    fig = plt.figure(figsize=(7.2, 6.2), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    # (a) MAE
    _plot_two_series_with_n_categorical(
        axa, xpos,
        bP["mae"], bS["mae"],
        bP["n"], bS["n"],
        ylabel="MAE (pred - obs) (s)",
        title="MAE vs distance bins",
        xticklabels=xticklabels,
        annotate_n=True,
        ylim=(0.0, 0.5),
    )
    _panel_label(axa, "(a)")
    axa.legend(frameon=True, loc="best")

    # (c) |Bias|
    abs_bias_P = np.abs(bP["bias"])
    abs_bias_S = np.abs(bS["bias"])
    _plot_two_series_with_n_categorical(
        axc, xpos,
        abs_bias_P, abs_bias_S,
        bP["n"], bS["n"],
        ylabel="|Mean(pred - obs)| (s)",
        title="Abs bias vs distance bins",
        xticklabels=xticklabels,
        annotate_n=True,
        ylim=(0.0, 0.5),
    )
    _panel_label(axc, "(c)")

    # (b) P residual histogram
    _setup_jgr_axes(axb)
    axb.hist(resid_P, bins=edges, histtype="bar", alpha=0.9)
    axb.axvline(0.0, linewidth=1.0)  # zero reference
    axb.set_xlabel("Residual (pred - obs) (s)")
    axb.set_ylabel("Count")
    axb.set_title("P residual distribution")
    _panel_label(axb, "(b)")

    # (d) S residual histogram
    _setup_jgr_axes(axd)
    axd.hist(resid_S, bins=edges, histtype="bar", alpha=0.9)
    axd.axvline(0.0, linewidth=1.0)
    axd.set_xlabel("Residual (pred - obs) (s)")
    axd.set_ylabel("Count")
    axd.set_title("S residual distribution")
    _panel_label(axd, "(d)")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
import matplotlib.ticker as mticker

def _setup_jgr_axes_gray(ax):
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.tick_params(direction="out", length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def _panel_label(ax, s: str):
    ax.text(
        0.02, 0.98, s,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold"
    )

def plot_4panel_mae_absbias_resid_hist_gray(
    out_png,
    dist_bins,
    bP, bS,
    resid_P, resid_S,
    hist_bins=40,
    y_fixed=(0.0, 0.5),
):
    """
    Panels:
      (a) MAE vs distance-bin (P solid, S dashed; grayscale)
      (b) |BIAS| vs distance-bin (P solid, S dashed; grayscale)
      (c) P residual distribution (pred-obs)
      (d) S residual distribution (pred-obs)

    Requirements:
      - grayscale only
      - P solid, S dashed
      - (c)(d) y-axis in scientific notation
      - (a)(b) xticklabels are bin ranges and rotated 45°
      - no titles; only (a)(b)(c)(d)
      - annotate n next to points in (a)(b)
    """
    nbin = len(dist_bins)
    xpos = np.arange(nbin)
    xticklabels = [f"{int(lo)}-{int(hi)}" for (lo, hi) in dist_bins]

    # shared symmetric histogram edges for fair comparison
    r_all = np.concatenate([resid_P, resid_S]).astype(float)
    if r_all.size == 0:
        R = 1.0
    else:
        R = float(np.percentile(np.abs(r_all), 99.5))
        R = max(R, 1e-3)
    edges = np.linspace(-R, R, hist_bins + 1)

    fig = plt.figure(figsize=(7.2, 6.0), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.22)

    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    # Line styles (grayscale)
    color = "k"
    lsP = "-"   # P solid
    lsS = "--"  # S dashed

    # ---------- (a) MAE ----------
    _setup_jgr_axes_gray(axa)
    axa.plot(xpos, bP["mae"], linestyle=lsP, color=color, marker="o",
             linewidth=1.2, markersize=4, label="P")
    axa.plot(xpos, bS["mae"], linestyle=lsS, color=color, marker="s",
             linewidth=1.2, markersize=4, label="S")
    axa.set_ylabel("MAE (pred - obs) (s)")
    axa.set_xlabel("Epicentral distance bin (km)")
    axa.set_ylim(y_fixed)
    axa.set_xticks(xpos)
    axa.set_xticklabels(xticklabels, rotation=15, ha="right")

    # annotate n near each point
    for xi, yi, ni in zip(xpos, bP["mae"], bP["n"]):
        if np.isfinite(yi) and int(ni) > 0:
            axa.annotate(f"{int(ni)}", (xi, yi),
                         textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=7, color="k")
    for xi, yi, ni in zip(xpos, bS["mae"], bS["n"]):
        if np.isfinite(yi) and int(ni) > 0:
            axa.annotate(f"{int(ni)}", (xi, yi),
                         textcoords="offset points", xytext=(0, -10),
                         ha="center", fontsize=7, color="k")

    # keep legend minimal & grayscale
    axa.legend(frameon=True, loc="best", fontsize=9)
    _panel_label(axa, "(a)")

    # ---------- (b) |BIAS| ----------
    _setup_jgr_axes_gray(axb)
    abs_bias_P = np.abs(bP["bias"])
    abs_bias_S = np.abs(bS["bias"])
    axb.plot(xpos, abs_bias_P, linestyle=lsP, color=color, marker="o",
             linewidth=1.2, markersize=4)
    axb.plot(xpos, abs_bias_S, linestyle=lsS, color=color, marker="s",
             linewidth=1.2, markersize=4)
    axb.set_ylabel("|BIAS| (s)")  # short label as requested
    axb.set_xlabel("Epicentral distance bin (km)")
    axb.set_ylim(y_fixed)
    axb.set_xticks(xpos)
    axb.set_xticklabels(xticklabels, rotation=15, ha="right")

    # annotate n
    for xi, yi, ni in zip(xpos, abs_bias_P, bP["n"]):
        if np.isfinite(yi) and int(ni) > 0:
            axb.annotate(f"{int(ni)}", (xi, yi),
                         textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=7, color="k")
    for xi, yi, ni in zip(xpos, abs_bias_S, bS["n"]):
        if np.isfinite(yi) and int(ni) > 0:
            axb.annotate(f"{int(ni)}", (xi, yi),
                         textcoords="offset points", xytext=(0, -10),
                         ha="center", fontsize=7, color="k")

    _panel_label(axb, "(b)")

    # ---------- (c) P residual distribution ----------
    _setup_jgr_axes_gray(axc)
    axc.hist(resid_P, bins=edges, color="0.35", edgecolor="0.35", histtype="bar")
    axc.axvline(0.0, color="k", linewidth=1.0)
    axc.set_xlabel("Residual (pred - obs) (s)")
    axc.set_ylabel("Count")
    axc.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    axc.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _panel_label(axc, "(c)")

    # ---------- (d) S residual distribution ----------
    _setup_jgr_axes_gray(axd)
    axd.hist(resid_S, bins=edges, color="0.65", edgecolor="0.65", histtype="bar")
    axd.axvline(0.0, color="k", linewidth=1.0)
    axd.set_xlabel("Residual (pred - obs) (s)")
    axd.set_ylabel("Count")
    axd.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    axd.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _panel_label(axd, "(d)")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_4panel_model_vs_real_gray(
    out_png: Path,
    dist_bins: List[Tuple[float, float]],
    bP_model: dict, bS_model: dict,
    bP_real: dict,  bS_real: dict,
    y_fixed=(0.0, 0.5),
):
    import matplotlib.pyplot as plt

    def setup(ax):
        ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
        ax.tick_params(direction="out", length=4, width=0.8)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

    def label(ax, s):
        ax.text(0.02, 0.98, s, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold")

    def plot_curve(ax, bP, bS, ylabel, y_fixed_new):
        xpos = np.arange(len(dist_bins))
        xt = [f"{int(lo)}-{int(hi)}" for lo, hi in dist_bins]

        # P solid / S dashed (both black)
        ax.plot(xpos, bP["val"], linestyle="-", color="k", marker="o",
                linewidth=1.2, markersize=4, label="P")
        ax.plot(xpos, bS["val"], linestyle="--", color="k", marker="s",
                linewidth=1.2, markersize=4, label="S")

        ax.set_ylim(y_fixed_new)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Epicentral distance bin (km)")
        ax.set_xticks(xpos)
        ax.set_xticklabels(xt, fontsize=7)

        # annotate n
        for xi, yi, ni in zip(xpos, bP["val"], bP["n"]):
            if np.isfinite(yi) and int(ni) > 0:
                ax.annotate(f"{int(ni)}", (xi, yi),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color="k")
        for xi, yi, ni in zip(xpos, bS["val"], bS["n"]):
            if np.isfinite(yi) and int(ni) > 0:
                ax.annotate(f"{int(ni)}", (xi, yi),
                            textcoords="offset points", xytext=(0, -10),
                            ha="center", fontsize=7, color="k")

    # Prepare MAE and |BIAS|
    def pack_mae(b):   return {"val": b["mae"],  "n": b["n"]}
    def pack_absb(b):  return {"val": np.abs(b["bias"]), "n": b["n"]}

    fig = plt.figure(figsize=(7.2, 6.0), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.30)

    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    for ax in (axa, axb, axc, axd):
        setup(ax)

    # (a) model/previous dataset MAE
    plot_curve(axa, pack_mae(bP_model), pack_mae(bS_model), ylabel="MAE (pred - obs) (s)", y_fixed_new=(0.0, 0.5))
    axa.legend(frameon=True, loc="best", fontsize=9)
    label(axa, "(a)")

    # (b) model/previous dataset |BIAS|
    plot_curve(axb, pack_absb(bP_model), pack_absb(bS_model), ylabel="|BIAS| (s)", y_fixed_new=(0.0, 0.5))
    label(axb, "(b)")

    # (c) real data MAE (Pg/Sg)
    plot_curve(axc, pack_mae(bP_real), pack_mae(bS_real), ylabel="MAE (pred - obs) (s)", y_fixed_new=(0.0, 2.5))
    label(axc, "(c)")

    # (d) real data |BIAS| (Pg/Sg)
    plot_curve(axd, pack_absb(bP_real), pack_absb(bS_real), ylabel="|BIAS| (s)", y_fixed_new=(0.0, 2.5))
    label(axd, "(d)")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =========================
# 6) Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase_file", type=str, default="run_fm3d/data/test.synth_arrivals_skfmm_noise.new.txt", help="Phase picks file")
    ap.add_argument("--meta_json", type=str, default="run_fm3d/data/xyz_vp_vs_meta.json", help="run_fm3d/data/xyz_vp_vs_meta.json")
    ap.add_argument("--ckpt", type=str, default="ckpt/time.v1.0.pt", help="ckpt/time.v1.0.pt")
    ap.add_argument("--out_dir", type=str, default="figures/figs")
    ap.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=65536)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--real_pha", type=str, default="data/2022.pha")
    ap.add_argument("--station_loc", type=str, default="data/china.loc")

    # receiver depth (if you later have station elevation/depth, modify here)
    ap.add_argument("--receiver_z_km", type=float, default=0.0, help="Receiver depth in km (positive down). Default 0.")
    args = ap.parse_args()

    phase_path = Path(args.phase_file)
    meta_path = Path(args.meta_json)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # distance bins requested
    dist_bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300)]

    # ---- load data
    events, picks = read_phase_file(phase_path)
    lon0, lat0, meta = load_meta_lonlat0(meta_path)
    print(lon0, lat0)
    fwd = build_aeqd_fwd(lon0, lat0)

    # ---- build arrays for NN
    # Only keep P/S picks with existing event
    keep: List[Pick] = []
    for pk in picks:
        ph = pk.phase.upper()
        if ph not in ("P", "S"):
            continue
        if pk.event_id not in events:
            continue
        keep.append(pk)

    if len(keep) == 0:
        raise ValueError("No usable P/S picks after filtering.")

    # project coords
    xr_m = np.empty(len(keep), dtype=np.float32)
    yr_m = np.empty(len(keep), dtype=np.float32)
    zr_m = np.full(len(keep), float(args.receiver_z_km) * 1000.0, dtype=np.float32)

    xe_m = np.empty(len(keep), dtype=np.float32)
    ye_m = np.empty(len(keep), dtype=np.float32)
    ze_m = np.empty(len(keep), dtype=np.float32)

    obs_tt = np.empty(len(keep), dtype=np.float32)
    phase = np.empty(len(keep), dtype="U1")
    dist_km = np.empty(len(keep), dtype=np.float32)


    for i, pk in enumerate(keep):
        ev = events[pk.event_id]

        xrm, yrm = fwd.transform(pk.sta_lon, pk.sta_lat)
        xem, yem = fwd.transform(ev.ev_lon, ev.ev_lat)

        xr_m[i] = xrm / 1000.0
        yr_m[i] = yrm / 1000.0
        xe_m[i] = xem / 1000.0
        ye_m[i] = yem / 1000.0
        ze_m[i] = float(ev.ev_dep_km)# * 1000.0

        obs_tt[i] = float(pk.tt)
        phase[i] = pk.phase.upper()
        dist_km[i] = float(pk.dist_km)

    # ---- load model
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = PINNTravelTime(hidden_dim=args.hidden_dim).to(device)
    model.eval()
    _epoch, _meta = load_checkpoint(str(ckpt_path), model, map_location=str(device), strict=True)

    # ---- inference in batches
    pred_tt = np.empty(len(keep), dtype=np.float32)

    with torch.no_grad():
        bs = int(args.batch_size)
        for i0 in range(0, len(keep), bs):
            i1 = min(i0 + bs, len(keep))

            x = torch.from_numpy(np.stack([xr_m[i0:i1], yr_m[i0:i1], zr_m[i0:i1]], axis=1)).to(device)
            xs = torch.from_numpy(np.stack([xe_m[i0:i1], ye_m[i0:i1], ze_m[i0:i1]], axis=1)).to(device)

            out = model(x, xs)  # (B,2) seconds
            out = out.detach().cpu().numpy()

            # convention: out[:,0]=P, out[:,1]=S
            ph = phase[i0:i1]
            tt_batch = np.where(ph == "P", out[:, 0], out[:, 1]).astype(np.float32)
            pred_tt[i0:i1] = tt_batch

    residual = (pred_tt - obs_tt).astype(np.float32)  # sec

    # ---- per-phase binning
    mP = phase == "P"
    mS = phase == "S"

    bP = bin_by_distance(dist_km[mP], residual[mP], dist_bins)
    bS = bin_by_distance(dist_km[mS], residual[mS], dist_bins)

    x = bP["bin_center"]  # same bins
    obs_tt_P = obs_tt[mP].astype(np.float32)
    obs_tt_S = obs_tt[mS].astype(np.float32)
    resid_P = residual[mP]
    resid_S = residual[mS]


    plot_4panel_model_vs_real_gray(
        out_png=out_dir / "tt_err_4panel_abcd_model_vs_real_gray.v1.0.png",
        dist_bins=dist_bins,
        bP_model=bP_model, bS_model=bS_model,
        bP_real=bP_real,   bS_real=bS_real,
        y_fixed=None,
    )



if __name__ == "__main__":
    main()
