#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -------------------------
# User config
# -------------------------
LON_RANGE = (96.0, 109.0)
LAT_RANGE = (20.0, 35.0)

REAL_TXT = Path("run_fm3d/data/loc.synth_arrivals_skfmm_noise.new.txt")
NLLOC_CSV = Path("run_fm3d/data/nlloc/all.locfiles.csv")
OURS_TXT = Path("run_fm3d/data/ours/reloc.synth.True.v1.0.txt")

# Thresholds for (NLLoc / Ours) uncertainty classification
H_THR_KM = 10.0
Z_THR_KM = 20.0

# Output
OUT_DIR = Path("figures/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Utilities
# -------------------------
def _in_bbox(lon: float, lat: float) -> bool:
    return (LON_RANGE[0] <= lon <= LON_RANGE[1]) and (LAT_RANGE[0] <= lat <= LAT_RANGE[1])

def _wrap360(deg: float) -> float:
    deg = deg % 360.0
    return deg + 360.0 if deg < 0 else deg

def azimuth_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Forward azimuth from (lon1,lat1) to (lon2,lat2), degrees in [0,360)."""
    # spherical approx
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
    max_gap = float(np.max(np.append(gaps, wrap_gap)))
    return max_gap


# -------------------------
# Data models
# -------------------------
@dataclass
class RealEventObs:
    evid: int
    lon: float
    lat: float
    dep_km: float
    nsta: int
    gap_deg: float


@dataclass
class LocSolution:
    evid: str
    lon: float
    lat: float
    dep_km: float
    h_err_km: float
    z_err_km: float


# -------------------------
# 1) REAL: parse and compute (nsta, gap)
# -------------------------
def parse_real_file_compute_obs(path: Path) -> List[RealEventObs]:
    """
    REAL file format (header line starts with integer evid):
      783 ... lat lon dep ...
    Pick lines include station lon/lat at the end:
      ... <sta_lon> <sta_lat>
    We compute:
      - nsta = number of unique stations with picks for that event
      - gap  = max azimuthal gap of station distribution around event
    """
    events: List[RealEventObs] = []

    cur_evid: Optional[int] = None
    cur_lon = cur_lat = cur_dep = None
    stas: Dict[str, Tuple[float, float]] = {}  # key: NET.STA or STA, value: (lon,lat)

    def flush():
        nonlocal cur_evid, cur_lon, cur_lat, cur_dep, stas
        if cur_evid is None or cur_lon is None or cur_lat is None or cur_dep is None:
            return
        if not _in_bbox(cur_lon, cur_lat):
            stas = {}
            return
        # compute nsta and gap
        nsta = len(stas)
        azs = []
        for (_k, (slon, slat)) in stas.items():
            if _in_bbox(slon, slat):  # optional bbox constraint on stations
                azs.append(azimuth_deg(cur_lon, cur_lat, slon, slat))
        gap = max_azimuth_gap(azs) if len(azs) > 0 else 360.0
        events.append(RealEventObs(
            evid=cur_evid, lon=cur_lon, lat=cur_lat, dep_km=cur_dep,
            nsta=nsta, gap_deg=gap
        ))
        stas = {}

    def is_event_header(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        tok = s.split()[0]
        return tok.isdigit()

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            if is_event_header(line):
                # flush previous event
                flush()

                parts = line.split()
                # from your earlier REAL parsing: lat=parts[7], lon=parts[8], dep=parts[9]
                cur_evid = int(parts[0])
                try:
                    cur_lat = float(parts[7])
                    cur_lon = float(parts[8])
                    cur_dep = float(parts[9])
                except Exception:
                    cur_evid = None
                    cur_lon = cur_lat = cur_dep = None
                    stas = {}
                continue

            # pick line: first token alpha (e.g., SC)
            parts = line.split()
            if len(parts) < 4:
                continue
            if not parts[0][0].isalpha():
                continue

            # Typical: NET STA ... sta_lon sta_lat at last two tokens
            net = parts[0]
            sta = parts[1]
            key = f"{net}.{sta}"
            try:
                slon = float(parts[-2])
                slat = float(parts[-1])
            except Exception:
                continue
            stas[key] = (slon, slat)

    # flush last
    flush()
    return events


# -------------------------
# 2) NLLoc: read solutions and classify by threshold
# -------------------------
def read_nlloc_csv(path: Path) -> List[LocSolution]:
    sols: List[LocSolution] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evid = (row.get("event_id") or "").strip()
            lat = row.get("lat")
            lon = row.get("lon")
            dep = row.get("dep_km")
            herr = row.get("h_err_km")
            zerr = row.get("z_err_km")
            if not evid:
                continue
            if not lat or not lon or not dep or not herr or not zerr:
                continue
            try:
                lat = float(lat); lon = float(lon); dep = float(dep)
                herr = float(herr); zerr = float(zerr)
            except Exception:
                continue
            if not _in_bbox(lon, lat):
                continue
            sols.append(LocSolution(evid=evid, lon=lon, lat=lat, dep_km=dep, h_err_km=herr, z_err_km=zerr))
    return sols


# -------------------------
# 3) Ours: parse "#evid, tstr, lon, lat, z, x, y, z, xs_std0..2, err0..2, orig_lon, orig_lat, orig_dep"
# -------------------------
def read_ours_txt(path: Path) -> List[LocSolution]:
    """
    Your write format:
    #{evid},{tstr},{lon},{lat},{z},{x},{y},{z},{xs_std0},{xs_std1},{xs_std2},{err0},{err1},{err2},{orig_lon},{orig_lat},{orig_dep}
    We'll interpret:
      lon,lat,dep_km from fields [2],[3],[4]
      horizontal uncertainty: sqrt(err0^2 + err1^2) / 1000 (if err in meters)
      vertical uncertainty: abs(err2) / 1000
    If your err0/err1/err2 are already in km, set SCALE=1.0 below.
    """
    SCALE = 1.0  # meters -> km. change to 1.0 if already km

    sols: List[LocSolution] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s.startswith("#"):
                continue
            s = s[1:]
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 17:
                continue
            evid = parts[0]
            try:
                lon = float(parts[2])
                lat = float(parts[3])
                dep = float(parts[4])
                err0 = float(parts[8])# * 0.001
                err1 = float(parts[9])# * 0.001 
                err2 = float(parts[10])# * 0.001 
            except Exception:
                continue
            if not _in_bbox(lon, lat):
                continue

            h_err = math.sqrt(err0 * err0 + err1 * err1) / SCALE
            z_err = abs(err2) / SCALE
            
            sols.append(LocSolution(
                evid=evid, lon=lon, lat=lat, dep_km=dep, h_err_km=h_err, z_err_km=z_err
            ))
    return sols


# -------------------------
# Plotting
# -------------------------
def _format_map_axes(ax):
    ax.set_xlim(LON_RANGE)
    ax.set_ylim(LAT_RANGE)

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")

    # JGR-like axes styling
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(direction="out", length=4, width=0.8, labelsize=9)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

def _format_map_axes_dep(ax):
    ax.set_xlim(LON_RANGE)
    ax.set_ylim(50, 0)

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Depth (km)")

    # JGR-like axes styling
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(direction="out", length=4, width=0.8, labelsize=9)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


def _panel_label(ax, s: str):
    ax.text(
        0.02, 0.98, s,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9),
        zorder=10,
    )


def plot_three_panels(
    real_events: List[RealEventObs],
    nlloc: List[LocSolution],
    ours: List[LocSolution],
    h_thr_km: float,
    z_thr_km: float,
    out_png: Path,
):
    # --- REAL categories
    lon1 = [e.lon for e in real_events if e.nsta < 4]
    lat1 = [e.lat for e in real_events if e.nsta < 4]
    dep1 = [e.dep_km for e in real_events if e.nsta < 4]
    lon2 = [e.lon for e in real_events if (e.nsta >= 4 and e.gap_deg > 180.0)]
    lat2 = [e.lat for e in real_events if (e.nsta >= 4 and e.gap_deg > 180.0)]
    dep2 = [e.dep_km for e in real_events if (e.nsta >= 4 and e.gap_deg > 180.0)]
    lon3 = [e.lon for e in real_events if (e.nsta >= 4 and e.gap_deg <= 180.0)]
    lat3 = [e.lat for e in real_events if (e.nsta >= 4 and e.gap_deg <= 180.0)]
    dep3 = [e.dep_km for e in real_events if (e.nsta >= 4 and e.gap_deg <= 180.0)]

    # --- NLLoc categories (good/bad)
    nll_good = [s for s in nlloc if (s.h_err_km < h_thr_km and s.z_err_km < z_thr_km)]
    nll_bad  = [s for s in nlloc if not (s.h_err_km < h_thr_km and s.z_err_km < z_thr_km)]


    # --- Ours categories (same colors as NLLoc)
    our_good = [s for s in ours if (s.h_err_km < h_thr_km and s.z_err_km < z_thr_km)]
    our_bad  = [s for s in ours if not (s.h_err_km < h_thr_km and s.z_err_km < z_thr_km)]

    # grayscale palette
    c_light = "0.75"
    c_mid   = "0.45"
    c_dark  = "0.10"
    c_good  = c_dark
    c_bad   = c_light

    fig = plt.figure(figsize=(10.5, 7.2), dpi=300)
    gs = fig.add_gridspec(2, 3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax1d = fig.add_subplot(gs[1, 0])
    ax2d = fig.add_subplot(gs[1, 1])
    ax3d = fig.add_subplot(gs[1, 2])
    # (a) REAL
    _format_map_axes(ax1)
    ax1.scatter(lon1, lat1, s=10, c=c_light, marker="$×$", linewidths=0,
                label=r"$n_{\mathrm{sta}}<4$")
    ax1.scatter(lon2, lat2, s=10, c=c_mid, marker="o", linewidths=0,
                label=r"$n_{\mathrm{sta}}\geq 4,\ \mathrm{gap}>180^\circ$")
    ax1.scatter(lon3, lat3, s=10, c=c_dark, marker="$+$", linewidths=0,
                label=r"$n_{\mathrm{sta}}\geq 4,\ \mathrm{gap}\leq 180^\circ$")
    _panel_label(ax1, "(a)")
    ax1.legend(
        frameon=True, fontsize=8, loc="lower left",
        framealpha=0.95, edgecolor="0.2"
    )

    _format_map_axes_dep(ax1d)
    ax1d.scatter(lon1, dep1, s=10, c=c_light, marker="$×$", linewidths=0,
                label=r"$n_{\mathrm{sta}}<4$")
    ax1d.scatter(lon2, dep2, s=10, c=c_mid, marker="o", linewidths=0,
                label=r"$n_{\mathrm{sta}}\geq 4,\ \mathrm{gap}>180^\circ$")
    ax1d.scatter(lon3, dep3, s=10, c=c_dark, marker="$+$", linewidths=0,
                label=r"$n_{\mathrm{sta}}\geq 4,\ \mathrm{gap}\leq 180^\circ$")
    _panel_label(ax1d, "(d)")
    ax1d.legend(
        frameon=True, fontsize=8, loc="lower left",
        framealpha=0.95, edgecolor="0.2"
    )

    # (b) NLLoc
    _format_map_axes(ax2)
    ax2.scatter([s.lon for s in nll_good], [s.lat for s in nll_good],
                s=10, c=c_good, marker="$+$", linewidths=0,
                label=rf"$h<{h_thr_km:g}\ \mathrm{{km}},\ z<{z_thr_km:g}\ \mathrm{{km}}, N={len(nll_good)}$")
    ax2.scatter([s.lon for s in nll_bad],  [s.lat for s in nll_bad],
                s=10, c=c_bad, marker="$×$", linewidths=0,
                label=rf"$h\geq {h_thr_km:g}\ \mathrm{{km}}\ \mathrm{{or}}\ z\geq {z_thr_km:g}\ \mathrm{{km}}, N={len(nll_bad)}$")
    _panel_label(ax2, "(b)")
    ax2.legend(
        frameon=True, fontsize=8, loc="lower left",
        framealpha=0.95, edgecolor="0.2"
    )

    _format_map_axes_dep(ax2d)
    ax2d.scatter([s.lon for s in nll_good], [s.dep_km for s in nll_good],
                s=10, c=c_good, marker="$+$", linewidths=0,
                label=rf"$h<{h_thr_km:g}\ \mathrm{{km}},\ z<{z_thr_km:g}\ \mathrm{{km}}, N={len(nll_good)}$")
    ax2d.scatter([s.lon for s in nll_bad],  [s.dep_km for s in nll_bad],
                s=10, c=c_bad, marker="$×$", linewidths=0,
                label=rf"$h\geq {h_thr_km:g}\ \mathrm{{km}}\ \mathrm{{or}}\ z\geq {z_thr_km:g}\ \mathrm{{km}}, N={len(nll_bad)}$")
    _panel_label(ax2d, "(e)")
    ax2d.legend(
        frameon=True, fontsize=8, loc="lower left",
        framealpha=0.95, edgecolor="0.2"
    )

    # (c) Ours
    _format_map_axes(ax3)
    ax3.scatter([s.lon for s in our_good], [s.lat for s in our_good],
                s=10, c=c_good, marker="$+$", linewidths=0,
                label=rf"$h<{h_thr_km:g}\ \mathrm{{km}},\ z<{z_thr_km:g}\ \mathrm{{km}}, N={len(our_good)}$")
    ax3.scatter([s.lon for s in our_bad],  [s.lat for s in our_bad],
                s=10, c=c_bad, marker="$×$", linewidths=0,
                label=rf"$h\geq {h_thr_km:g}\ \mathrm{{km}}\ \mathrm{{or}}\ z\geq {z_thr_km:g}\ \mathrm{{km}}, N={len(our_bad)}$")
    _panel_label(ax3, "(c)")
    ax3.legend(
        frameon=True, fontsize=8, loc="lower left",
        framealpha=0.95, edgecolor="0.2"
    )

    _format_map_axes_dep(ax3d)
    ax3d.scatter([s.lon for s in our_good], [s.dep_km for s in our_good],
                s=10, c=c_good, marker="$+$", linewidths=0,
                label=rf"$h<{h_thr_km:g}\ \mathrm{{km}},\ z<{z_thr_km:g}\ \mathrm{{km}}, N={len(our_good)}$")
    ax3d.scatter([s.lon for s in our_bad],  [s.dep_km for s in our_bad],
                s=10, c=c_bad, marker="$×$", linewidths=0,
                label=rf"$h\geq {h_thr_km:g}\ \mathrm{{km}}\ \mathrm{{or}}\ z\geq {z_thr_km:g}\ \mathrm{{km}}, N={len(our_bad)}$")
    _panel_label(ax3d, "(f)")
    ax3d.legend(
        frameon=True, fontsize=8, loc="lower left",
        framealpha=0.95, edgecolor="0.2"
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)



def main():
    real_events = parse_real_file_compute_obs(REAL_TXT)
    nlloc = read_nlloc_csv(NLLOC_CSV)
    ours = read_ours_txt(OURS_TXT)

    print(f"[REAL] events parsed: {len(real_events)}")
    print(f"[NLLoc] solutions parsed: {len(nlloc)}")
    print(f"[Ours] solutions parsed: {len(ours)}")

    out_png = OUT_DIR / "loc_quality_3panels.v1.0.png"
    plot_three_panels(
        real_events=real_events,
        nlloc=nlloc,
        ours=ours,
        h_thr_km=H_THR_KM,
        z_thr_km=Z_THR_KM,
        out_png=out_png
    )
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
