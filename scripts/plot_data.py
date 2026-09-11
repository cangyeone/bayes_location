#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_synth_catalog_v2.py

Rules:
- Event header line: first non-space token is an integer (e.g., "3", "61", "64")
- Phase line: first non-space token starts with a letter (e.g., "SC")

We use:
- Event lat/lon/dep from fixed columns in event header:
    parts[7]=lat, parts[8]=lon, parts[9]=dep_km
  (Matches your examples: "... 0.0000  19.9922  95.9708  15.00 ...")
- Station lat/lon from the LAST TWO columns of the phase line.

Plots:
1) Map: events + stations (lon/lat)
2) Travel time vs distance (P/S)
3) Pick count histograms (P and S per event)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


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
    return tok.isdigit()  # strict: "61" ok, "61.0" not ok


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
    # "SC" or other alpha code
    return first[0].isalpha()


def parse_event_header_fixed(line: str) -> Event:
    """
    Parse event header assuming fixed columns:
      event_id = parts[0] (int)
      lat      = parts[7]
      lon      = parts[8]
      dep_km   = parts[9]
    """
    parts = line.strip().split()
    if len(parts) < 10:
        raise ValueError(f"Event header too short (need >=10 tokens):\n{line}")

    event_id = int(parts[0])
    ev_lat = float(parts[7])
    ev_lon = float(parts[8])
    ev_dep = float(parts[9])

    # sanity checks
    if not (-90.0 <= ev_lat <= 90.0 and -180.0 <= ev_lon <= 180.0 and ev_dep >= 0.0):
        raise ValueError(f"Parsed weird event lat/lon/dep from:\n{line}\n"
                         f"got lat={ev_lat}, lon={ev_lon}, dep={ev_dep}")

    return Event(event_id=event_id, ev_lat=ev_lat, ev_lon=ev_lon, ev_dep_km=ev_dep)


def parse_pick_line(line: str, current_event_id: int) -> Pick:
    """
    Expected (robust):
      <NET> <STA> <P|S> <t_abs> <tt> ... <dist_km> <sta_lon> <sta_lat>

    We parse:
      sta     = parts[1]
      phase   = parts[2]
      t_abs   = parts[3]
      tt      = parts[4]
      dist_km = parts[-3]
      lon/lat = parts[-2], parts[-1]
    """
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

            # ignore anything else

    if not events:
        raise ValueError(f"No events parsed from {path}")
    if not picks:
        raise ValueError(f"No picks parsed from {path}")

    return events, picks


def plot_map(events: Dict[int, Event], picks: List[Pick], out: Optional[Path] = None) -> None:
    # unique stations using last lon/lat
    sta_dict: Dict[str, Tuple[float, float]] = {}
    for pk in picks:
        sta_dict[pk.sta] = (pk.sta_lon, pk.sta_lat)

    ev_lon = np.array([ev.ev_lon for ev in events.values()], dtype=float)
    ev_lat = np.array([ev.ev_lat for ev in events.values()], dtype=float)

    st_lon = np.array([v[0] for v in sta_dict.values()], dtype=float)
    st_lat = np.array([v[1] for v in sta_dict.values()], dtype=float)

    plt.figure()
    plt.scatter(st_lon, st_lat, s=8, alpha=0.01, label="Stations")
    plt.scatter(ev_lon, ev_lat, s=25, alpha=0.01, marker="*", label="Events")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title("Events and Stations (use lon/lat at end of station line)")
    plt.legend()
    plt.tight_layout()
    if out is not None:
        plt.savefig(out, dpi=200)
    else:
        plt.show()
    plt.close()


def plot_tt_vs_dist(picks: List[Pick], out: Optional[Path] = None) -> None:
    dist = np.array([pk.dist_km for pk in picks], dtype=float)
    tt = np.array([pk.tt for pk in picks], dtype=float)
    phase = np.array([pk.phase for pk in picks])

    plt.figure()
    mP = phase == "P"
    mS = phase == "S"
    if np.any(mP):
        plt.scatter(dist[mP], tt[mP], s=8, label="P")
    if np.any(mS):
        plt.scatter(dist[mS], tt[mS], s=8, label="S")
    plt.xlabel("Epicentral distance (km)")
    plt.ylabel("Travel time (s)")
    plt.title("Travel time vs distance")
    plt.legend()
    plt.tight_layout()
    if out is not None:
        plt.savefig(out, dpi=200)
    else:
        plt.show()
    plt.close()


def plot_pick_counts(events: Dict[int, Event], picks: List[Pick], out: Optional[Path] = None) -> None:
    ev_ids = sorted(events.keys())
    cntP = {eid: 0 for eid in ev_ids}
    cntS = {eid: 0 for eid in ev_ids}

    for pk in picks:
        if pk.phase == "P":
            cntP[pk.event_id] += 1
        elif pk.phase == "S":
            cntS[pk.event_id] += 1

    p_vals = np.array([cntP[eid] for eid in ev_ids], dtype=int)
    s_vals = np.array([cntS[eid] for eid in ev_ids], dtype=int)

    plt.figure()
    plt.hist(p_vals, bins=20, alpha=0.7, label="P counts/event")
    plt.hist(s_vals, bins=20, alpha=0.7, label="S counts/event")
    plt.xlabel("# picks per event")
    plt.ylabel("# events")
    plt.title("Pick counts distribution")
    plt.legend()
    plt.tight_layout()
    if out is not None:
        plt.savefig(out, dpi=200)
    else:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase_file", type=str, default="run_fm3d/data/test.synth_arrivals_skfmm_noise.new.txt", help="Your synthetic phase output text")
    ap.add_argument("--outdir", type=str, default="run_fm3d/data/figs", help="If set, save PNGs to this dir; else show interactively.")
    args = ap.parse_args()

    phase_path = Path(args.phase_file)
    events, picks = read_phase_file(phase_path)

    outdir = Path(args.outdir) if args.outdir else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    plot_map(events, picks, out=None if outdir is None else outdir / "map_events_stations.png")
    plot_tt_vs_dist(picks, out=None if outdir is None else outdir / "tt_vs_dist.png")
    plot_pick_counts(events, picks, out=None if outdir is None else outdir / "pick_counts.png")

    print(f"Parsed events  : {len(events)}")
    print(f"Parsed picks   : {len(picks)}")
    print(f"Unique stations: {len({pk.sta for pk in picks})}")
    if outdir is not None:
        print(f"Wrote figures to: {outdir}")


if __name__ == "__main__":
    main()
