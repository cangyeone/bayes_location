#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import datetime, timedelta
from pathlib import Path
import argparse


def is_event_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    tok0 = s.split()[0]
    return tok0.isdigit()


def parse_event_header(line: str):
    """
    Example:
      0  2022 09 05 00:10:00.000  600.000  0.0000  21.4195  97.8299  0.00  3.00 ...
    """
    parts = line.split()
    idx = int(parts[0])
    year = int(parts[1]); month = int(parts[2]); day = int(parts[3])
    tstr = parts[4]  # HH:MM:SS.sss

    # parse origin time
    origin_dt = datetime.strptime(f"{year:04d}-{month:02d}-{day:02d} {tstr}",
                                  "%Y-%m-%d %H:%M:%S.%f")

    origin_sec = float(parts[5])  # keep for reference
    rms = float(parts[6])
    lat = float(parts[7])
    lon = float(parts[8])
    depth_km = float(parts[9])
    mag = float(parts[10])

    ev = {
        "event_index": idx,
        "OriginTime": origin_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "origin_sec": origin_sec,
        "rms": rms,
        "Latitude": lat,
        "Longitude": lon,
        "DepthKm": depth_km,
        "Magnitude": mag,
        "raw_header": line.rstrip("\n"),
    }
    return origin_dt, ev


def parse_pick_line(line: str):
    """
    Example:
      SC L09875B02280  P  630.0622 30.0622 0.00e+00 0.0000 0.300 180.2776 98.7539 22.8012

    We use:
      Network  = parts[0]
      Station  = parts[1]
      PhasePick= parts[2]
      t_res    = float(parts[4])   # seconds after origin time
      PickError= parts[7]          # e.g. 0.300 / 0.400 (keep as string)
    """
    parts = line.split()
    if len(parts) < 8:
        return None

    net = parts[0]
    sta = parts[1]
    phase = parts[2]

    # residual seconds since origin time
    try:
        t_res = float(parts[4])
    except Exception:
        return None

    pick_err = parts[7]  # keep string (0.300 etc.)

    return net, sta, phase, t_res, pick_err


def real_to_easy_json(real_txt: str, out_json: str, start_id: int = 1000000):
    real_txt = str(real_txt)
    out_json = str(out_json)

    out = {}
    current_origin_dt = None
    current_event = None
    current_picks = None
    cur_id = start_id

    def flush():
        nonlocal cur_id, current_event, current_picks
        if current_event is None:
            return
        out[current_event['event_index']] = {
            "Event": current_event,
            "Picks": current_picks,
            "n_picks": len(current_picks),
        }
        cur_id += 1

    with open(real_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            if is_event_header(line):
                # close previous event
                flush()

                # start new event
                current_origin_dt, current_event = parse_event_header(line)
                current_picks = []
                continue

            # pick line
            if current_origin_dt is None:
                continue

            parsed = parse_pick_line(line)
            if parsed is None:
                continue

            net, sta, phase, t_res, pick_err = parsed
            pick_dt = current_origin_dt + timedelta(seconds=t_res)

            current_picks.append({
                "Network": net,
                "Station": sta,
                "PhasePick": phase,
                "DT": pick_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "PickError": str(pick_err),
            })

    # flush last event
    flush()

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {len(out)} events -> {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert REAL output to easy-to-read Picks JSON.")
    ap.add_argument("--real_txt", default="run_fm3d/data/loc.synth_arrivals_skfmm_noise.new.txt", help="Input REAL text file")
    ap.add_argument("--out_json", default="run_fm3d/data/svi/events_picks.json", help="Output JSON file")
    ap.add_argument("--start-id", type=int, default=1000000, help="Starting event id (default: 1000000)")
    args = ap.parse_args()

    real_to_easy_json(args.real_txt, args.out_json, start_id=args.start_id)
