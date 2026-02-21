#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List


HEADER_RE = re.compile(
    r"^\s*(\d+)\s+(\d{4})\s+(\d{2})\s+(\d{2})\s+(\d{2}:\d{2}:\d{2}\.\d+)\s+([+-]?\d+(?:\.\d+)?)\s+"
)
VPVS_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?$")


def parse_header(line: str) -> Optional[Dict[str, Any]]:
    m = HEADER_RE.match(line)
    if not m:
        return None

    event_id = m.group(1)
    yyyy, mm, dd = int(m.group(2)), int(m.group(3)), int(m.group(4))
    hhmmss = m.group(5)
    t0 = float(m.group(6))

    origin_dt = datetime.strptime(f"{yyyy:04d}-{mm:02d}-{dd:02d} {hhmmss}", "%Y-%m-%d %H:%M:%S.%f")
    origin_dt_str = origin_dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    parts = line.strip().split()
    event_vpvs = parts[-1] if parts and VPVS_RE.match(parts[-1]) else None

    return {
        "event_id": event_id,
        "origin_dt": origin_dt,
        "origin_dt_str": origin_dt_str,
        "t0": t0,
        "event_vpvs": event_vpvs,
    }


def parse_pick(line: str) -> Optional[Dict[str, Any]]:
    s = line.strip()
    if not s:
        return None
    parts = s.split()
    if len(parts) < 7:
        return None

    net = parts[0]
    sta = parts[1]
    phase = parts[2]
    try:
        abs_tt = float(parts[3])
        rel_tt = float(parts[4])
    except ValueError:
        return None

    # last three tokens: dist lon lat (based on your example)
    dist_km = sta_lon = sta_lat = None
    if len(parts) >= 3:
        try:
            dist_km = float(parts[-3])
            sta_lon = float(parts[-2])
            sta_lat = float(parts[-1])
        except ValueError:
            pass

    # find station vp/vs token between index 5 and len-3
    sta_vpvs = None
    vpvs_idx = None
    for i in range(5, max(5, len(parts) - 3)):
        if VPVS_RE.match(parts[i]):
            sta_vpvs = parts[i]
            vpvs_idx = i
            break

    # pick_error is first float token after sta_vpvs (or after index 5) before dist/lon/lat
    pick_error = None
    start = (vpvs_idx + 1) if vpvs_idx is not None else 5
    for i in range(start, len(parts) - 3):
        try:
            pick_error = float(parts[i])
            break
        except ValueError:
            continue

    return {
        "Network": net,
        "Station": sta,
        "PhasePick": phase,
        "AbsTT": abs_tt,
        "RelTT": rel_tt,
        "PickError": pick_error,
        "EpiDistKm": dist_km,
        "StaLon": sta_lon,
        "StaLat": sta_lat,
        "StaVPVS": sta_vpvs,
    }


def to_indexed_dict(values: List[Any], fmt=None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, v in enumerate(values):
        if fmt is None:
            out[str(i)] = "" if v is None else str(v)
        else:
            out[str(i)] = fmt(v)
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert arrivals text -> JSON pick dict")
    ap.add_argument("--in_txt", default="run_fm3d/data/loc.synth_arrivals_skfmm_noise.new.txt", help="Input arrivals text file")
    ap.add_argument("--out_json", default="run_fm3d/data/svi/event_picks.json", help="Output JSON path")
    ap.add_argument("--default_pick_error", type=float, default=0.05,
                    help="Used when pick_error cannot be parsed from line")
    ap.add_argument("--keep_extras", action="store_true",
                    help="Also store AbsTT/RelTT/EpiDist/Lon/Lat/VPVS/T0/EventVPVS fields")
    args = ap.parse_args()

    in_path = Path(args.in_txt)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    events_out: Dict[str, Any] = {}

    cur_eid: Optional[str] = None
    cur_origin_dt: Optional[datetime] = None
    cur_t0: Optional[float] = None
    cur_event_vpvs: Optional[str] = None

    # buffers per event
    nets: List[str] = []
    stas: List[str] = []
    phases: List[str] = []
    dts: List[str] = []
    perrs: List[float] = []

    # extras
    abs_tts: List[float] = []
    rel_tts: List[float] = []
    dists: List[Optional[float]] = []
    lons: List[Optional[float]] = []
    lats: List[Optional[float]] = []
    sta_vpvs_list: List[Optional[str]] = []
    t0_list: List[Optional[float]] = []
    ev_vpvs_list: List[Optional[str]] = []

    def flush_current():
        nonlocal cur_eid, nets, stas, phases, dts, perrs
        nonlocal abs_tts, rel_tts, dists, lons, lats, sta_vpvs_list, t0_list, ev_vpvs_list

        if cur_eid is None:
            return

        if len(nets) == 0:
            events_out[cur_eid] = {"Picks": {}}
            return

        picks: Dict[str, Any] = {
            "Network": to_indexed_dict(nets),
            "Station": to_indexed_dict(stas),
            "PhasePick": to_indexed_dict(phases),
            "DT": to_indexed_dict(dts),
            "PickError": to_indexed_dict(perrs, fmt=lambda x: f"{float(x):.6g}"),
        }

        if args.keep_extras:
            picks["AbsTT"] = to_indexed_dict(abs_tts, fmt=lambda x: f"{float(x):.4f}")
            picks["RelTT"] = to_indexed_dict(rel_tts, fmt=lambda x: f"{float(x):.4f}")
            picks["EpiDistKm"] = to_indexed_dict(dists, fmt=lambda x: "" if x is None else f"{float(x):.4f}")
            picks["StaLon"] = to_indexed_dict(lons, fmt=lambda x: "" if x is None else f"{float(x):.4f}")
            picks["StaLat"] = to_indexed_dict(lats, fmt=lambda x: "" if x is None else f"{float(x):.4f}")
            picks["StaVPVS"] = to_indexed_dict(sta_vpvs_list, fmt=lambda x: "" if x is None else str(x))
            picks["T0"] = to_indexed_dict(t0_list, fmt=lambda x: "" if x is None else f"{float(x):.3f}")
            picks["EventVPVS"] = to_indexed_dict(ev_vpvs_list, fmt=lambda x: "" if x is None else str(x))

        events_out[cur_eid] = {"Picks": picks}

        # reset
        nets.clear(); stas.clear(); phases.clear(); dts.clear(); perrs.clear()
        abs_tts.clear(); rel_tts.clear()
        dists.clear(); lons.clear(); lats.clear()
        sta_vpvs_list.clear(); t0_list.clear(); ev_vpvs_list.clear()

    with in_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue

            h = parse_header(raw)
            if h is not None:
                flush_current()
                cur_eid = h["event_id"]
                cur_origin_dt = h["origin_dt"]
                cur_t0 = h["t0"]
                cur_event_vpvs = h["event_vpvs"]
                continue

            # pick lines must follow a header
            if cur_eid is None or cur_origin_dt is None or cur_t0 is None:
                continue

            # parse pick
            pk = parse_pick(raw)
            if pk is None:
                continue

            net = pk["Network"]
            sta = pk["Station"]
            ph = pk["PhasePick"]
            abs_tt = pk["AbsTT"]
            rel_tt = pk["RelTT"]
            pick_error = pk["PickError"]
            if pick_error is None:
                pick_error = float(args.default_pick_error)

            # compute DT: origin_dt + (abs_tt - t0)
            dt = cur_origin_dt + timedelta(seconds=(abs_tt - cur_t0))
            dt_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")

            nets.append(net)
            stas.append(sta)
            phases.append(ph)
            dts.append(dt_str)
            perrs.append(float(pick_error))

            if args.keep_extras:
                abs_tts.append(float(abs_tt))
                rel_tts.append(float(rel_tt))
                dists.append(pk["EpiDistKm"])
                lons.append(pk["StaLon"])
                lats.append(pk["StaLat"])
                sta_vpvs_list.append(pk["StaVPVS"])
                t0_list.append(float(cur_t0))
                ev_vpvs_list.append(cur_event_vpvs)

    # flush last event
    flush_current()

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(events_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote: {out_path}  (events={len(events_out)})")


if __name__ == "__main__":
    main()
