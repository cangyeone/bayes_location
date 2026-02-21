#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

def convert_station_file(in_path: str, out_path: str, elev_unit: str = "m"):
    """
    Convert station list:
      input : NET STA LON LAT ELEV
      output: NET STA LAT LON ELEV

    elev_unit:
      - "m": assume elevation already in meters (default)
      - "km": convert km -> m
    """
    in_path = str(in_path)
    out_path = str(out_path)

    scale = 1000.0 if elev_unit.lower() == "km" else 1.0

    lines_out = []
    n_in, n_ok, n_skip = 0, 0, 0

    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            n_in += 1
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 5:
                n_skip += 1
                continue

            net, sta = parts[0], parts[1]

            try:
                lon = float(parts[2])
                lat = float(parts[3])
                elev = float(parts[4]) * scale
            except ValueError:
                n_skip += 1
                continue

            # Output: NET STA LAT LON ELEV
            # lat/lon keep 5 decimals; elev as integer meters
            elev_i = int(round(elev))
            lines_out.append(f"{net} {sta} {lat:.5f} {lon:.5f} {elev_i:d}")
            n_ok += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))

    print(f"[OK] Read {n_in} lines, wrote {n_ok} stations, skipped {n_skip}.")
    print(f"[OK] Output: {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Convert station list format.")
    ap.add_argument("--in_file", default="run_fm3d/data/loc.synth_stations.new.txt", help="Input station file (NET STA LON LAT ELEV)")
    ap.add_argument("--out_file", default="run_fm3d/data/svi/stations.csv", help="Output station file (NET STA LAT LON ELEV)")
    ap.add_argument("--elev-unit", choices=["m", "km"], default="m",
                    help="Elevation unit in input file (default: m)")
    args = ap.parse_args()

    convert_station_file(args.in_file, args.out_file, elev_unit=args.elev_unit)

if __name__ == "__main__":
    main()
