#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_reloc_to_orig.py

Robust reader for relocation output files that may contain:
  - many comment/debug lines
  - "result lines" beginning with '#'
  - variable number of comma-separated columns

We parse ONLY lines that:
  1) after lstrip() start with '#'
  2) after removing '#', first token (before first comma) is an integer event id

Then we interpret columns as:
  col0: evid (int)
  col1: time string
  col2,col3,col4: relocated lon, lat, z (float)
  last3 cols: original lon0, lat0, z0 (float)

Everything in between is ignored, so it works for both your old 18-col format
and any shortened formats.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.geodetics import gps2dist_azimuth


def _is_int_token(tok: str) -> bool:
    tok = tok.strip()
    if tok.startswith(("+", "-")):
        tok = tok[1:]
    return tok.isdigit()


def parse_result_line_flexible(line: str) -> Optional[Dict]:
    """
    Flexible parser for one result line.

    Accept examples like:
      #0,2022-...,107.49,28.51,10.0,...,97.8299,21.4195,0.0
      #  12,2022-...,...
      ##0, ...   (will also work if you want; we strip leading '#')

    Reject debug lines like:
      # x: [...]
      #something without leading integer id
    """
    s = line.lstrip()
    if not s.startswith("#"):
        return None

    # remove leading #'s and spaces
    s2 = s.lstrip("#").strip()
    if not s2:
        return None

    parts = [x.strip() for x in s2.split(",") if x.strip() != ""]
    if len(parts) < 8:
        # must at least have: id,tstr,lon,lat,z,...,lon0,lat0,z0
        return None

    if not _is_int_token(parts[0]):
        return None

    try:
        evid = int(parts[0])
        tstr = parts[1]

        lon = float(parts[2])
        lat = float(parts[3])
        z = float(parts[4])

        lon0 = float(parts[-3])
        lat0 = float(parts[-2])
        z0 = float(parts[-1])

        err3 = float(parts[-4])
        err2 = float(parts[-5]) 
        err1 = float(parts[-6])
    except Exception:
        return None
    if err1 > 40.0 or err2 > 40.0 or err3 > 80.0:
        return None
    return dict(
        id=evid,
        tstr=tstr,
        lon=lon, lat=lat, z=z,
        lon0=lon0, lat0=lat0, z0=z0,
        ncols=len(parts),
        raw=s2,
        err1=err1,
        err2=err2,
        err3=err3, 
    )


def read_out_file(path: str, debug_n: int = 5) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    records: List[Dict] = []
    n_hash = 0
    first_hash_lines: List[str] = []

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            if line.lstrip().startswith("#"):
                n_hash += 1
                if len(first_hash_lines) < debug_n:
                    first_hash_lines.append(f"[line {ln}] {line.rstrip()}")

            rec = parse_result_line_flexible(line)
            if rec is None:
                continue
            rec["line_no"] = ln
            records.append(rec)

    if not records:
        msg = (
            f"No valid result lines parsed from {path}.\n"
            f"Total lines starting with '#': {n_hash}\n"
            f"First {len(first_hash_lines)} '#' lines:\n"
            + "\n".join(first_hash_lines)
            + "\n\nExpected a result line like:\n"
              "#<id>,<time>,<lon>,<lat>,<z>,...,<orig_lon>,<orig_lat>,<orig_z>\n"
              "If your file is different, paste 5 lines around one event block."
        )
        raise RuntimeError(msg)

    df = pd.DataFrame.from_records(records)
    df = df.dropna(subset=["id", "lon", "lat", "z", "lon0", "lat0", "z0"]).copy()
    df["id"] = df["id"].astype(int)
    return df


def compute_errors(df: pd.DataFrame) -> pd.DataFrame:
    horiz_km = np.empty(len(df), dtype=float)
    for i, r in enumerate(df.itertuples(index=False)):
        dist_m, _, _ = gps2dist_azimuth(r.lat0, r.lon0, r.lat, r.lon)
        horiz_km[i] = dist_m / 1000.0

    out = df.copy()
    out["horiz_err_km"] = horiz_km
    out["depth_err_km"] = out["z"] - out["z0"]
    return out


def print_stats(df: pd.DataFrame) -> None:
    def s(x: pd.Series) -> str:
        q50 = x.quantile(0.5)
        q90 = x.quantile(0.9)
        q95 = x.quantile(0.95)
        return f"N={len(x)} mean={x.mean():.3f} p50={q50:.3f} p90={q90:.3f} p95={q95:.3f} max={x.max():.3f}"

    z_abs_max = float(np.nanmax(np.abs(df["z"].values)))
    if z_abs_max > 200.0:
        print(f"[WARN] relocated z max|z|={z_abs_max:.3f} looks NOT like depth in km (may be grid/local z).")

    print("Horizontal error (km):", s(df["horiz_err_km"]))
    print("Depth error (km):     ", s(df["depth_err_km"]))


def plot_compare(df: pd.DataFrame, save_dir: str = "") -> None:
    save_dir = save_dir.strip()
    outdir = Path(save_dir) if save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    print(len(df), "events to plot.")
    # Epicenter
    plt.figure(figsize=(7, 7))

    plt.scatter(df["lon0"], df["lat0"], s=24, alpha=0.75, label="Original")
    plt.scatter(df["lon"], df["lat"], s=24, alpha=0.75, marker="x", label="Relocated")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Epicenter Comparison")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    if outdir:
        plt.savefig(outdir / "epicenter_compare.png", dpi=200)
    plt.show()

    # Error hist
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(df["horiz_err_km"], bins=30)
    axes[0].set_xlabel("Horizontal error (km)")
    axes[0].set_ylabel("Count")
    axes[0].grid(True)

    axes[1].hist(df["depth_err_km"], bins=30)
    axes[1].set_xlabel("Depth error (km) (z - z0)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True)

    plt.tight_layout()
    if outdir:
        plt.savefig(outdir / "error_hist.png", dpi=200)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="run_fm3d/data/ours/synth.new.True.txt", help="Relocation result file, e.g. bayse_reloc.txt")
    ap.add_argument("--save_dir", default="run_fm3d/run_demo/data/figs", help="If set, save figures to this directory")
    args = ap.parse_args()

    df = read_out_file(args.infile, debug_n=8)
    df = compute_errors(df)

    print(df.head())
    print_stats(df)
    plot_compare(df, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
