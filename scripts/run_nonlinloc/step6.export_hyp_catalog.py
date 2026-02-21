#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple


def safe_find(parts: List[str], key: str) -> Optional[int]:
    """Return index of key in parts, or None if not found."""
    try:
        return parts.index(key)
    except ValueError:
        return None


def safe_get_float(parts: List[str], key: str) -> Optional[float]:
    """Find key and parse the next token as float."""
    i = safe_find(parts, key)
    if i is None or i + 1 >= len(parts):
        return None
    try:
        return float(parts[i + 1])
    except ValueError:
        return None


def safe_get_int(parts: List[str], key: str) -> Optional[int]:
    """Find key and parse the next token as int (allow float-like)."""
    v = safe_get_float(parts, key)
    if v is None:
        return None
    return int(v)


def parse_geographic_line(parts: List[str]) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float]]:
    """
    Parse a GEOGRAPHIC line like:
    GEOGRAPHIC  OT  YYYY MM DD HH MM SS.SSS  ... Lat <lat>  Long <lon>  Depth <dep> ...
    """
    # Guard: we need at least YYYY MM DD HH MM SS
    # Typical indices: parts[2:8]
    ot_str = None
    lat = lon = dep = None

    try:
        year = int(parts[2]); mon = int(parts[3]); day = int(parts[4])
        hh = int(parts[5]); mm = int(parts[6]); sec = float(parts[7])
        ot_str = f"{year:04d}-{mon:02d}-{day:02d}T{hh:02d}:{mm:02d}:{sec:06.3f}"
    except Exception:
        ot_str = None

    lat = safe_get_float(parts, "Lat")
    lon = safe_get_float(parts, "Long")
    dep = safe_get_float(parts, "Depth")
    return ot_str, lat, lon, dep


def parse_hyp_file(path: str) -> Dict[str, Optional[float]]:
    """Parse one NLLoc .hyp file into a dict (robust to missing keys)."""
    res: Dict[str, Optional[float]] = {
        "event_id": Path(path).name.replace(".grid0.loc.hyp", ""),
        "ot_str": None,

        "lat_ml": None, "lon_ml": None, "dep_ml": None,
        "lat_exp": None, "lon_exp": None, "dep_exp": None,

        "hmin": None, "hmax": None,
        "ell_len1": None, "ell_len2": None, "ell_len3": None,

        "rms": None, "nphs": None, "gap": None,
    }

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            line = line.strip()
            if not line:
                continue

            if line.startswith("GEOGRAPHIC"):
                parts = line.split()
                ot_str, lat, lon, dep = parse_geographic_line(parts)
                res["ot_str"] = ot_str
                res["lat_ml"] = lat
                res["lon_ml"] = lon
                res["dep_ml"] = dep

            elif line.startswith("STAT_GEOG"):
                # Often contains ExpectLat, Long, Depth etc.
                parts = line.split()
                res["lat_exp"] = safe_get_float(parts, "ExpectLat")
                # some files may use ExpectLong instead of Long; try both
                lon = safe_get_float(parts, "ExpectLong")
                if lon is None:
                    lon = safe_get_float(parts, "Long")
                res["lon_exp"] = lon
                res["dep_exp"] = safe_get_float(parts, "Depth")

            elif line.startswith("STATISTICS"):
                parts = line.split()
                res["ell_len1"] = safe_get_float(parts, "Len1") or res["ell_len1"]
                res["ell_len2"] = safe_get_float(parts, "Len2") or res["ell_len2"]
                res["ell_len3"] = safe_get_float(parts, "Len3") or res["ell_len3"]

            elif line.startswith("QML_OriginUncertainty"):
                parts = line.split()
                res["hmin"] = safe_get_float(parts, "minHorUnc") or res["hmin"]
                res["hmax"] = safe_get_float(parts, "maxHorUnc") or res["hmax"]

            elif line.startswith("QUALITY"):
                parts = line.split()
                res["rms"] = safe_get_float(parts, "RMS") or res["rms"]
                nphs = safe_get_int(parts, "Nphs")
                if nphs is not None:
                    res["nphs"] = nphs
                res["gap"] = safe_get_float(parts, "Gap") or res["gap"]

    return res


def pick_solution(info: Dict[str, Optional[float]], prefer: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Choose (lat, lon, dep) by preference: 'exp'|'ml'|'auto'."""
    lat_ml, lon_ml, dep_ml = info.get("lat_ml"), info.get("lon_ml"), info.get("dep_ml")
    lat_ex, lon_ex, dep_ex = info.get("lat_exp"), info.get("lon_exp"), info.get("dep_exp")

    if prefer == "ml":
        return lat_ml, lon_ml, dep_ml
    if prefer == "exp":
        return lat_ex, lon_ex, dep_ex

    # auto: use exp if available, else ml
    lat = lat_ex if lat_ex is not None else lat_ml
    lon = lon_ex if lon_ex is not None else lon_ml
    dep = dep_ex if dep_ex is not None else dep_ml
    return lat, lon, dep


def fmt(v, default="-1", nd=3) -> str:
    if v is None:
        return default
    if isinstance(v, int):
        return str(v)
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return default


def collect_files(in_dir: str, pattern: str, recursive: bool) -> List[str]:
    in_dir = str(Path(in_dir).expanduser())
    if recursive:
        pat = str(Path(in_dir) / "**" / pattern)
        return sorted(glob.glob(pat, recursive=True))
    else:
        pat = str(Path(in_dir) / pattern)
        return sorted(glob.glob(pat))

#Elapsed time: 1030.7 seconds
def main():
    ap = argparse.ArgumentParser(
        description="Export NLLoc *.hyp locations into a simple catalog file (configurable)."
    )
    ap.add_argument("--in-dir", default="run_fm3d/data/nlloc_sc_crop/out", help="Directory containing *.hyp files.")
    ap.add_argument("--pattern", default="*.sum.grid0.loc.hyp", help="Glob pattern for hyp files.")
    ap.add_argument("--recursive", action="store_true", help="Search hyp files recursively.")
    ap.add_argument("--out", default="run_fm3d/data/nlloc_sc_crop/all.locfiles.csv", help="Output catalog path.")
    ap.add_argument("--format", choices=["txt", "tsv", "csv"], default="csv", help="Output format.")
    ap.add_argument("--prefer", choices=["auto", "exp", "ml"], default="auto",
                    help="Which solution to output: auto=expect if available else ml.")
    #ap.add_argument("--header", action="store_true", help="Write header line.")
    ap.add_argument("--min-nphs", type=int, default=None, help="Filter: keep events with Nphs >= this.")
    ap.add_argument("--max-rms", type=float, default=None, help="Filter: keep events with RMS <= this.")
    ap.add_argument("--max-gap", type=float, default=None, help="Filter: keep events with Gap <= this.")
    ap.add_argument("--drop-missing-ot", action="store_true", help="Drop events missing origin time.")
    ap.add_argument(
        "--no-header",
        dest="header",
        action="store_false",
        help="Do not write header line."
    )
    ap.set_defaults(header=True)
    args = ap.parse_args()

    files = collect_files(args.in_dir, args.pattern, args.recursive)
    if not files:
        print(f"[WARN] No hyp files found: dir={args.in_dir}, pattern={args.pattern}, recursive={args.recursive}")
        return

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # delimiter by format
    if args.format == "csv":
        sep = ","
    elif args.format == "tsv":
        sep = "\t"
    else:
        sep = " "  # keep your original style-like

    # standard columns
    cols = ["event_id", "ot_str", "lat", "lon", "dep_km", "h_err_km", "z_err_km", "rms", "nphs", "gap"]

    kept = 0
    dropped = 0

    with open(out_path, "w", encoding="utf-8", newline="\n") as fout:
        if args.header:
            if args.format == "txt":
                fout.write("# " + " ".join(cols) + "\n")
            else:
                fout.write(sep.join(cols) + "\n")

        for path in files:
            info = parse_hyp_file(path)
            event_id_orig = path.split(".")[0]

            if args.drop_missing_ot and not info.get("ot_str"):
                dropped += 1
                continue

            # filters
            nphs = info.get("nphs")
            rms = info.get("rms")
            gap = info.get("gap")

            if args.min_nphs is not None:
                if nphs is None or nphs < args.min_nphs:
                    dropped += 1
                    continue
            if args.max_rms is not None:
                if rms is None or rms > args.max_rms:
                    dropped += 1
                    continue
            if args.max_gap is not None:
                if gap is None or gap > args.max_gap:
                    dropped += 1
                    continue

            lat, lon, dep = pick_solution(info, args.prefer)

            # uncertainty
            h_err = info.get("hmax")        # horizontal max uncertainty (km)
            z_err = info.get("ell_len3")    # vertical axis length; you used this as Z_err

            if args.format == "txt":
                # preserve your original "EVENT-style" line
                line = (
                    f"#EVENT {event_id_orig} {info['event_id']} {info.get('ot_str','')} "
                    f"{fmt(lat, nd=6)} {fmt(lon, nd=6)} {fmt(dep, nd=3)} "
                    f"{fmt(h_err, nd=3)} {fmt(z_err, nd=3)} "
                    f"{fmt(rms, nd=3)} {str(nphs if nphs is not None else -1)} {fmt(gap, nd=1)}\n"
                )
            else:
                row = [
                    str(info["event_id"]),
                    str(info.get("ot_str") or ""),
                    fmt(lat, default="", nd=6),
                    fmt(lon, default="", nd=6),
                    fmt(dep, default="", nd=3),
                    fmt(h_err, default="", nd=3),
                    fmt(z_err, default="", nd=3),
                    fmt(rms, default="", nd=3),
                    str(nphs if nphs is not None else ""),
                    fmt(gap, default="", nd=1),
                ]
                line = sep.join(row) + "\n"

            fout.write(line)
            kept += 1

    print(f"[OK] Wrote catalog: {out_path}")
    print(f"[OK] Found files: {len(files)}, kept: {kept}, dropped: {dropped}")


if __name__ == "__main__":
    main()
