#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step3.make_nll_in_file.py

Goal
----
Generate the three essential NLLoc input files from:
  (1) REAL-format event/pick file
  (2) station coordinate file

Outputs (written under cfg.out_dir)
----------------------------------
1) Station file (NLLoc STATION format)
   - <out_dir>/<station_out_name>   e.g., data/nlloc/sc.stations

2) Observation file (NLLoc NLLOC_OBS format)
   - <out_dir>/<obs_out_name>       e.g., data/nlloc/real.obs

3) NLLoc control file template
   - <out_dir>/<nlloc_in_name>      e.g., data/nlloc/nlloc.in

Design choice for usability
---------------------------
This script does NOT take CLI arguments. All parameters (paths, sigmas, LOCGRID,
TT roots, TRANS reference) come from a single config object (NLLProjectConfig),
which reuses NLLGridConfig from Step2.

Assumptions about REAL format
-----------------------------
- Event header line starts with an integer event_id and includes date/time:
    <event_id> <YYYY> <MM> <DD> <HH:MM:SS[.sss]> ...
- Pick line includes:
    <net> <sta> <phase> ... <rel_t_sec>
  where rel_t_sec is relative to event origin in seconds (float) at index 4.
If your REAL output differs, adjust parse_origin_header() and the pick parsing
in convert_real_to_nll_obs().
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import re


# =============================================================================
# Reuse Step2 config and extend to project-level config for Step3
# =============================================================================

@dataclass
class NLLGridConfig:
    input_npz: str = "nlloc_script/demo_data/velocity_grid_lon97_108_lat21_34_dz2p5.npz"
    lon0: float = 102.5
    lat0: float = 27.5
    earth_radius_km: float = 6371.0
    grid_step_km: float = 5.0
    z_min_km: float = 0.0
    z_max_km: float = 70.0
    #out_basename_p: str = "nlloc_script/demo_data/slow_P_cubic"
    #out_basename_s: str = "nlloc_script/demo_data/slow_S_cubic"
    eps_vel: float = 1e-6
    interp_linear: str = "linear"
    interp_fill: str = "nearest"


@dataclass
class NLLProjectConfig:
    grid: NLLGridConfig = field(default_factory=NLLGridConfig) 

    # Step3 inputs
    real_file: str = "nlloc_script/test_file/test_events.catalog"
    station_file: str = "nlloc_script/test_file/location.txt"

    # Step3 outputs
    out_dir: str = "data/nlloc"
    station_out_name: str = "sc.stations"
    obs_out_name: str = "real.obs"
    nlloc_in_name: str = "nlloc.in"

    # Travel-time grids directory and root names used by NLLoc
    tt_dir: str = "data/nlloc/time"
    ttPS_rootname = "tt_PS"  # travel-time root for P and S
    #ttP_rootname: str = "tt_P"
    #ttS_rootname: str = "tt_S"

    # Pick uncertainties (Gaussian sigma, seconds)
    sigma_p: float = 0.05
    sigma_s: float = 0.10

    # TRANS SIMPLE reference point:
    # - True: use mean station lat/lon
    # - False: use cfg.grid.lat0/cfg.grid.lon0
    trans_ref_use_station_mean: bool = True

    # NLLoc search grid (LOCGRID) in local XYZ (km)
    loc_x_min: float = -80.0
    loc_x_max: float = 80.0
    loc_dx: float = 2.5
    loc_y_min: float = -80.0
    loc_y_max: float = 80.0
    loc_dy: float = 2.5
    loc_z_min: float = 0.0
    loc_z_max: float = 40.0
    loc_dz: float = 2.5
    slow_p_root: str = "slow_P_cubic"
    slow_s_root: str = "slow_S_cubic"
    # NLLoc output prefix relative to out_dir (NLLoc will append event IDs etc.)
    loc_out_prefix: str = "out/sc_loc_"
    # LOCGRID strategy:
    # - True: derive LOCGRID from travel-time grid header (*.hdr)
    # - False: use loc_x_min/... from config
    locgrid_from_tt_hdr: bool = True

    # Which travel-time grid to read for LOCGRID (usually P is enough)
    # Root name should correspond to <tt_dir>/<ttP_rootname>.hdr
    locgrid_hdr_phase: str = "S"   # "P" or "S"


def hdr_params_to_locgrid_line(params: dict,
                               grid_type: str = "PROB_DENSITY",
                               save_flag: str = "SAVE") -> str:
    """
    Convert parsed .hdr params to NLLoc LOCGRID line (numeric form).

    Expected params keys:
      nx, ny, nz (int)
      x_orig, y_orig, z_orig (float)
      dx, dy, dz (float)
    """
    nx, ny, nz = int(params["nx"]), int(params["ny"]), int(params["nz"])
    x0, y0, z0 = float(params["x_orig"]), float(params["y_orig"]), float(params["z_orig"])
    dx, dy, dz = float(params["dx"]), float(params["dy"]), float(params["dz"])

    return (f"LOCGRID {nx} {ny} {nz}  "
            f"{x0:.6f} {y0:.6f} {z0:.6f}  "
            f"{dx:.6f} {dy:.6f} {dz:.6f}  "
            f"{grid_type} {save_flag}\n\n")

def read_nll_hdr_params(hdr_path: str) -> Dict[str, float]:
    """
    Parse an NLLoc-style *.hdr file and extract key grid geometry parameters.

    Supports two common formats:
    1) Numeric-first-line format (most common):
       nx ny nz x_orig y_orig z_orig dx dy dz [extra tokens...]
    2) Token format containing keys like NX/XORIG/DX...
    """
    text = Path(hdr_path).read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # ------------------------------
    # (A) Preferred: numeric first line
    # ------------------------------
    if lines:
        # Extract numeric tokens from the first non-empty line
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", lines[0])
        if len(nums) >= 9:
            nx, ny, nz = map(int, nums[0:3])
            x_orig, y_orig, z_orig = map(float, nums[3:6])
            dx, dy, dz = map(float, nums[6:9])
            return {
                "nx": nx, "ny": ny, "nz": nz,
                "x_orig": x_orig, "y_orig": y_orig, "z_orig": z_orig,
                "dx": dx, "dy": dy, "dz": dz,
            }

    # ------------------------------
    # (B) Fallback: token-based format (your old logic)
    # ------------------------------
    norm = re.sub(r"[=\t]", " ", text)
    norm = re.sub(r"\s+", " ", norm)

    patterns = {
        "nx": [r"\bNX\b\s*([0-9]+)"],
        "ny": [r"\bNY\b\s*([0-9]+)"],
        "nz": [r"\bNZ\b\s*([0-9]+)"],

        "x_orig": [r"\bXORIG\b\s*([-+0-9.eE]+)"],
        "y_orig": [r"\bYORIG\b\s*([-+0-9.eE]+)"],
        "z_orig": [r"\bZORIG\b\s*([-+0-9.eE]+)"],

        "dx": [r"\bDX\b\s*([-+0-9.eE]+)"],
        "dy": [r"\bDY\b\s*([-+0-9.eE]+)"],
        "dz": [r"\bDZ\b\s*([-+0-9.eE]+)"],
    }

    out: Dict[str, float] = {}
    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, norm, flags=re.IGNORECASE)
            if m:
                out[key] = float(m.group(1))
                break

    required = ["nx", "ny", "nz", "x_orig", "y_orig", "z_orig", "dx", "dy", "dz"]
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(
            f"Failed to parse {missing} from hdr: {hdr_path}. "
            "Your .hdr appears not to contain NX/XORIG/DX tokens, and the first-line numeric parse also failed."
        )

    out["nx"] = int(out["nx"])
    out["ny"] = int(out["ny"])
    out["nz"] = int(out["nz"])
    return out

def hdr_params_to_locgrid(params: Dict[str, float]) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Convert header geometry parameters to LOCGRID bounds.

    NLLoc grid convention (common)
    ------------------------------
    Grid node coordinates:
        x(i) = x_orig + i * dx,  i = 0..nx-1
    Similarly for y and z.

    Therefore bounds are:
        x_min = x_orig
        x_max = x_orig + (nx-1)*dx
        dx = dx
    etc.

    Returns
    -------
    x_min, x_max, dx, y_min, y_max, dy, z_min, z_max, dz
    """
    nx, ny, nz = int(params["nx"]), int(params["ny"]), int(params["nz"])
    x0, y0, z0 = float(params["x_orig"]), float(params["y_orig"]), float(params["z_orig"])
    dx, dy, dz = float(params["dx"]), float(params["dy"]), float(params["dz"])

    x_min = x0
    x_max = x0 + (nx - 1) * dx
    y_min = y0
    y_max = y0 + (ny - 1) * dy
    z_min = z0
    z_max = z0 + (nz - 1) * dz

    return x_min, x_max, dx, y_min, y_max, dy, z_min, z_max, dz


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _split_csv_or_space(row: List[str]) -> List[str]:
    """
    Normalize a row that may be CSV or a single whitespace-delimited string.

    - If csv.reader returns one cell, treat it as whitespace-delimited.
    - Otherwise strip cells and drop empties.
    """
    if not row:
        return []
    if len(row) == 1:
        return row[0].split()
    return [x.strip() for x in row if x and x.strip()]


# =============================================================================
# Station file I/O
# =============================================================================

def read_stations(station_file: str) -> Tuple[Dict[str, dict], float, float]:
    """
    Read station coordinate file and compute mean lat/lon.

    Expected minimal columns (CSV or whitespace):
        net, sta, lon, lat, elev_km

    Returns
    -------
    stations : dict keyed by station code:
        stations[sta] = {"net": net, "lat": lat, "lon": lon, "elev": elev_km}
    lat0_mean, lon0_mean : mean station latitude/longitude (degrees)

    Notes
    -----
    - elev_km is written to station file as the last column.
      Keep consistent with your station convention in NLLoc.
    """
    stations: Dict[str, dict] = {}
    lats: List[float] = []
    lons: List[float] = []

    with open(station_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            parts = _split_csv_or_space(row)
            if len(parts) < 5:
                continue

            net = parts[0]
            sta = parts[1]
            lon = float(parts[2])
            lat = float(parts[3])
            elev = float(parts[4])  # km

            stations[sta] = dict(net=net, lat=lat, lon=lon, elev=elev)
            lats.append(lat)
            lons.append(lon)

    if not stations:
        raise RuntimeError(
            f"No stations parsed from: {station_file}. "
            "Expected format: net sta lon lat elev_km (CSV or whitespace)."
        )

    lat0 = sum(lats) / len(lats)
    lon0 = sum(lons) / len(lons)
    return stations, lat0, lon0


def write_nlloc_station_file(stations: Dict[str, dict], out_file: str) -> None:
    """
    Write NLLoc station file.

    Output format (LATLON)
    ----------------------
    STATION  <STA>  ?  LATLON  <lat> <lon>  <z>  <elev>

    Here we set:
    - z = 0.0 km (station depth relative to reference surface)
    - elev = station elevation in km from input

    Many NLLoc examples use '?' as the third token; it can represent component/alias.
    """
    ensure_dir(str(Path(out_file).parent))

    lines: List[str] = []
    for sta, info in stations.items():
        lat = info["lat"]
        lon = info["lon"]
        elev = info["elev"]
        line = (
            f"STATION  {sta:<6s} ?  LATLON  "
            f"{lat:9.4f} {lon:9.4f}  0.0  {elev:7.3f}"
        )
        lines.append(line)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# NLLoc station file generated by step3.make_nll_in_file.py\n")
        for line in lines:
            f.write(line + "\n")
        f.write("END_STATION\n")


# =============================================================================
# REAL -> NLLoc OBS conversion
# =============================================================================

def parse_origin_header(line: str) -> Tuple[int, datetime, Optional[float]]:
    """
    Parse a REAL event header line:
        <event_id> <YYYY> <MM> <DD> <HH:MM:SS[.sss]> <origin_sec> ...

    Returns
    -------
    event_id : int
    origin_dt : datetime
    origin_sec : float or None
        The absolute "seconds" used by REAL picks (same scale as pick abs times),
        e.g. 58240.000 in your sample.
    """
    parts = line.strip().split()
    if len(parts) < 6:
        raise ValueError(f"Invalid REAL event header (need >=6 cols): {line}")

    event_id = int(parts[0])
    year = int(parts[1])
    month = int(parts[2])
    day = int(parts[3])
    time_str = parts[4]

    try:
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
    except ValueError:
        t = datetime.strptime(time_str, "%H:%M:%S")

    origin_dt = datetime(year, month, day, t.hour, t.minute, t.second, t.microsecond)

    origin_sec = None
    try:
        origin_sec = float(parts[5])
    except Exception:
        origin_sec = None

    return event_id, origin_dt, origin_sec


def _format_dt_msec(dt: datetime) -> str:
    """Format datetime as 'YYYY MM DD HH MM SS.sss' using milliseconds."""
    ms = int(dt.microsecond / 1000)
    return (
        f"{dt.year:4d} {dt.month:02d} {dt.day:02d}  "
        f"{dt.hour:02d} {dt.minute:02d} "
        f"{dt.second:02d}.{ms:03d}"
    )


def _event_key_from_origin(origin_dt: datetime, event_id: int) -> str:
    """
    Build a unique EventID string for NLLOC_OBS.
    Example: 202209050330330001
      - prefix: YYYYMMDDHHMMSSmmm (millisecond)
      - suffix: event_id as 4 digits (or 6 digits) to avoid collision
    Adjust width to your needs.
    """
    # YYYYMMDDHHMMSSmmm
    prefix = origin_dt.strftime("%Y%m%d%H%M%S") + f"{int(origin_dt.microsecond/1000):03d}"
    # append event_id (4 digits). You can use 6 digits if you want.
    return f"{prefix}_{event_id:08d}"

def convert_real_to_nll_obs(
    real_file: str,
    out_obs: str,
    sigma_p: float,
    sigma_s: float,
    default_sigma: Optional[float] = None,
    stations: Optional[Dict[str, dict]] = None,   # NEW
) -> int:
    """
    Convert REAL picks into NLLoc NLLOC_OBS file (one pick per line).

    Key fixes for your REAL format:
    - Use event-header origin_sec (e.g., 58240.000) to validate/derive rel_t.
    - Station code in OBS is aligned to station file keys when possible
      (avoid writing SC_xxx if station file only has xxx).
    - Phase token is detected robustly (P/S/Pg/Pn/Sg/Sn...).
    """
    ensure_dir(str(Path(out_obs).parent))

    if default_sigma is None:
        default_sigma = sigma_s

    station_keys = set(stations.keys()) if stations else set()

    with open(real_file, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    out_lines: List[str] = []
    current_event_id: Optional[int] = None
    current_origin_dt: Optional[datetime] = None
    current_origin_sec: Optional[float] = None
    event_count = 0

    # Acceptable phase tokens (expand if you need)
    phase_pat = re.compile(r"^(P|S|PG|PN|SG|SN)$", re.IGNORECASE)

    for line in lines:
        if not line.strip():
            continue

        first_token = line.strip().split()[0]

        # Event header
        if first_token.isdigit():
            current_event_id, current_origin_dt, current_origin_sec = parse_origin_header(line)
            event_count += 1
            continue

        # Pick lines require an active event
        if current_origin_dt is None or current_event_id is None:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        net = parts[0].strip()
        sta0 = parts[1].strip()

        # -------- phase: robust detection --------
        # In your sample: parts[2] is P/S.
        # But we scan a bit to be robust.
        phase = None
        phase_idx = None
        for i in range(2, min(len(parts), 6)):
            if phase_pat.match(parts[i].strip()):
                phase = parts[i].strip().upper()
                phase_idx = i
                break
        if phase is None:
            continue

        # -------- time: abs_sec + rel_sec --------
        # Your sample: abs at parts[3], rel at parts[4].
        abs_sec = None
        rel_t = None

        # try the "canonical" columns first (right after phase)
        # expected layout: net sta phase abs rel ...
        try:
            abs_sec = float(parts[phase_idx + 1])
            rel_t = float(parts[phase_idx + 2])
        except Exception:
            abs_sec = None
            rel_t = None

        # fallback: try old assumption (phase at 2, rel at 4)
        if rel_t is None:
            try:
                rel_t = float(parts[4])
            except Exception:
                rel_t = None

        # validate/repair rel_t using origin_sec if possible
        if rel_t is not None and abs_sec is not None and current_origin_sec is not None:
            rel_from_abs = abs_sec - current_origin_sec
            # if mismatch is large, prefer the consistent one
            if abs(rel_from_abs - rel_t) > 1e-2:
                rel_t = rel_from_abs
        elif rel_t is None and abs_sec is not None and current_origin_sec is not None:
            rel_t = abs_sec - current_origin_sec

        if rel_t is None:
            continue

        arr_dt = current_origin_dt + timedelta(seconds=float(rel_t))

        # -------- station code: align to station file --------
        # Prefer sta0 (e.g. L10457B03245) if it exists in station list.
        # Only add net_ prefix if that is what your station file uses.
        if sta0 in station_keys:
            sta = sta0
        elif f"{net}_{sta0}" in station_keys:
            sta = f"{net}_{sta0}"
        else:
            # safest default: keep sta0 (avoid injecting SC_ that breaks matching)
            sta = sta0

        # -------- phase normalization + sigma --------
        if phase.startswith("P"):
            std_phase = "P"
            sigma = sigma_p
        elif phase.startswith("S"):
            std_phase = "S"
            sigma = sigma_s
        else:
            std_phase = phase[:1]
            sigma = default_sigma

        ymd = arr_dt.strftime("%Y%m%d")
        hrmn = arr_dt.strftime("%H%M")
        sec = arr_dt.second + arr_dt.microsecond / 1e6

        evkey = _event_key_from_origin(current_origin_dt, current_event_id)

        line_out = (
            f"SC_{sta:8s}\t?\t?\t?\t{std_phase:1s}\t0\t"
            f"{ymd}\t{hrmn}\t{sec:8.3f}\t"
            f"GAU\t{sigma:8.2e}\t0.00e+00\t"
            f"0.00e+00\t2.00e-02\t1\t>\t-1\t-1\t{evkey}\n"
        )
        out_lines.append(line_out)
    print("[NOTICE], the net name is set to 'SC_' as default")
    with open(out_obs, "w", encoding="utf-8", newline="\n") as f:
        f.write("# NLLOC_OBS file generated from REAL format\n")
        f.writelines(out_lines)

    return event_count


import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

def convert_crop_to_nll_obs(
    real_file: str,
    out_obs: str,
    sigma_p: float,
    sigma_s: float,
    default_sigma: Optional[float] = None,
    stations: Optional[Dict[str, dict]] = None,   # NEW
) -> int:
    """
    Convert REAL picks into NLLoc NLLOC_OBS file (one pick per line).

    Supports:
      (A) Old whitespace REAL format (your existing)
      (B) New CSV-like format:
          #...,origin_time,...
          Pg,Pg,1.0,1.8497,phase_time,...,SC.LDXX

    NEW RULES for CSV picks (per your request):
      - Only use the FIRST phase token (cols[0]) and map to P/S.
      - DO NOT use cols[3] for rel_t.
      - rel_t = phase_time(cols[4]) - origin_time(header cols[1]).
      - Drop picks with rel_t > 50 s (also drop rel_t < 0).
    """
    ensure_dir(str(Path(out_obs).parent))

    if default_sigma is None:
        default_sigma = sigma_s

    station_keys = set(stations.keys()) if stations else set()

    with open(real_file, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    out_lines: List[str] = []
    current_event_id: Optional[int] = None
    current_origin_dt: Optional[datetime] = None
    current_origin_sec: Optional[float] = None
    event_count = 0

    # Old-format acceptable phase tokens
    phase_pat_old = re.compile(r"^(P|S|PG|PN|SG|SN)$", re.IGNORECASE)

    def _origin_sec_of_day(dt: datetime) -> float:
        return (
            dt.hour * 3600.0
            + dt.minute * 60.0
            + dt.second
            + dt.microsecond / 1e6
        )

    def parse_origin_header_csv(line: str) -> Tuple[int, datetime, float]:
        """
        CSV header:
          #LD.20220905033033730000,2022-09-05 03:30:33.730000,lon,lat,dep,...
        """
        s = line.strip()
        if s.startswith("#"):
            s = s[1:]
        cols = [c.strip() for c in s.split(",")]
        try:
            origin_dt = datetime.strptime(cols[1], "%Y-%m-%d %H:%M:%S.%f")
        except:
            origin_dt = datetime.strptime(cols[1], "%Y-%m-%d %H:%M:%S")
        origin_sec = _origin_sec_of_day(origin_dt)

        nonlocal event_count
        event_id = event_count + 1  # keep consistent counting
        return event_id, origin_dt, origin_sec

    def phase_to_ps(phase_token: str) -> Optional[str]:
        """
        Map any Pg/Pn/P... -> P ; any Sg/Sn/S... -> S.
        Only accept tokens that start with P or S (case-insensitive).
        """
        if not phase_token:
            return None
        c = phase_token.strip().upper()[:1]
        if c == "P":
            return "P"
        if c == "S":
            return "S"
        return None

    for line in lines:
        if not line.strip():
            continue

        sline = line.strip()

        # (B) CSV event header
        if sline.startswith("#"):
            current_event_id, current_origin_dt, current_origin_sec = parse_origin_header_csv(sline)
            event_count += 1
            continue

        # (A) Old whitespace event header
        first_token = sline.split()[0]
        if first_token.isdigit():
            current_event_id, current_origin_dt, current_origin_sec = parse_origin_header(line)
            event_count += 1
            continue

        # Need active event
        if current_origin_dt is None or current_event_id is None:
            continue

        # Decide parsing mode
        is_csv = ("," in sline) and (len(sline.split(",")) >= 6)

        if is_csv:
            # CSV pick line:
            # Pg,Pg,1.0,1.8497,2022-09-05 03:30:35.579700,...,SC.LDXX
            cols = [c.strip() for c in sline.split(",")]
            if len(cols) < 6:
                continue

            # 1) phase from FIRST token only
            std_phase = phase_to_ps(cols[0])
            if std_phase is None:
                continue

            # 2) compute rel_t from phase absolute time - origin time
            try:
                phase_dt = datetime.strptime(cols[4], "%Y-%m-%d %H:%M:%S.%f")
            except Exception:
                continue

            rel_t = (phase_dt - current_origin_dt).total_seconds()

            # 3) drop unreasonable picks
            if rel_t < 0:
                continue
            if rel_t > 50.0:
                continue

            arr_dt = current_origin_dt + timedelta(seconds=float(rel_t))

            # 4) station from last column "SC.LDXX" -> net="SC", sta0="LDXX"
            sta_token = cols[-1]
            net = ""
            sta0 = sta_token
            if "." in sta_token:
                net, sta0 = sta_token.split(".", 1)

        else:
            # Old whitespace pick line (your original logic; still uses rel_t columns)
            parts = line.split()
            if len(parts) < 5:
                continue

            net = parts[0].strip()
            sta0 = parts[1].strip()

            phase = None
            phase_idx = None
            for i in range(2, min(len(parts), 6)):
                if phase_pat_old.match(parts[i].strip()):
                    phase = parts[i].strip().upper()
                    phase_idx = i
                    break
            if phase is None:
                continue

            abs_sec = None
            rel_t = None
            try:
                abs_sec = float(parts[phase_idx + 1])
                rel_t = float(parts[phase_idx + 2])
            except Exception:
                abs_sec = None
                rel_t = None

            if rel_t is None:
                try:
                    rel_t = float(parts[4])
                except Exception:
                    rel_t = None

            if rel_t is not None and abs_sec is not None and current_origin_sec is not None:
                rel_from_abs = abs_sec - current_origin_sec
                if abs(rel_from_abs - rel_t) > 1e-2:
                    rel_t = rel_from_abs
            elif rel_t is None and abs_sec is not None and current_origin_sec is not None:
                rel_t = abs_sec - current_origin_sec

            if rel_t is None:
                continue
            if rel_t < 0:
                continue
            if rel_t > 50.0:
                continue

            arr_dt = current_origin_dt + timedelta(seconds=float(rel_t))

            std_phase = phase_to_ps(phase)
            if std_phase is None:
                continue

        # station code align (same as before)
        if sta0 in station_keys:
            sta = sta0
        elif f"{net}_{sta0}" in station_keys:
            sta = f"{net}_{sta0}"
        else:
            sta = sta0

        # sigma by phase
        if std_phase == "P":
            sigma = sigma_p
        else:
            sigma = sigma_s

        ymd = arr_dt.strftime("%Y%m%d")
        hrmn = arr_dt.strftime("%H%M")
        sec = arr_dt.second + arr_dt.microsecond / 1e6

        evkey = _event_key_from_origin(current_origin_dt, current_event_id)

        line_out = (
            f"SC_{sta:8s}\t?\t?\t?\t{std_phase:1s}\t0\t"
            f"{ymd}\t{hrmn}\t{sec:8.3f}\t"
            f"GAU\t{sigma:8.2e}\t0.00e+00\t"
            f"0.00e+00\t2.00e-02\t1\t>\t-1\t-1\t{evkey}\n"
        )
        out_lines.append(line_out)

    print("[NOTICE], the net name is set to 'SC_' as default")
    with open(out_obs, "w", encoding="utf-8", newline="\n") as f:
        f.write("# NLLOC_OBS file generated from REAL format\n")
        f.writelines(out_lines)

    return event_count



# =============================================================================
# NLLoc control file writer
# =============================================================================

def write_nlloc_in_file(cfg: NLLProjectConfig, lat0: float, lon0: float) -> None:
    out_path = Path(cfg.out_dir) / cfg.nlloc_in_name
    ensure_dir(str(out_path.parent))

    # --- obs / out_root / tt_root ---
    obs_file = (Path(cfg.out_dir) / cfg.obs_out_name).resolve().as_posix()

    out_root_dir = (Path(cfg.out_dir) / getattr(cfg, "loc_out_dir", "out")).resolve()
    ensure_dir(str(out_root_dir))
    out_root = (out_root_dir / getattr(cfg, "loc_out_prefix", "sc_loc")).resolve().as_posix()

    tt_rootname = getattr(cfg, "ttPS_rootname", None)
    if tt_rootname is None:
        raise RuntimeError("Need cfg.ttPS_rootname, e.g., 'tt_PS'.")

    tt_root = (Path(cfg.tt_dir) / tt_rootname).resolve().as_posix()

    # --- LOCGRID：从走时表 hdr 获取（tt_PS.S.mod.hdr / tt_PS.P.mod.hdr）---
    if getattr(cfg, "locgrid_from_tt_hdr", True):
        # 你已确认存在：.../time/tt_PS.S.mod.hdr
        phase = str(getattr(cfg, "locgrid_hdr_phase", "S")).upper()
        if phase not in ("P", "S"):
            phase = "S"

        # primary candidate
        hdr_path = tt_root + f".{phase}.mod.hdr"

        # fallback candidate (other phase)
        if not Path(hdr_path).is_file():
            other = "P" if phase == "S" else "S"
            hdr_path2 = tt_root + f".{other}.mod.hdr"
            if Path(hdr_path2).is_file():
                hdr_path = hdr_path2

        # optional extra fallback: some builds may output tt_root.hdr (rare)
        if not Path(hdr_path).is_file():
            hdr_path3 = tt_root + ".hdr"
            if Path(hdr_path3).is_file():
                hdr_path = hdr_path3

        if not Path(hdr_path).is_file():
            raise RuntimeError(
                "Cannot find hdr for LOCGRID. Tried:\n"
                f"  - {tt_root}.S.mod.hdr\n"
                f"  - {tt_root}.P.mod.hdr\n"
                f"  - {tt_root}.hdr\n"
                f"(tt_root={tt_root})"
            )

        params = read_nll_hdr_params(hdr_path)
        locgrid_line = hdr_params_to_locgrid_line(params, grid_type="PROB_DENSITY", save_flag="SAVE")
    else:
        # 手动指定数字 LOCGRID（与你模板一致的格式）
        x0, y0, z0 = cfg.loc_x_orig, cfg.loc_y_orig, cfg.loc_z_orig
        nx, ny, nz = cfg.loc_nx, cfg.loc_ny, cfg.loc_nz
        dx, dy, dz = cfg.loc_dx, cfg.loc_dy, cfg.loc_dz
        locgrid_line = (
            f"LOCGRID {nx} {ny} {nz}  "
            f"{x0:.6f} {y0:.6f} {z0:.6f}  "
            f"{dx:.6f} {dy:.6f} {dz:.6f}  "
            "PROB_DENSITY SAVE\n\n"
        )

    # --- write file with template spacing/comments ---
    with open(str(out_path), "w", encoding="utf-8", newline="\n") as f:
        f.write("CONTROL 1 54321\n")
        f.write(f"TRANS SIMPLE {lat0:.6f} {lon0:.6f} 0.0\n\n")

        f.write("LOCSIG  SC_catalog_test\n")
        f.write("LOCCOM  Sichuan REAL catalog located with NLLoc (P-only first test)\n\n")

        f.write("# LOCFILES obsFiles obsFileType ttimeFileRoot outputFileRoot [iSwapBytes]\n")
        f.write(f"LOCFILES {obs_file} NLLOC_OBS {tt_root} {out_root}\n\n\n")

        f.write("LOCHYPOUT SAVE_NLLOC_ALL NLL_FORMAT_VER_2 SAVE_HYPOINV_SUM\n\n")

        f.write("# LOCSEARCH OCT initNumCells_x initNumCells_y initNumCells_z minNodeSize "
                "maxNumNodes numScatter useStationsDensity stopOnMinNodeSize\n")
        f.write("LOCSEARCH OCT 10 10 4 0.01 20000 5000 0 1\n\n")

        f.write("# LOCGRID xNum yNum zNum xOrig yOrig zOrig dx dy dz gridType saveFlag\n")
        f.write("# 水平约 (-80,80) km，深度 0–40 km，步长 2.5 km\n")
        f.write(locgrid_line)

        f.write("# 定位方法：直接用官方例子\n")
        f.write("LOCMETH EDT_OT_WT 9999.0 4 -1 -1 0.0 6 -1.0 1\n\n")

        f.write("LOCGAU 0.2 0.0\n")
        f.write("LOCGAU2 0.02 0.05 2.0\n\n")

        f.write("# 相位名映射：你的 real.obs 里应该只有 P / S，这样就够了\n")
        f.write("LOCPHASEID  P   P p Pg Pn\n")
        f.write("LOCPHASEID  S   S s Sg Sn\n\n")

        f.write("LOCQUAL2ERR 0.1 0.5 1.0 2.0 99999.9\n\n")

        f.write("# 先不需要 take-off angle，就关掉\n")
        f.write("LOCANGLES ANGLES_NO 5\n")



def write_nlloc_temp_in_file(cfg: "NLLProjectConfig", lat0: float, lon0: float) -> None:
    """
    Write an NLLoc control file in the 'template' style (nlloc.temp.in), e.g.:

      CONTROL ...
      TRANS SIMPLE <lat0> <lon0> 0.0
      LOCFILES <obsFiles> NLLOC_OBS <ttimeFileRoot> <outputFileRoot>
      LOCSEARCH OCT 10 10 4 0.01 20000 5000 0 1
      LOCGRID xNum yNum zNum xOrig yOrig zOrig dx dy dz PROB_DENSITY SAVE
      LOCMETH ...
      LOCPHASEID ...
      LOCQUAL2ERR ...
      LOCANGLES NO

    Compared with your old writer:
    - No fragile '\' line continuation
    - LOCFILES uses official multi-arg form (obs + obsType + ttimeRoot + outRoot)
    """

    out_path = Path(cfg.out_dir) / getattr(cfg, "nlloc_temp_in_name", "nlloc.temp.in")
    ensure_dir(str(out_path.parent))

    # --- Paths (absolute is safer for automation) ---
    obs_file = (Path(cfg.out_dir) / cfg.obs_out_name).resolve().as_posix()

    # output root/prefix (no extension; NLLoc will append)
    # e.g. .../out/sc_loc  (you can add suffix outside)
    out_root_dir = (Path(cfg.out_dir) / getattr(cfg, "loc_out_dir", "out")).resolve()
    ensure_dir(str(out_root_dir))
    out_file_root = (out_root_dir / getattr(cfg, "loc_out_prefix", "sc_loc")).resolve().as_posix()

    # travel-time root: prefer cfg.ttPS_root or cfg.ttPS_rootname, else fall back
    # This must match how you produced travel-time files.
    if hasattr(cfg, "ttPS_root"):
        ttime_root = Path(cfg.ttPS_root).resolve().as_posix()
    elif hasattr(cfg, "ttPS_rootname"):
        ttime_root = (Path(cfg.tt_dir) / cfg.ttPS_rootname).resolve().as_posix()
    else:
        # fallback default: <tt_dir>/tt_PS
        ttime_root = (Path(cfg.tt_dir) / "tt_PS").resolve().as_posix()

    # --- LOCGRID: derive from hdr if requested ---
    # Template LOCGRID is: xNum yNum zNum xOrig yOrig zOrig dx dy dz gridType saveFlag
    if getattr(cfg, "locgrid_from_tt_hdr", True):
        # pick which hdr to read (P/S doesn't matter for geometry; we just need grid dims/orig/dx)
        # In your earlier code, hdr_root was something like slow_P_cubic.P.mod.hdr
        phase = getattr(cfg, "locgrid_hdr_phase", "P").upper()
        if phase == "S":
            hdr_path = os.path.join(cfg.tt_dir, cfg.ttPS_rootname + ".S.mod.hdr")
        else:
            hdr_path = os.path.join(cfg.tt_dir, cfg.ttPS_rootname + ".S.mod.hdr")

        params = read_nll_hdr_params(hdr_path)
        # params: nx,ny,nz,x_orig,y_orig,z_orig,dx,dy,dz
        xNum, yNum, zNum = int(params["nx"]), int(params["ny"]), int(params["nz"])
        xOrig, yOrig, zOrig = float(params["x_orig"]), float(params["y_orig"]), float(params["z_orig"])
        dx, dy, dz = float(params["dx"]), float(params["dy"]), float(params["dz"])
    else:
        # Manual fallback:
        # If you only have min/max + d, we convert to counts and origin=min.
        # xNum = round((x_max-x_min)/dx)+1
        x_min, x_max, dx = cfg.loc_x_min, cfg.loc_x_max, cfg.loc_dx
        y_min, y_max, dy = cfg.loc_y_min, cfg.loc_y_max, cfg.loc_dy
        z_min, z_max, dz = cfg.loc_z_min, cfg.loc_z_max, cfg.loc_dz
        xNum = int(round((x_max - x_min) / dx)) + 1
        yNum = int(round((y_max - y_min) / dy)) + 1
        zNum = int(round((z_max - z_min) / dz)) + 1
        xOrig, yOrig, zOrig = float(x_min), float(y_min), float(z_min)

    grid_type = getattr(cfg, "locgrid_type", "PROB_DENSITY")
    grid_save = getattr(cfg, "locgrid_save", "SAVE")

    # --- LOCSEARCH OCT defaults (you can expose to cfg) ---
    # LOCSEARCH OCT initNumCells_x initNumCells_y initNumCells_z minNodeSize maxNumNodes numScatter useStationsDensity stopOnMinNodeSize
    locsearch = getattr(cfg, "locsearch_oct", None)
    if locsearch is None:
        # Reasonable defaults; safer than your old "20000 4 4 4 4" line
        initx = getattr(cfg, "oct_init_x", 10)
        inity = getattr(cfg, "oct_init_y", 10)
        initz = getattr(cfg, "oct_init_z", 4)
        minNodeSize = getattr(cfg, "oct_min_node_km", 0.01)
        maxNumNodes = getattr(cfg, "oct_max_nodes", 20000)
        numScatter = getattr(cfg, "oct_num_scatter", 5000)
        useStaDen = getattr(cfg, "oct_use_station_density", 0)
        stopOnMin = getattr(cfg, "oct_stop_on_min_node", 1)
        locsearch_line = f"LOCSEARCH OCT {initx} {inity} {initz} {minNodeSize} {maxNumNodes} {numScatter} {useStaDen} {stopOnMin}\n"
    else:
        # allow cfg.locsearch_oct to be a ready-made string tail
        locsearch_line = f"LOCSEARCH OCT {locsearch}\n"

    # --- LOCMETH / LOCQUAL2ERR / GAU ---
    locmeth_line = getattr(cfg, "locmeth_line", "LOCMETH EDT_OT_WT 9999.0 4 -1 -1 0.0 6 -1.0 1\n")
    locqual2err_line = getattr(cfg, "locqual2err_line", "LOCQUAL2ERR 0.1 0.5 1.0 2.0 99999.9\n")

    locgau_line = getattr(cfg, "locgau_line", "LOCGAU 0.2 0.0\n")
    locgau2_line = getattr(cfg, "locgau2_line", "LOCGAU2 0.02 0.05 2.0\n")

    # --- Phase mapping (template style) ---
    # You can keep yours (P-only/S-only) or the expanded mapping.
    # Use cfg overrides if provided.
    phase_map_lines = getattr(cfg, "phase_map_lines", None)
    if phase_map_lines is None:
        phase_map_lines = (
            "LOCPHASEID  P   P p Pg Pn\n"
            "LOCPHASEID  S   S s Sg Sn\n"
        )

    # --- Angles: safest is NO ---
    # Your earlier error had "unrecognized angle mode" when it couldn't parse,
    # but as a stable default, NO is widely accepted.
    angles_line = getattr(cfg, "locangles_line", "LOCANGLES NO\n")

    # --- Optional outputs ---
    hyposave_line = getattr(cfg, "lochypout_line", "LOCHYPOUT SAVE_NLLOC_ALL NLL_FORMAT_VER_2 SAVE_HYPOINV_SUM\n")

    # --- Write file (force unix newlines) ---
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("CONTROL 1 54321\n")
        f.write(f"TRANS SIMPLE {lat0:.6f} {lon0:.6f} 0.0\n\n")

        f.write("LOCSIG  SC_catalog_test\n")
        f.write("LOCCOM  Sichuan REAL catalog located with NLLoc (P-only first test)\n\n")

        f.write("# LOCFILES obsFiles obsFileType ttimeFileRoot outputFileRoot [iSwapBytes]\n")
        f.write(f"LOCFILES {obs_file} {getattr(cfg, 'obs_type', 'NLLOC_OBS')} {ttime_root} {out_file_root}\n\n")

        f.write(hyposave_line.strip() + "\n\n")

        f.write("# LOCSEARCH OCT initNumCells_x initNumCells_y initNumCells_z minNodeSize maxNumNodes numScatter useStationsDensity stopOnMinNodeSize\n")
        f.write(locsearch_line + "\n")

        f.write("# LOCGRID xNum yNum zNum xOrig yOrig zOrig dx dy dz gridType saveFlag\n")
        f.write(f"LOCGRID {xNum} {yNum} {zNum}  {xOrig:.6f} {yOrig:.6f} {zOrig:.6f}  {dx:.6f} {dy:.6f} {dz:.6f}  {grid_type} {grid_save}\n\n")

        f.write(locmeth_line.strip() + "\n\n")
        f.write(locgau_line.strip() + "\n")
        f.write(locgau2_line.strip() + "\n\n")

        f.write(phase_map_lines.strip() + "\n\n")
        f.write(locqual2err_line.strip() + "\n\n")

        f.write(angles_line.strip() + "\n")

# =============================================================================
# Main entry
# =============================================================================

def run_step3(cfg: NLLProjectConfig) -> None:
    """
    Run the full Step 3 pipeline:
      1) Read stations -> write NLLoc station file
      2) Read REAL picks -> write NLLoc OBS file
      3) Write NLLoc control file template

    All I/O paths are derived from cfg.
    """
    # Prepare directories
    ensure_dir(cfg.out_dir)
    ensure_dir(str(Path(cfg.out_dir) / "out"))

    # 1) Stations
    stations, lat_mean, lon_mean = read_stations(cfg.station_file)
    station_out = str(Path(cfg.out_dir) / cfg.station_out_name)
    write_nlloc_station_file(stations, station_out)

    # Choose TRANS SIMPLE reference point
    if cfg.trans_ref_use_station_mean:
        lat0, lon0 = lat_mean, lon_mean
    else:
        lat0, lon0 = cfg.grid.lat0, cfg.grid.lon0

    # 2) REAL -> OBS
    obs_out = str(Path(cfg.out_dir) / cfg.obs_out_name)
    n_events = convert_crop_to_nll_obs(
        real_file=cfg.real_file,
        out_obs=obs_out,
        sigma_p=cfg.sigma_p,
        sigma_s=cfg.sigma_s,
        stations=stations,   # NEW: station-name alignment
    )


    # 3) nlloc.in template
    write_nlloc_in_file(cfg, lat0=lat0, lon0=lon0)
    write_nlloc_temp_in_file(cfg, lat0=lat0, lon0=lon0)

    # Summary
    nlloc_in_path = str(Path(cfg.out_dir) / cfg.nlloc_in_name)
    print("[OK] Step 3 completed.")
    print(f"  Station file : {station_out}")
    print(f"  Obs file     : {obs_out}  (events written: {n_events})")
    print(f"  Control file : {nlloc_in_path}")
    print(f"  TRANS SIMPLE : lat0={lat0:.4f}, lon0={lon0:.4f} "
          f"({'station mean' if cfg.trans_ref_use_station_mean else 'cfg.grid'})")
    print("  Reminder: LOCGRID must be within your TT grid volume.")


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Single source of truth: config object.
    # In documentation, users modify only this block.
    # -------------------------------------------------------------------------
    cfg = NLLProjectConfig(
        grid=NLLGridConfig(
            # reuse your Step2 config exactly
            input_npz="run_fm3d/data/xyz_vp_vs_axes_km.npz",
            lon0=102.5,
            lat0=27.5,
            earth_radius_km=6371.0,
            grid_step_km=5.0,
            z_min_km=0.0,
            z_max_km=70.0,
            #out_basename_p="nlloc_script/demo_data/slow_P_cubic",
            #out_basename_s="nlloc_script/demo_data/slow_S_cubic",
        ),
        real_file="data/real.crop.txt",
        station_file="run_fm3d/data/sc_station.txt",
        out_dir="run_fm3d/data/nlloc_sc_crop",
        tt_dir="run_fm3d/data/time_sc",
        sigma_p=0.2,
        sigma_s=0.3,
        trans_ref_use_station_mean=False,
        # LOCGRID should be consistent with your TT grids:
        loc_x_min=-80, loc_x_max=80, loc_dx=2.5,
        loc_y_min=-80, loc_y_max=80, loc_dy=2.5,
        loc_z_min=0, loc_z_max=40, loc_dz=2.5,
    )

    run_step3(cfg)
