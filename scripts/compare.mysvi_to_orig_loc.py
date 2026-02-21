#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

# -----------------------------
# Config
# -----------------------------
IN_JSON = Path("run_fm3d/data/svi/out/location_results_v1.0.json")   # 改成你的文件
OUT_DIR = Path("run_fm3d/data/svi/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 和你定位脚本保持一致（用于把 lon/lat 转成 km）
LON0 = 102.5
LAT0 = 27.5
H_THRESH = 30.0   # km
Z_THRESH = 60.0   # km
# -----------------------------
# Projection: lon/lat -> AEQD meters
# -----------------------------
def build_forward_aeqd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(CRS.from_epsg(4326), crs_aeqd, always_xy=True)

proj_fwd = build_forward_aeqd(LON0, LAT0)

def is_finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def load_records(path: Path):
    """
    Input JSON schema (per event):
      {
        "783": {
          "event_id": 783,
          "lon_med": ...,
          "lat_med": ...,
          "depth_med_km": ...,
          "real_lon": ...,
          "real_lat": ...,
          "real_dep_km": ...,
          ...
        },
        ...
      }
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        if "error" in v:
            continue

        eid = v.get("event_id", None)
        if eid is None:
            # 兼容 key 本身就是 eid
            try:
                eid = int(k)
            except Exception:
                continue

        # required fields
        need = ["lon_med", "lat_med", "depth_med_km", "real_lon", "real_lat", "real_dep_km"]
        if any((f not in v) for f in need):
            continue

        lon_p, lat_p, dep_p = v["lon_med"], v["lat_med"], v["depth_med_km"]
        lon_r, lat_r, dep_r = v["real_lon"], v["real_lat"], v["real_dep_km"]

        if not all(map(is_finite, [lon_p, lat_p, dep_p, lon_r, lat_r, dep_r])):
            continue

        lon_p = float(lon_p); lat_p = float(lat_p); dep_p = float(dep_p)
        lon_r = float(lon_r); lat_r = float(lat_r); dep_r = float(dep_r)

        # AEQD km coordinates for horizontal error
        x_p_m, y_p_m = proj_fwd.transform(lon_p, lat_p)
        x_r_m, y_r_m = proj_fwd.transform(lon_r, lat_r)

        dx_km = (x_p_m - x_r_m) / 1000.0
        dy_km = (y_p_m - y_r_m) / 1000.0
        dh_km = float(math.hypot(dx_km, dy_km))

        dlon = lon_p - lon_r
        dlat = lat_p - lat_r
        ddep = dep_p - dep_r
        mad_x, mad_y, mad_z = v["x_mad_km"]
        horiz_mad = math.hypot(mad_x, mad_y)

        #if horiz_mad > H_THRESH or mad_z > Z_THRESH:
        #    continue  # skip outliers
        rows.append({
            "event_id": int(eid),
            "real_lon": lon_r, "real_lat": lat_r, "real_dep_km": dep_r,
            "pred_lon": lon_p, "pred_lat": lat_p, "pred_dep_km": dep_p,
            "dlon_deg": dlon, "dlat_deg": dlat, "ddep_km": ddep,
            "dx_km": dx_km, "dy_km": dy_km, "horiz_err_km": dh_km,
        })

    # sort by event_id
    rows.sort(key=lambda r: r["event_id"])
    return rows

def robust_limits(a, pad=0.05):
    a = np.asarray(a, dtype=float)
    lo = np.nanpercentile(a, 1)
    hi = np.nanpercentile(a, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(a); hi = np.nanmax(a)
    span = hi - lo if hi > lo else 1.0
    return lo - pad*span, hi + pad*span

def save_csv(rows, out_csv: Path):
    import csv
    cols = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    rows = load_records(IN_JSON)
    if not rows:
        raise SystemExit("No valid events found (missing fields / NaN / error entries).")

    save_csv(rows, OUT_DIR / "compare_errors.csv")

    eid = np.array([r["event_id"] for r in rows], dtype=int)

    real_lon = np.array([r["real_lon"] for r in rows], dtype=float)
    real_lat = np.array([r["real_lat"] for r in rows], dtype=float)
    pred_lon = np.array([r["pred_lon"] for r in rows], dtype=float)
    pred_lat = np.array([r["pred_lat"] for r in rows], dtype=float)

    real_dep = np.array([r["real_dep_km"] for r in rows], dtype=float)
    pred_dep = np.array([r["pred_dep_km"] for r in rows], dtype=float)

    dlon = np.array([r["dlon_deg"] for r in rows], dtype=float)
    dlat = np.array([r["dlat_deg"] for r in rows], dtype=float)
    ddep = np.array([r["ddep_km"] for r in rows], dtype=float)

    dh = np.array([r["horiz_err_km"] for r in rows], dtype=float)

    # -----------------------------
    # 1) Lon-Lat map comparison
    # -----------------------------
    plt.figure()
    plt.scatter(real_lon, real_lat, marker="o", label="REAL")
    plt.scatter(pred_lon, pred_lat, marker="x", label="Located")

    # connect each event
    for i in range(len(rows)):
        plt.plot([real_lon[i], pred_lon[i]], [real_lat[i], pred_lat[i]])

    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(f"Location comparison (N={len(rows)})")
    plt.legend()

    xlim = robust_limits(np.r_[real_lon, pred_lon])
    ylim = robust_limits(np.r_[real_lat, pred_lat])
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_lonlat_compare.png", dpi=200)
    plt.close()

    # -----------------------------
    # 2) Depth comparison scatter
    # -----------------------------
    plt.figure()
    plt.scatter(real_dep, pred_dep)
    lo, hi = robust_limits(np.r_[real_dep, pred_dep], pad=0.08)
    plt.plot([lo, hi], [lo, hi])  # y=x
    plt.xlabel("REAL depth (km)")
    plt.ylabel("Located depth (km)")
    plt.title("Depth: Located vs REAL (y=x reference)")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_depth_scatter.png", dpi=200)
    plt.close()

    # -----------------------------
    # 3) Error vs event_id
    # -----------------------------
    plt.figure()
    plt.plot(eid, dlon, label="Δlon (deg)")
    plt.plot(eid, dlat, label="Δlat (deg)")
    plt.plot(eid, ddep, label="Δdepth (km)")
    plt.axhline(0.0)
    plt.xlabel("event_id")
    plt.ylabel("error (pred - real)")
    plt.title("Errors vs event_id")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_errors_vs_eid.png", dpi=200)
    plt.close()

    # -----------------------------
    # 4) Horizontal error histogram (km)
    # -----------------------------
    plt.figure()
    plt.hist(dh, bins=40)
    plt.xlabel("Horizontal error (km)")
    plt.ylabel("Count")
    plt.title("Horizontal error distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_horiz_err_hist.png", dpi=200)
    plt.close()

    # quick console summary
    def pctl(a, q): return float(np.nanpercentile(a, q))
    print(f"Saved plots to: {OUT_DIR.resolve()}")
    print("Horizontal error (km):",
          f"median={pctl(dh,50):.2f}, p90={pctl(dh,90):.2f}, p95={pctl(dh,95):.2f}, max={np.nanmax(dh):.2f}")
    print("Depth error (km):",
          f"median={pctl(np.abs(ddep),50):.2f}, p90={pctl(np.abs(ddep),90):.2f}, max={np.nanmax(np.abs(ddep)):.2f}")

if __name__ == "__main__":
    main()
