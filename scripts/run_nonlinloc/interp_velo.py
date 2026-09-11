#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Workflow:
1) Fill missing values on a regular (lon,lat,depth_km) grid using:
   - vertical 1D interpolation/extrapolation per (lat,lon) column
   - horizontal kNN-IDW fill per depth slice
   This produces a COMPLETE regular grid (no NaNs).

2) Build a local projection using pyproj, centered at (lon0,lat0) = center of step-1 grid.
   Use AEQD (azimuthal equidistant) for local metric accuracy.

3) Create a UNIFORM projected grid with spacing = 5 km (x,y,z all 5 km by default),
   with range automatically derived from step-1 lon/lat bounds projected to x/y (km)
   and depth range from step-1 depth bounds.
   Then resample vp/vs onto that grid using SciPy RegularGridInterpolator.

4) Save xyz(km) + vp + vs:
   - xyz_vp_vs.npy  : float32 array (N,5) columns [x_km,y_km,z_km,vp,vs]
   - xyz_vp_vs.csv  : optional
   - grid_axes_km.npz: x_km,y_km,z_km arrays + lon0,lat0 metadata
"""

import math
import json
import argparse
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyproj import CRS, Transformer


# ----------------------------
# Basic utils
# ----------------------------
def robust_step(vals: np.ndarray, name: str) -> float:
    """Infer grid step from unique sorted values. Uses median of positive diffs, rounded."""
    u = np.unique(np.round(vals.astype(float), 10))
    u.sort()
    if len(u) < 2:
        raise ValueError(f"Cannot infer step for {name}: only one unique value.")
    diffs = np.diff(u)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError(f"Cannot infer step for {name}: no positive diffs.")
    step = float(np.median(diffs))
    step = float(f"{step:.6g}")
    if step <= 0:
        raise ValueError(f"Bad inferred step for {name}: {step}")
    return step


def make_axis(vmin: float, vmax: float, step: float) -> np.ndarray:
    """Build inclusive axis [vmin, vmax] with given step (tolerant to floating error)."""
    n = int(round((vmax - vmin) / step)) + 1
    axis = vmin + step * np.arange(n, dtype=float)
    axis[-1] = vmax
    return axis


def parse_velo(path: str) -> np.ndarray:
    """
    Read velo.txt: lon,lat,depth_km,Vp,Vs (comma or whitespace separated), ignore blank/# lines.
    Returns array shape [N,5].
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(",", " ")
            parts = [p for p in s.split() if p]
            if len(parts) < 5:
                continue
            try:
                lon, lat, dep, vp, vs = map(float, parts[:5])
            except ValueError:
                continue
            rows.append((lon, lat, dep, vp, vs))
    if not rows:
        raise ValueError(f"No valid rows read from {path} (expect 5 columns: lon lat depth vp vs).")
    return np.asarray(rows, dtype=float)


def idw_knn_fill(xy_known: np.ndarray, v_known: np.ndarray,
                 xy_query: np.ndarray, k: int = 8, power: float = 2.0,
                 eps: float = 1e-12) -> np.ndarray:
    """
    Inverse-distance weighting with k nearest neighbors (pure numpy).
    xy_known: [M,2], v_known: [M]
    xy_query: [Q,2]
    """
    dx = xy_query[:, None, 0] - xy_known[None, :, 0]
    dy = xy_query[:, None, 1] - xy_known[None, :, 1]
    d2 = dx * dx + dy * dy

    k = min(k, xy_known.shape[0])
    nn_idx = np.argpartition(d2, kth=k-1, axis=1)[:, :k]  # [Q,k]
    nn_d2 = np.take_along_axis(d2, nn_idx, axis=1)        # [Q,k]
    nn_v = v_known[nn_idx]                                # [Q,k]

    exact = nn_d2 <= eps
    out = np.empty(xy_query.shape[0], dtype=float)
    for i in range(xy_query.shape[0]):
        if np.any(exact[i]):
            out[i] = float(nn_v[i, np.argmax(exact[i])])
        else:
            w = 1.0 / (np.power(nn_d2[i], power / 2.0) + eps)
            out[i] = float(np.sum(w * nn_v[i]) / np.sum(w))
    return out


# ----------------------------
# Step 1: fill to complete regular lon/lat/dep grid (with cushion)
# ----------------------------
def build_full_grid(
    data: np.ndarray,
    lon_pad_deg: float = 1.0,
    lat_pad_deg: float = 1.0,
    knn: int = 8,
    idw_power: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create full (with cushion) regular grid for Vp and Vs.
    Returns:
      lon_tot, lat_tot, dep_tot, vp_grid, vs_grid
    where *_tot include cushion nodes => sizes (nvp+2),(nvt+2),(nvr+2).
    Grids are shaped [nr, nt, np] corresponding to [depth,lat,lon].
    """
    lon = data[:, 0]
    lat = data[:, 1]
    dep = data[:, 2]

    dlon = robust_step(lon, "lon")
    dlat = robust_step(lat, "lat")
    dz = robust_step(dep, "depth")

    lon_min = float(np.min(lon) - lon_pad_deg)
    lon_max = float(np.max(lon) + lon_pad_deg)
    lat_min = float(np.min(lat) - lat_pad_deg)
    lat_max = float(np.max(lat) + lat_pad_deg)
    dep_min = float(np.min(dep))
    dep_max = float(np.max(dep))

    def snap_floor(x, step): return math.floor(x / step) * step
    def snap_ceil(x, step):  return math.ceil(x / step) * step

    lon_min = snap_floor(lon_min, dlon)
    lon_max = snap_ceil(lon_max, dlon)
    lat_min = snap_floor(lat_min, dlat)
    lat_max = snap_ceil(lat_max, dlat)
    dep_min = snap_floor(dep_min, dz)
    dep_max = snap_ceil(dep_max, dz)

    lon_in = make_axis(lon_min, lon_max, dlon)
    lat_in = make_axis(lat_min, lat_max, dlat)
    dep_in = make_axis(dep_min, dep_max, dz)

    # cushion nodes
    lon_tot = np.concatenate(([lon_in[0] - dlon], lon_in, [lon_in[-1] + dlon]))
    lat_tot = np.concatenate(([lat_in[0] - dlat], lat_in, [lat_in[-1] + dlat]))
    dep_tot = np.concatenate(([dep_in[0] - dz], dep_in, [dep_in[-1] + dz]))

    np_tot = len(lon_tot)
    nt_tot = len(lat_tot)
    nr_tot = len(dep_tot)

    vp_grid = np.full((nr_tot, nt_tot, np_tot), np.nan, dtype=float)
    vs_grid = np.full((nr_tot, nt_tot, np_tot), np.nan, dtype=float)

    # Build per-column sample lists by snapping lon/lat to nearest nodes
    col_samples: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = defaultdict(list)
    for (lo, la, de, vpi, vsi) in data:
        klo = int(np.argmin(np.abs(lon_tot - lo)))
        kla = int(np.argmin(np.abs(lat_tot - la)))
        col_samples[(kla, klo)].append((de, vpi, vsi))

    # Stage 1: vertical fill per (lat,lon) column
    for kla in range(nt_tot):
        for klo in range(np_tot):
            samp = col_samples.get((kla, klo))
            if not samp:
                continue
            samp = sorted(samp, key=lambda t: t[0])
            d_s = np.array([t[0] for t in samp], dtype=float)
            vp_s = np.array([t[1] for t in samp], dtype=float)
            vs_s = np.array([t[2] for t in samp], dtype=float)

            # remove duplicate depths by averaging
            udep, inv = np.unique(d_s, return_inverse=True)
            if len(udep) != len(d_s):
                vp_acc = np.zeros_like(udep)
                vs_acc = np.zeros_like(udep)
                cnt = np.zeros_like(udep)
                for idx, gi in enumerate(inv):
                    vp_acc[gi] += vp_s[idx]
                    vs_acc[gi] += vs_s[idx]
                    cnt[gi] += 1
                vp_s = vp_acc / np.maximum(cnt, 1)
                vs_s = vs_acc / np.maximum(cnt, 1)
                d_s = udep

            vp_col = np.interp(dep_tot, d_s, vp_s)  # extrapolate by endpoints
            vs_col = np.interp(dep_tot, d_s, vs_s)

            m = np.isnan(vp_grid[:, kla, klo])
            vp_grid[m, kla, klo] = vp_col[m]
            m = np.isnan(vs_grid[:, kla, klo])
            vs_grid[m, kla, klo] = vs_col[m]

    # Stage 2: horizontal fill per depth layer using kNN-IDW
    lon2d = np.tile(lon_tot[None, :], (nt_tot, 1))
    lat2d = np.tile(lat_tot[:, None], (1, np_tot))
    xy_all = np.stack([lon2d.reshape(-1), lat2d.reshape(-1)], axis=1)  # [nt*np,2]

    for ir in range(nr_tot):
        # Vp
        layer = vp_grid[ir, :, :].reshape(-1)
        known_mask = ~np.isnan(layer)
        if np.any(~known_mask):
            xy_k = xy_all[known_mask]
            v_k = layer[known_mask]
            xy_q = xy_all[~known_mask]
            layer[~known_mask] = idw_knn_fill(xy_k, v_k, xy_q, k=knn, power=idw_power)
            vp_grid[ir, :, :] = layer.reshape(nt_tot, np_tot)

        # Vs
        layer = vs_grid[ir, :, :].reshape(-1)
        known_mask = ~np.isnan(layer)
        if np.any(~known_mask):
            xy_k = xy_all[known_mask]
            v_k = layer[known_mask]
            xy_q = xy_all[~known_mask]
            layer[~known_mask] = idw_knn_fill(xy_k, v_k, xy_q, k=knn, power=idw_power)
            vs_grid[ir, :, :] = layer.reshape(nt_tot, np_tot)

    if np.isnan(vp_grid).any() or np.isnan(vs_grid).any():
        raise RuntimeError("Still has NaNs after fill. Increase knn or check data coverage.")

    return lon_tot, lat_tot, dep_tot, vp_grid, vs_grid


# ----------------------------
# Step 2-3-4: pyproj projection + SciPy resample to uniform 5 km grid + save xyz,vp,vs
# ----------------------------
def make_local_aeqd_transformer(lon0: float, lat0: float) -> Tuple[Transformer, Transformer, CRS]:
    """
    Local azimuthal equidistant projection centered at lon0/lat0.
    Units: meters.
    """
    crs_geod = CRS.from_epsg(4326)
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    fwd = Transformer.from_crs(crs_geod, crs_aeqd, always_xy=True)
    inv = Transformer.from_crs(crs_aeqd, crs_geod, always_xy=True)
    return fwd, inv, crs_aeqd


def axis_from_range_step(vmin: float, vmax: float, step: float) -> np.ndarray:
    """Inclusive axis, snap to step."""
    vmin_s = math.floor(vmin / step) * step
    vmax_s = math.ceil(vmax / step) * step
    n = int(round((vmax_s - vmin_s) / step)) + 1
    return vmin_s + step * np.arange(n, dtype=float)


def resample_to_uniform_km_grid(
    lon_tot: np.ndarray, lat_tot: np.ndarray, dep_tot: np.ndarray,
    vp_grid: np.ndarray, vs_grid: np.ndarray,
    spacing_km: float = 5.0,
    chunk: int = 2000000,
    out_prefix: str = "xyz_vp_vs",
    write_csv: bool = False
) -> None:
    """
    Build uniform (x_km,y_km,z_km) grid at given spacing_km,
    query vp/vs by mapping query points back to (lon,lat,dep) then using RegularGridInterpolator.
    Save xyz+vp+vs.
    """
    # Use INTERIOR (exclude cushion) to define bounds and projection center
    lon_in = lon_tot[1:-1]
    lat_in = lat_tot[1:-1]
    dep_in = dep_tot[1:-1]

    lon0 = float(0.5 * (lon_in[0] + lon_in[-1]))
    lat0 = float(0.5 * (lat_in[0] + lat_in[-1]))

    fwd, inv, crs_local = make_local_aeqd_transformer(lon0, lat0)

    # Project 4 corners to determine x/y range (meters)
    corners_lon = np.array([lon_in[0], lon_in[0], lon_in[-1], lon_in[-1]], dtype=float)
    corners_lat = np.array([lat_in[0], lat_in[-1], lat_in[0], lat_in[-1]], dtype=float)
    x_m, y_m = fwd.transform(corners_lon, corners_lat)
    x_km_c = x_m / 1000.0
    y_km_c = y_m / 1000.0

    x_min, x_max = float(np.min(x_km_c)), float(np.max(x_km_c))
    y_min, y_max = float(np.min(y_km_c)), float(np.max(y_km_c))

    # Depth range (km), use interior bounds
    z_min, z_max = float(dep_in[0]), float(dep_in[-1])  # positive down

    # Build uniform km axes
    x_axis = axis_from_range_step(x_min, x_max, spacing_km)
    y_axis = axis_from_range_step(y_min, y_max, spacing_km)
    z_axis = axis_from_range_step(z_min, z_max, spacing_km)

    # Interpolators in (dep, lat, lon) space
    # Note: vp_grid/vs_grid are [dep, lat, lon] corresponding to dep_tot/lat_tot/lon_tot
    vp_itp = RegularGridInterpolator(
        (dep_tot, lat_tot, lon_tot),
        vp_grid,
        method="linear",
        bounds_error=False,
        fill_value=None  # allow extrapolation; but we will keep queries within projected bounds
    )
    vs_itp = RegularGridInterpolator(
        (dep_tot, lat_tot, lon_tot),
        vs_grid,
        method="linear",
        bounds_error=False,
        fill_value=None
    )

    # Prepare output
    nxyz = len(x_axis) * len(y_axis) * len(z_axis)
    out_npy = f"{out_prefix}.npy"
    out_csv = f"{out_prefix}.csv"
    out_axes = f"{out_prefix}_axes_km.npz"
    out_meta = f"{out_prefix}_meta.json"

    # We'll write NPY in one shot (memory) if not too huge; otherwise use memmap
    # float32 (N,5) ~ 20 bytes/pt
    est_gb = nxyz * 20 / (1024**3)
    use_memmap = est_gb > 2.0  # threshold
    if use_memmap:
        arr = np.memmap(out_npy, dtype=np.float32, mode="w+", shape=(nxyz, 5))
    else:
        arr = np.empty((nxyz, 5), dtype=np.float32)

    # Optional CSV streaming
    csv_f = None
    if write_csv:
        csv_f = open(out_csv, "w", encoding="utf-8")
        csv_f.write("x_km,y_km,z_km,vp,vs\n")

    # Iterate in chunks over (z, y, x) to keep locality
    # Build 2D mesh for x/y once
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    xy_q = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)  # [Ny*Nx,2]
    nxy = xy_q.shape[0]

    idx0 = 0
    for z in z_axis:
        # Inverse project (x,y)->(lon,lat)
        # inv expects meters
        lon_q, lat_q = inv.transform(xy_q[:, 0] * 1000.0, xy_q[:, 1] * 1000.0)

        # Build query points for interpolator: (dep, lat, lon)
        dep_q = np.full(nxy, z, dtype=float)
        pts = np.stack([dep_q, lat_q, lon_q], axis=1)  # [nxy,3]

        vp_vals = vp_itp(pts)
        vs_vals = vs_itp(pts)

        # Some points may be None/NaN if outside; convert to NaN
        vp_vals = np.asarray(vp_vals, dtype=float)
        vs_vals = np.asarray(vs_vals, dtype=float)

        # Write into output
        n = nxy
        sl = slice(idx0, idx0 + n)
        arr[sl, 0] = xy_q[:, 0].astype(np.float32)  # x_km
        arr[sl, 1] = xy_q[:, 1].astype(np.float32)  # y_km
        arr[sl, 2] = np.float32(z)                  # z_km
        arr[sl, 3] = vp_vals.astype(np.float32)
        arr[sl, 4] = vs_vals.astype(np.float32)

        if csv_f is not None:
            # stream CSV line by line (slower but safe)
            for i in range(n):
                csv_f.write(f"{xy_q[i,0]:.3f},{xy_q[i,1]:.3f},{z:.3f},{vp_vals[i]:.6f},{vs_vals[i]:.6f}\n")

        idx0 += n

    if isinstance(arr, np.memmap):
        arr.flush()
    else:
        np.save(out_npy, arr)

    if csv_f is not None:
        csv_f.close()

    np.savez(out_axes, x_km=x_axis, y_km=y_axis, z_km=z_axis, lon0=lon0, lat0=lat0)
    meta = {
        "projection": "AEQD",
        "lon0": lon0,
        "lat0": lat0,
        "spacing_km": spacing_km,
        "x_range_km": [float(x_axis[0]), float(x_axis[-1])],
        "y_range_km": [float(y_axis[0]), float(y_axis[-1])],
        "z_range_km": [float(z_axis[0]), float(z_axis[-1])],
        "shape": [int(len(z_axis)), int(len(y_axis)), int(len(x_axis))],
        "n_points": int(nxyz),
        "axes_file": out_axes,
        "npy_file": out_npy if not use_memmap else out_npy,
        "csv_file": out_csv if write_csv else None,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=== Uniform projected grid (km) ===")
    print(f"Center (lon0,lat0)=({lon0:.6f},{lat0:.6f}) using AEQD")
    print(f"Spacing: {spacing_km} km (x,y,z)")
    print(f"x: {x_axis[0]:.3f} .. {x_axis[-1]:.3f}  (Nx={len(x_axis)})")
    print(f"y: {y_axis[0]:.3f} .. {y_axis[-1]:.3f}  (Ny={len(y_axis)})")
    print(f"z: {z_axis[0]:.3f} .. {z_axis[-1]:.3f}  (Nz={len(z_axis)})")
    print(f"Total points: {nxyz}  (~{(nxyz*20)/(1024**3):.2f} GB as float32 N×5)")
    print(f"Wrote: {out_npy}, {out_axes}, {out_meta}" + (f", {out_csv}" if write_csv else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velo", default="run_fm3d/run_demo/data/velo.txt", help="Input velo.txt: lon,lat,depth_km,Vp,Vs")
    ap.add_argument("--lon_pad", type=float, default=1.0, help="Extend longitude bounds by this many degrees (interior)")
    ap.add_argument("--lat_pad", type=float, default=1.0, help="Extend latitude bounds by this many degrees (interior)")
    ap.add_argument("--knn", type=int, default=8, help="kNN for horizontal IDW fill")
    ap.add_argument("--power", type=float, default=2.0, help="IDW power")
    ap.add_argument("--spacing_km", type=float, default=5.0, help="Uniform spacing in projected km grid (x,y,z)")
    ap.add_argument("--out_prefix", default="run_fm3d/run_demo/data/xyz_vp_vs", help="Output prefix for npy/csv/npz/meta")
    ap.add_argument("--write_csv", action="store_true", help="Also write CSV (large files; slower)")
    args = ap.parse_args()

    data = parse_velo(args.velo)

    lon_tot, lat_tot, dep_tot, vp_grid, vs_grid = build_full_grid(
        data,
        lon_pad_deg=args.lon_pad,
        lat_pad_deg=args.lat_pad,
        knn=args.knn,
        idw_power=args.power
    )

    # Basic summary
    lon_in, lat_in, dep_in = lon_tot[1:-1], lat_tot[1:-1], dep_tot[1:-1]
    print("=== Step 1 (filled lon/lat/dep grid, with cushion) ===")
    print(f"lon(deg): [{lon_in[0]:.6f}, {lon_in[-1]:.6f}]  (n={len(lon_in)})")
    print(f"lat(deg): [{lat_in[0]:.6f}, {lat_in[-1]:.6f}]  (n={len(lat_in)})")
    print(f"dep(km):  [{dep_in[0]:.6f}, {dep_in[-1]:.6f}]  (n={len(dep_in)})")
    print(f"Filled grids shape (dep,lat,lon) = {vp_grid.shape} (includes cushion)")

    # Step 2-4
    resample_to_uniform_km_grid(
        lon_tot, lat_tot, dep_tot,
        vp_grid, vs_grid,
        spacing_km=args.spacing_km,
        out_prefix=args.out_prefix,
        write_csv=args.write_csv
    )


if __name__ == "__main__":
    main()
