import os
import json
import math
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyproj import CRS, Transformer


# ----------------------------
# Helpers
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
    """Inclusive axis [vmin, vmax] with given step."""
    n = int(round((vmax - vmin) / step)) + 1
    axis = vmin + step * np.arange(n, dtype=float)
    axis[-1] = vmax
    return axis


def snap_floor(x: float, step: float) -> float:
    return math.floor(x / step) * step


def snap_ceil(x: float, step: float) -> float:
    return math.ceil(x / step) * step


def parse_velo(filename: str) -> np.ndarray:
    """
    Read velo.txt with 5 columns: lon, lat, depth_km, vp, vs
    Supports comma or whitespace; ignores blank/# lines.
    """
    rows = []
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
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
        raise ValueError(f"No valid rows read from {filename}")
    return np.asarray(rows, dtype=float)


def idw_knn_fill(
    xy_known: np.ndarray,
    v_known: np.ndarray,
    xy_query: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Inverse-distance weighting with k nearest neighbors (pure numpy).
    xy_known: [M,2], v_known: [M]
    xy_query: [Q,2]
    """
    dx = xy_query[:, None, 0] - xy_known[None, :, 0]
    dy = xy_query[:, None, 1] - xy_known[None, :, 1]
    d2 = dx * dx + dy * dy

    k = min(k, xy_known.shape[0])
    nn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    nn_d2 = np.take_along_axis(d2, nn_idx, axis=1)
    nn_v = v_known[nn_idx]

    exact = nn_d2 <= eps
    out = np.empty(xy_query.shape[0], dtype=float)
    for i in range(xy_query.shape[0]):
        if np.any(exact[i]):
            out[i] = float(nn_v[i, np.argmax(exact[i])])
        else:
            w = 1.0 / (np.power(nn_d2[i], power / 2.0) + eps)
            out[i] = float(np.sum(w * nn_v[i]) / np.sum(w))
    return out


def make_local_aeqd_transformer(lon0: float, lat0: float) -> Tuple[Transformer, Transformer]:
    """
    Local AEQD projection centered at lon0/lat0.
    Output units: meters.
    """
    crs_geod = CRS.from_epsg(4326)
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    fwd = Transformer.from_crs(crs_geod, crs_aeqd, always_xy=True)
    inv = Transformer.from_crs(crs_aeqd, crs_geod, always_xy=True)
    return fwd, inv


def axis_from_range_step(vmin: float, vmax: float, step: float) -> np.ndarray:
    """Inclusive axis, snapped to step."""
    vmin_s = math.floor(vmin / step) * step
    vmax_s = math.ceil(vmax / step) * step
    n = int(round((vmax_s - vmin_s) / step)) + 1
    return vmin_s + step * np.arange(n, dtype=float)


# ----------------------------
# Step 1: fill missing values on regular lon/lat/depth grid
# ----------------------------
def build_filled_lonlat_grid(
    data: np.ndarray,
    lon_pad_deg: float = 1.0,
    lat_pad_deg: float = 1.0,
    knn: int = 8,
    idw_power: float = 2.0,
    # if you want cushion nodes for fm3d/grid3dg, set with_cushion=True
    with_cushion: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a COMPLETE regular grid in (lon,lat,depth) after filling missing values.

    Returns:
      lon_axis, lat_axis, dep_axis, Vp_grid, Vs_grid
    where grids are [Nz, Ny, Nx] = [depth, lat, lon].
    """
    lon = data[:, 0]
    lat = data[:, 1]
    dep = data[:, 2]
    vp = data[:, 3]
    vs = data[:, 4]

    # infer steps from data
    dlon = robust_step(lon, "lon")
    dlat = robust_step(lat, "lat")
    dz = robust_step(dep, "depth")

    # bounds (pad lon/lat by 1 deg, depth by min/max)
    lon_min = snap_floor(float(np.min(lon) - lon_pad_deg), dlon)
    lon_max = snap_ceil(float(np.max(lon) + lon_pad_deg), dlon)
    lat_min = snap_floor(float(np.min(lat) - lat_pad_deg), dlat)
    lat_max = snap_ceil(float(np.max(lat) + lat_pad_deg), dlat)
    dep_min = snap_floor(float(np.min(dep)), dz)
    dep_max = snap_ceil(float(np.max(dep)), dz)

    lon_axis = make_axis(lon_min, lon_max, dlon)
    lat_axis = make_axis(lat_min, lat_max, dlat)
    dep_axis = make_axis(dep_min, dep_max, dz)

    if with_cushion:
        lon_axis = np.concatenate(([lon_axis[0] - dlon], lon_axis, [lon_axis[-1] + dlon]))
        lat_axis = np.concatenate(([lat_axis[0] - dlat], lat_axis, [lat_axis[-1] + dlat]))
        dep_axis = np.concatenate(([dep_axis[0] - dz], dep_axis, [dep_axis[-1] + dz]))

    Nx = len(lon_axis)
    Ny = len(lat_axis)
    Nz = len(dep_axis)

    Vp = np.full((Nz, Ny, Nx), np.nan, dtype=float)
    Vs = np.full((Nz, Ny, Nx), np.nan, dtype=float)

    # For vertical fill, group samples by nearest (lat,lon) node
    col_samples: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = defaultdict(list)
    for lo, la, de, vpi, vsi in data:
        ix = int(np.argmin(np.abs(lon_axis - lo)))
        iy = int(np.argmin(np.abs(lat_axis - la)))
        col_samples[(iy, ix)].append((de, vpi, vsi))

    # Stage 1: vertical interpolation/extrapolation per column
    for iy in range(Ny):
        for ix in range(Nx):
            samp = col_samples.get((iy, ix))
            if not samp:
                continue
            samp.sort(key=lambda t: t[0])
            d_s = np.array([t[0] for t in samp], dtype=float)
            vp_s = np.array([t[1] for t in samp], dtype=float)
            vs_s = np.array([t[2] for t in samp], dtype=float)

            # merge duplicate depths
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

            vp_col = np.interp(dep_axis, d_s, vp_s)  # extrapolate by endpoints
            vs_col = np.interp(dep_axis, d_s, vs_s)

            # fill NaNs for that column
            m = np.isnan(Vp[:, iy, ix])
            Vp[m, iy, ix] = vp_col[m]
            m = np.isnan(Vs[:, iy, ix])
            Vs[m, iy, ix] = vs_col[m]

    # Stage 2: horizontal kNN-IDW fill for each depth slice
    lon2d = np.tile(lon_axis[None, :], (Ny, 1))
    lat2d = np.tile(lat_axis[:, None], (1, Nx))
    xy_all = np.stack([lon2d.reshape(-1), lat2d.reshape(-1)], axis=1)

    for iz in range(Nz):
        # Vp slice
        layer = Vp[iz].reshape(-1)
        known = ~np.isnan(layer)
        if np.any(~known):
            if np.sum(known) == 0:
                raise RuntimeError("No known values in a depth slice; data too sparse.")
            layer[~known] = idw_knn_fill(xy_all[known], layer[known], xy_all[~known], k=knn, power=idw_power)
            Vp[iz] = layer.reshape(Ny, Nx)

        # Vs slice
        layer = Vs[iz].reshape(-1)
        known = ~np.isnan(layer)
        if np.any(~known):
            layer[~known] = idw_knn_fill(xy_all[known], layer[known], xy_all[~known], k=knn, power=idw_power)
            Vs[iz] = layer.reshape(Ny, Nx)

    if np.isnan(Vp).any() or np.isnan(Vs).any():
        raise RuntimeError("Still has NaNs after fill; increase knn or check data coverage.")

    return lon_axis, lat_axis, dep_axis, Vp.astype(np.float32), Vs.astype(np.float32)


# ----------------------------
# Main API: build grids + project uniform km grid + save xyz/vp/vs
# ----------------------------
def build_velocity_grid(
    filename: str,
    # If you pass None, these will be auto-inferred from data + padding rule
    lon_min: Optional[float] = None, lon_max: Optional[float] = None, dlon: Optional[float] = None,
    lat_min: Optional[float] = None, lat_max: Optional[float] = None, dlat: Optional[float] = None,
    depth_min: Optional[float] = None, depth_max: Optional[float] = None, ddepth: Optional[float] = None,
    lon_pad_deg: float = 1.0,
    lat_pad_deg: float = 1.0,
    knn: int = 8,
    idw_power: float = 2.0,
    # projected uniform grid settings
    spacing_km: float = 5.0,
    out_prefix: str = "run_fm3d/run_demo/data/xyz_vp_vs",
    write_csv: bool = False,
):
    """
    Returns:
      lon_axis, lat_axis, dep_axis, Vp_lonlat, Vs_lonlat,  (filled lon/lat/dep grid)
      x_km, y_km, z_km, VP_km, VS_km                      (projected uniform grid)
    Also writes:
      out_prefix.npy            (N,5) [x,y,z,vp,vs]
      out_prefix_axes_km.npz    (x_km,y_km,z_km, lon0,lat0)
      out_prefix_meta.json
      optionally out_prefix.csv
    """
    data = parse_velo(filename)

    # Step 1: fill missing values on regular lon/lat/dep grid (auto axes)
    lon_axis, lat_axis, dep_axis, Vp_ll, Vs_ll = build_filled_lonlat_grid(
        data,
        lon_pad_deg=lon_pad_deg,
        lat_pad_deg=lat_pad_deg,
        knn=knn,
        idw_power=idw_power,
        with_cushion=False,  # for NLLoc grids, usually no cushion
    )

    # Center for projection uses Step1 grid center (your requirement)
    lon0 = float(0.5 * (lon_axis[0] + lon_axis[-1]))
    lat0 = float(0.5 * (lat_axis[0] + lat_axis[-1]))

    fwd, inv = make_local_aeqd_transformer(lon0, lat0)

    # Determine projected x/y bounds by projecting the 4 corners
    corners_lon = np.array([lon_axis[0], lon_axis[0], lon_axis[-1], lon_axis[-1]], dtype=float)
    corners_lat = np.array([lat_axis[0], lat_axis[-1], lat_axis[0], lat_axis[-1]], dtype=float)
    x_m, y_m = fwd.transform(corners_lon, corners_lat)
    x_km_c = x_m / 1000.0
    y_km_c = y_m / 1000.0
    x_min, x_max = float(np.min(x_km_c)), float(np.max(x_km_c))
    y_min, y_max = float(np.min(y_km_c)), float(np.max(y_km_c))

    # z range from depth axis (positive down)
    z_min, z_max = float(dep_axis[0]), float(dep_axis[-1])

    # Uniform 5 km axes
    x_km = axis_from_range_step(x_min, x_max, spacing_km)
    y_km = axis_from_range_step(y_min, y_max, spacing_km)
    z_km = axis_from_range_step(z_min, z_max, spacing_km)

    # SciPy interpolators in (depth, lat, lon) space
    itp_vp = RegularGridInterpolator(
        (dep_axis, lat_axis, lon_axis),
        Vp_ll, method="linear",
        bounds_error=False, fill_value=None
    )
    itp_vs = RegularGridInterpolator(
        (dep_axis, lat_axis, lon_axis),
        Vs_ll, method="linear",
        bounds_error=False, fill_value=None
    )

    # Query uniform grid: loop by z, mesh xy once (memory-friendly)
    xx, yy = np.meshgrid(x_km, y_km, indexing="xy")
    xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    nxy = xy.shape[0]

    Nz, Ny, Nx = len(z_km), len(y_km), len(x_km)
    n_points = Nz * nxy

    # Output array (N,5) float32
    out_arr = np.empty((n_points, 5), dtype=np.float32)

    idx0 = 0
    for z in z_km:
        lon_q, lat_q = inv.transform(xy[:, 0] * 1000.0, xy[:, 1] * 1000.0)
        dep_q = np.full(nxy, z, dtype=float)
        pts = np.stack([dep_q, lat_q, lon_q], axis=1)

        vp_val = np.asarray(itp_vp(pts), dtype=float)
        vs_val = np.asarray(itp_vs(pts), dtype=float)

        sl = slice(idx0, idx0 + nxy)
        out_arr[sl, 0] = xy[:, 0].astype(np.float32)
        out_arr[sl, 1] = xy[:, 1].astype(np.float32)
        out_arr[sl, 2] = np.float32(z)
        out_arr[sl, 3] = vp_val.astype(np.float32)
        out_arr[sl, 4] = vs_val.astype(np.float32)
        idx0 += nxy

    # reshape to grids (Nz,Ny,Nx)
    VP_km = out_arr[:, 3].reshape(Nz, Ny, Nx)
    VS_km = out_arr[:, 4].reshape(Nz, Ny, Nx)

    # Save
    out_npy = f"{out_prefix}.npy"
    out_axes = f"{out_prefix}_axes_km.npz"
    out_meta = f"{out_prefix}_meta.json"
    out_csv = f"{out_prefix}.csv"

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    np.save(out_npy, out_arr)
    np.savez(out_axes, x_km=x_km, y_km=y_km, z_km=z_km, lon0=lon0, lat0=lat0)

    meta = {
        "projection": "AEQD",
        "lon0": lon0,
        "lat0": lat0,
        "spacing_km": spacing_km,
        "x_range_km": [float(x_km[0]), float(x_km[-1])],
        "y_range_km": [float(y_km[0]), float(y_km[-1])],
        "z_range_km": [float(z_km[0]), float(z_km[-1])],
        "shape": [int(Nz), int(Ny), int(Nx)],
        "n_points": int(n_points),
        "axes_file": out_axes,
        "npy_file": out_npy,
        "csv_file": out_csv if write_csv else None,
        "step1_lonlat_grid": {
            "lon_range": [float(lon_axis[0]), float(lon_axis[-1])],
            "lat_range": [float(lat_axis[0]), float(lat_axis[-1])],
            "dep_range": [float(dep_axis[0]), float(dep_axis[-1])],
            "shape": [int(len(dep_axis)), int(len(lat_axis)), int(len(lon_axis))],  # (Nz,Ny,Nx)
        },
        "fill_method": {
            "vertical": "np.interp per (lon,lat) column with endpoint extrapolation",
            "horizontal": f"kNN-IDW per depth slice (k={knn}, power={idw_power})"
        }
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if write_csv:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("x_km,y_km,z_km,vp,vs\n")
            for row in out_arr:
                f.write(f"{row[0]:.3f},{row[1]:.3f},{row[2]:.3f},{row[3]:.6f},{row[4]:.6f}\n")

    print("=== Step 1: filled lon/lat/dep grid ===")
    print("lon:", lon_axis[0], lon_axis[-1], "n=", len(lon_axis))
    print("lat:", lat_axis[0], lat_axis[-1], "n=", len(lat_axis))
    print("dep:", dep_axis[0], dep_axis[-1], "n=", len(dep_axis))
    print("Vp/Vs shape (dep,lat,lon):", Vp_ll.shape)

    print("=== Step 2-3: projected uniform km grid ===")
    print(f"center lon0/lat0: {lon0:.6f}/{lat0:.6f} spacing_km={spacing_km}")
    print("x_km:", x_km[0], x_km[-1], "Nx=", len(x_km))
    print("y_km:", y_km[0], y_km[-1], "Ny=", len(y_km))
    print("z_km:", z_km[0], z_km[-1], "Nz=", len(z_km))
    print("Saved:", out_npy, out_axes, out_meta, ("and "+out_csv if write_csv else ""))

    return lon_axis, lat_axis, dep_axis, Vp_ll, Vs_ll, x_km, y_km, z_km, VP_km, VS_km


if __name__ == "__main__":
    # Example:
    #  - reads scattered velo.txt
    #  - fills missing values robustly
    #  - projects & resamples to 5 km uniform grid
    #  - saves xyz_vp_vs.npy / axes / meta
    build_velocity_grid(
        filename="run_fm3d/run_demo/data/velo.txt",
        lon_pad_deg=1.0,
        lat_pad_deg=1.0,
        knn=8,
        idw_power=2.0,
        spacing_km=5.0,
        out_prefix="run_fm3d/run_demo/data/xyz_vp_vs",
        write_csv=False,
    )
