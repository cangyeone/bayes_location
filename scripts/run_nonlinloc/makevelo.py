#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert scattered/partially-missing velo.txt (lon,lat,depth_km,Vp,Vs)
into FM3D/grid3dg external 3D velocity grid format:
  - one value per line
  - total entries = (nvr+2)*(nvt+2)*(nvp+2)
  - read order: outermost radius(i), middle latitude(j), innermost longitude(k)
    DO i=0,nvr+1
      DO j=0,nvt+1
        DO k=0,nvp+1
          READ value
        ENDDO
      ENDDO
    ENDDO
per FMTOMO manual. Also generates a grid3dg.in skeleton using external 3D file.

Assumptions:
- depth is positive downward in km.
- lon/lat are degrees East/North.
"""

import math
import argparse
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np


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
    # Round step to a sensible decimal (avoid 0.009999999)
    # Keep 6 significant digits
    step = float(f"{step:.6g}")
    if step <= 0:
        raise ValueError(f"Bad inferred step for {name}: {step}")
    return step


def make_axis(vmin: float, vmax: float, step: float) -> np.ndarray:
    """
    Build inclusive axis [vmin, vmax] with given step (tolerant to floating error).
    """
    n = int(round((vmax - vmin) / step)) + 1
    axis = vmin + step * np.arange(n, dtype=float)
    # Force last to be exactly vmax (within tolerance)
    axis[-1] = vmax
    return axis


def parse_velo(path: str) -> np.ndarray:
    """
    Read velo.txt: lon,lat,depth_km,Vp,Vs (comma or whitespace separated), ignore blank/# lines.
    Returns array shape [N,5].
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(",", " ")
            parts = [p for p in s.split() if p]
            if len(parts) < 5:
                continue
            lon, lat, dep, vp, vs = map(float, parts[:5])
            rows.append((lon, lat, dep, vp, vs))
    if not rows:
        raise ValueError(f"No valid rows read from {path}")
    return np.asarray(rows, dtype=float)


def idw_knn_fill(xy_known: np.ndarray, v_known: np.ndarray,
                 xy_query: np.ndarray, k: int = 8, power: float = 2.0,
                 eps: float = 1e-12) -> np.ndarray:
    """
    Inverse-distance weighting with k nearest neighbors (pure numpy).
    xy_known: [M,2], v_known: [M]
    xy_query: [Q,2]
    """
    # Compute squared distances [Q,M]
    # For large grids, this can be heavy; typical regional models are ok.
    dx = xy_query[:, None, 0] - xy_known[None, :, 0]
    dy = xy_query[:, None, 1] - xy_known[None, :, 1]
    d2 = dx * dx + dy * dy

    # Indices of k nearest for each query
    k = min(k, xy_known.shape[0])
    nn_idx = np.argpartition(d2, kth=k-1, axis=1)[:, :k]  # [Q,k]
    nn_d2 = np.take_along_axis(d2, nn_idx, axis=1)        # [Q,k]
    nn_v = v_known[nn_idx]                                # [Q,k]

    # If any exact match (distance ~0), return that directly
    exact = nn_d2 <= eps
    out = np.empty(xy_query.shape[0], dtype=float)
    for i in range(xy_query.shape[0]):
        if np.any(exact[i]):
            out[i] = float(nn_v[i, np.argmax(exact[i])])
        else:
            w = 1.0 / (np.power(nn_d2[i], power / 2.0) + eps)
            out[i] = float(np.sum(w * nn_v[i]) / np.sum(w))
    return out


def build_full_grid(data: np.ndarray,
                    lon_pad_deg: float = 1.0,
                    lat_pad_deg: float = 1.0,
                    knn: int = 8,
                    idw_power: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create full (with cushion) regular grid for Vp and Vs.
    Returns:
      lon_nodes_total, lat_nodes_total, dep_nodes_total, vp_grid_total, vs_grid_total
    where *_nodes_total include cushion nodes => sizes (nvp+2),(nvt+2),(nvr+2).
    Grids are shaped [nr_total, nt_total, np_total] corresponding to [depth,lat,lon].
    """
    lon = data[:, 0]
    lat = data[:, 1]
    dep = data[:, 2]
    vp = data[:, 3]
    vs = data[:, 4]

    # Infer steps from the data itself
    dlon = robust_step(lon, "lon")
    dlat = robust_step(lat, "lat")
    dz = robust_step(dep, "depth")

    # Define interior model bounds (user requirement: lon/lat extend by 1 degree; depth use min/max)
    lon_min = float(np.min(lon) - lon_pad_deg)
    lon_max = float(np.max(lon) + lon_pad_deg)
    lat_min = float(np.min(lat) - lat_pad_deg)
    lat_max = float(np.max(lat) + lat_pad_deg)
    dep_min = float(np.min(dep))
    dep_max = float(np.max(dep))

    # Snap bounds to inferred step grid (so that axis is regular)
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

    # Add cushion nodes on both sides (total = interior + 2)
    lon_tot = np.concatenate(([lon_in[0] - dlon], lon_in, [lon_in[-1] + dlon]))
    lat_tot = np.concatenate(([lat_in[0] - dlat], lat_in, [lat_in[-1] + dlat]))
    dep_tot = np.concatenate(([dep_in[0] - dz], dep_in, [dep_in[-1] + dz]))

    np_tot = len(lon_tot)
    nt_tot = len(lat_tot)
    nr_tot = len(dep_tot)

    # Map scattered points into a dict keyed by (ilon, ilat, idep) in the TOTAL grid (but only if exactly on nodes)
    # We'll later interpolate; so here we store whatever exact-node values exist.
    lon_index = {float(f"{v:.10g}"): i for i, v in enumerate(lon_tot)}
    lat_index = {float(f"{v:.10g}"): j for j, v in enumerate(lat_tot)}
    dep_index = {float(f"{v:.10g}"): k for k, v in enumerate(dep_tot)}

    def keyify(x): return float(f"{x:.10g}")

    cell_vp = {}
    cell_vs = {}
    for (lo, la, de, vpi, vsi) in data:
        klo = lon_index.get(keyify(lo))
        kla = lat_index.get(keyify(la))
        kde = dep_index.get(keyify(de))
        if klo is None or kla is None or kde is None:
            # Point not exactly on derived grid node; ignore here (will be handled via interpolation)
            continue
        cell_vp[(kde, kla, klo)] = float(vpi)
        cell_vs[(kde, kla, klo)] = float(vsi)

    # Initialize grids with NaN
    vp_grid = np.full((nr_tot, nt_tot, np_tot), np.nan, dtype=float)
    vs_grid = np.full((nr_tot, nt_tot, np_tot), np.nan, dtype=float)

    for (kde, kla, klo), vpi in cell_vp.items():
        vp_grid[kde, kla, klo] = vpi
    for (kde, kla, klo), vsi in cell_vs.items():
        vs_grid[kde, kla, klo] = vsi

    # Stage 1: Vertical interpolation/extrapolation for each (lat,lon) column using existing samples in that column
    # We use any scattered values near that column by nearest node (in case original points are not snapped).
    # Build per-column sample lists from raw data by snapping lon/lat to nearest total grid node.
    col_samples: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = defaultdict(list)
    for (lo, la, de, vpi, vsi) in data:
        klo = int(np.argmin(np.abs(lon_tot - lo)))
        kla = int(np.argmin(np.abs(lat_tot - la)))
        col_samples[(kla, klo)].append((de, vpi, vsi))

    for kla in range(nt_tot):
        for klo in range(np_tot):
            samp = col_samples.get((kla, klo))
            if not samp:
                continue
            samp = sorted(samp, key=lambda t: t[0])
            d_s = np.array([t[0] for t in samp], dtype=float)
            vp_s = np.array([t[1] for t in samp], dtype=float)
            vs_s = np.array([t[2] for t in samp], dtype=float)

            # Remove duplicate depths by averaging
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

            # Interp along depth nodes (dep_tot). np.interp extrapolates using endpoints.
            vp_col = np.interp(dep_tot, d_s, vp_s)
            vs_col = np.interp(dep_tot, d_s, vs_s)

            # Fill where NaN
            m = np.isnan(vp_grid[:, kla, klo])
            vp_grid[m, kla, klo] = vp_col[m]
            m = np.isnan(vs_grid[:, kla, klo])
            vs_grid[m, kla, klo] = vs_col[m]

    # Stage 2: Horizontal fill at each depth using kNN-IDW based on available (lat,lon) nodes
    lonlat_mesh = np.stack(np.meshgrid(lon_tot, lat_tot, indexing="xy"), axis=-1)  # [nt,np,2] but careful indexing
    # meshgrid with indexing="xy" returns [lat,lon]; we want [nt,np,2] = (lon,lat)
    # Let's rebuild explicitly:
    lon2d = np.tile(lon_tot[None, :], (nt_tot, 1))
    lat2d = np.tile(lat_tot[:, None], (1, np_tot))
    xy_all = np.stack([lon2d.reshape(-1), lat2d.reshape(-1)], axis=1)  # [nt*np,2]

    for ir in range(nr_tot):
        # Vp
        layer = vp_grid[ir, :, :].reshape(-1)
        known_mask = ~np.isnan(layer)
        if np.any(~known_mask):
            if np.sum(known_mask) == 0:
                raise ValueError("After vertical fill, still no known values at some depth layer.")
            xy_k = xy_all[known_mask]
            v_k = layer[known_mask]
            xy_q = xy_all[~known_mask]
            filled = idw_knn_fill(xy_k, v_k, xy_q, k=knn, power=idw_power)
            layer[~known_mask] = filled
            vp_grid[ir, :, :] = layer.reshape(nt_tot, np_tot)

        # Vs
        layer = vs_grid[ir, :, :].reshape(-1)
        known_mask = ~np.isnan(layer)
        if np.any(~known_mask):
            xy_k = xy_all[known_mask]
            v_k = layer[known_mask]
            xy_q = xy_all[~known_mask]
            filled = idw_knn_fill(xy_k, v_k, xy_q, k=knn, power=idw_power)
            layer[~known_mask] = filled
            vs_grid[ir, :, :] = layer.reshape(nt_tot, np_tot)

    # Final sanity
    if np.isnan(vp_grid).any() or np.isnan(vs_grid).any():
        raise RuntimeError("Still has NaNs after fill. Check data coverage or increase knn.")

    return lon_tot, lat_tot, dep_tot, vp_grid, vs_grid


def write_fm3d_grid(out_path: str, grid: np.ndarray) -> None:
    """
    Write in required read order: radius outermost, lat middle, lon innermost,
    one value per line.
    grid shape [nr, nt, np]
    """
    nr, nt, np_ = grid.shape
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(nr):         # radius/depth
            for j in range(nt):     # latitude
                for k in range(np_):# longitude
                    f.write(f"{grid[i, j, k]:.6f}\n")


def write_grid3dg_in(path: str,
                     lon_tot: np.ndarray, lat_tot: np.ndarray, dep_tot: np.ndarray,
                     npr: int = None, npt: int = None, npp: int = None,
                     vp_file: str = "vp3d.fm3d",
                     vs_file: str = "vs3d.fm3d",
                     earth_radius_km: float = 6371.0) -> None:
    """
    Generate a practical grid3dg.in for one layer, using external 3D P and S velocity files.
    Notes:
    - grid3dg model dimensions are defined WITHOUT cushion; it automatically adds cushion nodes.
      We therefore use the INTERIOR bounds (i.e. exclude the first/last of *_tot).
    - Radial range is relative to surface, top can be above surface (positive), bottom negative.
      Given depth positive downward, we set radial_top = -dep_min, radial_bottom = -dep_max.
    """
    # Interior axes (exclude cushion nodes)
    lon_in = lon_tot[1:-1]
    lat_in = lat_tot[1:-1]
    dep_in = dep_tot[1:-1]

    lon_min, lon_max = float(lon_in[0]), float(lon_in[-1])
    lat_min, lat_max = float(lat_in[0]), float(lat_in[-1])
    dep_min, dep_max = float(dep_in[0]), float(dep_in[-1])

    radial_top = -dep_min
    radial_bottom = -dep_max

    # Velocity node counts (excluding cushion)
    nvp = len(lon_in)
    nvt = len(lat_in)
    nvr = len(dep_in)

    # Propagation grid points (default: 3x density, but at least 21/41 like manual example)
    if npr is None:
        npr = max(21, 3 * nvr)
    if npt is None:
        npt = max(41, 3 * nvt)
    if npp is None:
        npp = max(41, 3 * nvp)

    # One layer => 2 interfaces. Interface grids have same horizontal node distribution; use nvt/nvp.
    # Top interface at radial_top (km), bottom at radial_bottom (km). Use planar (option 0) by setting corners equal.

    txt = f"""\
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c grid3dg.in (auto-generated)
c One layer; external 3-D P and S velocity grids for fm3d.
c
c Notes from manual:
c - External 3-D file must have (nvr+2)*(nvt+2)*(nvp+2) entries,
c   include boundary (cushion) nodes, one value per line.
c - Read order is lon innermost, radius outermost.
c - grid3dg automatically adds cushion boundary nodes to the defined grid.
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

1               c: Number of layers

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Set 3-D grid size and location (WITHOUT cushion; grid3dg adds cushion)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
{radial_top:.6f} {radial_bottom:.6f}    c: Radial range (top-bottom) of grid (km; +up, -down)
{lat_max:.6f} {lat_min:.6f}             c: Latitudinal range (N-S) of grid (degrees)
{lon_min:.6f} {lon_max:.6f}             c: Longitudinal range (E-W) of grid (degrees)
{earth_radius_km:.1f}                   c: Earth radius (km)

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Set up propagation grid file (fm3d uses this for wavefront tracking)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
propgrid.in     c: Name of propagation grid file
{npr:d} {npt:d} {npp:d}  c: Number of points in rad lat, long
5 10            c: Refine factor & no. of local cells
0.05            c: Cushion factor for prop grid (<<1)

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Layer 1 velocity grid definition (P)
c (This block mirrors the manual's "velocity grid" option structure:
c  external file option 0, then choose P or S, then filename, then 1-D/3-D flag)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
{nvr:d}         c: nvr (number of velocity nodes in radius, WITHOUT cushion)
{nvt:d}         c: nvt (number of velocity nodes in theta/lat, WITHOUT cushion)
{nvp:d}         c: nvp (number of velocity nodes in phi/lon, WITHOUT cushion)
0               c: Velocity option (0=external file, 1=constant gradient)
1               c: Use P(1) or S(2) velocities when option=0
{vp_file}       c: External velocity file name
3               c: External file type (1=1-D depth model, 3=3-D grid)

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Layer 1 velocity grid definition (S) - separate grid (recommended)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
{nvr:d}
{nvt:d}
{nvp:d}
0
2
{vs_file}
3

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Interface grids
c For M layers, need M+1 interfaces. Here M=1 => 2 interfaces.
c Interface grids require same horizontal node distribution; grid3dg adds cushion.
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
{nvt:d}         c: Number of grid points in theta (N-S) WITHOUT cushion
{nvp:d}         c: Number of grid points in phi (E-W) WITHOUT cushion

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Interface 1 (top) - planar
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
0               c: Obtain grid from external file (0=no,1=yes)
interface1.z     c: External interface grid file (if option=1)
{radial_top:.6f} c: Height of NW grid point (km)
{radial_top:.6f} c: Height of NE grid point (km)
{radial_top:.6f} c: Height of SW grid point (km)

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Interface 2 (bottom) - planar
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
0
interface2.z
{radial_bottom:.6f}
{radial_bottom:.6f}
{radial_bottom:.6f}

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Output file names expected by fm3d
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
vgrids.in
interfaces.in

12345           c: Random seed (only used if you add random structure)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velo", default="run_fm3d/run_demo/data/velo.txt", help="Input velo.txt: lon,lat,depth_km,Vp,Vs")
    ap.add_argument("--out_vp", default="run_fm3d/run_demo/data/vp3d.fm3d", help="Output P velocity 3D file (one value per line)")
    ap.add_argument("--out_vs", default="run_fm3d/run_demo/data/vs3d.fm3d", help="Output S velocity 3D file (one value per line)")
    ap.add_argument("--out_grid3dg", default="run_fm3d/run_demo/data/grid3dg.in", help="Output grid3dg.in")
    ap.add_argument("--lon_pad", type=float, default=1.0, help="Extend longitude bounds by this many degrees (interior)")
    ap.add_argument("--lat_pad", type=float, default=1.0, help="Extend latitude bounds by this many degrees (interior)")
    ap.add_argument("--knn", type=int, default=8, help="kNN for horizontal IDW fill")
    ap.add_argument("--power", type=float, default=2.0, help="IDW power")
    args = ap.parse_args()

    data = parse_velo(args.velo)
    lon_tot, lat_tot, dep_tot, vp_grid, vs_grid = build_full_grid(
        data,
        lon_pad_deg=args.lon_pad,
        lat_pad_deg=args.lat_pad,
        knn=args.knn,
        idw_power=args.power
    )

    # Write external 3D files in required order
    write_fm3d_grid(args.out_vp, vp_grid)
    write_fm3d_grid(args.out_vs, vs_grid)

    # Write grid3dg.in
    write_grid3dg_in(args.out_grid3dg, lon_tot, lat_tot, dep_tot,
                     vp_file=args.out_vp, vs_file=args.out_vs)

    # Print a concise summary
    lon_in, lat_in, dep_in = lon_tot[1:-1], lat_tot[1:-1], dep_tot[1:-1]
    print("=== Derived grid (INTERIOR, no cushion) ===")
    print(f"lon: [{lon_in[0]:.6f}, {lon_in[-1]:.6f}], nvp={len(lon_in)}, dlon~{lon_in[1]-lon_in[0]:.6g}")
    print(f"lat: [{lat_in[0]:.6f}, {lat_in[-1]:.6f}], nvt={len(lat_in)}, dlat~{lat_in[1]-lat_in[0]:.6g}")
    print(f"dep: [{dep_in[0]:.6f}, {dep_in[-1]:.6f}], nvr={len(dep_in)}, dz~{dep_in[1]-dep_in[0]:.6g}")
    print("=== TOTAL grid (with cushion; written to external 3D files) ===")
    print(f"np_tot={len(lon_tot)}, nt_tot={len(lat_tot)}, nr_tot={len(dep_tot)}")
    print(f"Total entries per file = {len(lon_tot)*len(lat_tot)*len(dep_tot)}")
    print(f"Wrote: {args.out_vp}, {args.out_vs}, {args.out_grid3dg}")


if __name__ == "__main__":
    main()
