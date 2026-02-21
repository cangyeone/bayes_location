#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from pyproj import CRS, Transformer

import rasterio
from rasterio.warp import transform as rio_transform

from obspy.geodetics.base import gps2dist_azimuth


# ----------------------------
# 1) 读取地震 & 筛选
# ----------------------------
def read_events_real(
    fname: str,
    bbox: Tuple[float, float, float, float],
    vmin_km: float = 0.0,
    vmax_km: float = 80.0,
    eh_max_km: float = 10.0,
    ez_max_km: float = 20.0,
):
    lon_min, lon_max, lat_min, lat_max = bbox

    lons: List[float] = []
    lats: List[float] = []
    deps: List[float] = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                continue
            p = line.strip().split(",")
            if len(p) < 15:
                continue

            try:
                lon, lat, dep = map(float, p[2:5])
            except Exception:
                continue

            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue
            if dep < vmin_km or dep > vmax_km:
                continue

            # uncertainty screening (optional)
            try:
                e1, e2, e3 = map(float, p[11:14])
                eh = math.sqrt(e1 * e1 + e2 * e2)
                ez = e3
                if eh > eh_max_km or ez > ez_max_km:
                    continue
            except Exception:
                pass

            lons.append(lon)
            lats.append(lat)
            deps.append(dep)

    if len(lons) == 0:
        raise RuntimeError(f"No events after screening from: {fname}")

    return (np.asarray(lons, dtype=np.float64),
            np.asarray(lats, dtype=np.float64),
            np.asarray(deps, dtype=np.float64))
from datetime import datetime

def read_events_reloc_csv(
    fname: str,
    bbox: Tuple[float, float, float, float],
    vmin_km: float = 0.0,
    vmax_km: float = 80.0,
    eh_max_km: float = 10.0,
    ez_max_km: float = 20.0,
):
    lon_min, lon_max, lat_min, lat_max = bbox

    lons: List[float] = []
    lats: List[float] = []
    deps: List[float] = []
    times: List[datetime] = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("#"):
                continue

            p = s.split(",")
            if len(p) < 6:
                continue

            # format:
            # #1,2022-09-05 03:30:33.363611,lon,lat,dep,...
            try:
                t = datetime.fromisoformat(p[1].strip())
                lon = float(p[2]); lat = float(p[3]); dep = float(p[4])
            except Exception:
                continue

            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue
            if dep < vmin_km or dep > vmax_km:
                continue

            # uncertainty screening (optional): e1,e2,e3 仍沿用你旧 REAL 的列（如果不适配 reloc.* 就自动跳过）
            try:
                # 这里 reloc.* 你给的例子里 e1/e2/e3 是否在 11:14 不确定，所以用 try/except 保持兼容
                e1, e2, e3 = map(float, p[11:14])
                eh = math.sqrt(e1 * e1 + e2 * e2)
                ez = e3
                if eh > eh_max_km or ez > ez_max_km:
                    continue
            except Exception:
                pass

            lons.append(lon); lats.append(lat); deps.append(dep); times.append(t)

    if len(lons) == 0:
        raise RuntimeError(f"No events after screening from: {fname}")

    return (
        np.asarray(lons, dtype=np.float64),
        np.asarray(lats, dtype=np.float64),
        np.asarray(deps, dtype=np.float64),
        np.asarray(times, dtype=object),
    )
from datetime import datetime

def read_events_real_newfmt(
    fname: str,
    bbox: Tuple[float, float, float, float],
    vmin_km: float = 0.0,
    vmax_km: float = 80.0,
    eh_max_km: float = 10.0,
    ez_max_km: float = 20.0,
):
    lon_min, lon_max, lat_min, lat_max = bbox

    lons: List[float] = []
    lats: List[float] = []
    deps: List[float] = []
    times: List[datetime] = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("#"):
                continue

            p = s.split()
            if len(p) < 10:
                continue

            try:
                # # YYYY MM DD HH MM SS.ss LAT LON DEP ...
                yyyy = int(p[1]); mm = int(p[2]); dd = int(p[3])
                HH = int(p[4]); MM = int(p[5]); SS = float(p[6])

                sec_i = int(SS)
                usec = int(round((SS - sec_i) * 1e6))
                t = datetime(yyyy, mm, dd, HH, MM, sec_i, usec)

                lat = float(p[7])
                lon = float(p[8])
                dep = float(p[9])
            except Exception:
                continue

            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue
            if dep < vmin_km or dep > vmax_km:
                continue

            # uncertainty screening (optional) — 仍按你之前 heuristic
            try:
                if len(p) >= 14:
                    e1 = float(p[11]); e2 = float(p[12]); e3 = float(p[13])
                    eh = math.sqrt(e1 * e1 + e2 * e2)
                    ez = e3
                    if eh > eh_max_km or ez > ez_max_km:
                        continue
            except Exception:
                pass

            lons.append(lon); lats.append(lat); deps.append(dep); times.append(t)

    if len(lons) == 0:
        raise RuntimeError(f"No events after screening from: {fname}")

    return (
        np.asarray(lons, dtype=np.float64),
        np.asarray(lats, dtype=np.float64),
        np.asarray(deps, dtype=np.float64),
        np.asarray(times, dtype=object),
    )
def read_dem_geotiff_geo(
    dem_tif: str,
    bbox: Tuple[float, float, float, float],
    dem_downsample: int = 4,
):
    lon_min, lon_max, lat_min, lat_max = bbox

    with rasterio.open(dem_tif) as ds:
        dem_crs = ds.crs
        if dem_crs is None:
            raise RuntimeError("DEM has no CRS metadata (ds.crs is None).")

        lon_corners = [lon_min, lon_max, lon_max, lon_min]
        lat_corners = [lat_min, lat_min, lat_max, lat_max]

        if str(dem_crs).lower().find("epsg:4326") >= 0:
            xs, ys = lon_corners, lat_corners
        else:
            xs, ys = rio_transform(CRS.from_epsg(4326), dem_crs, lon_corners, lat_corners)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        row_min, col_min = ds.index(x_min, y_max)
        row_max, col_max = ds.index(x_max, y_min)

        row0, row1 = sorted([row_min, row_max])
        col0, col1 = sorted([col_min, col_max])

        window = rasterio.windows.Window.from_slices((row0, row1), (col0, col1))
        dem = ds.read(1, window=window, masked=True).astype(np.float64)

        if dem_downsample > 1:
            dem = dem[::dem_downsample, ::dem_downsample]
            transform = ds.window_transform(window) * rasterio.Affine.scale(dem_downsample, dem_downsample)
        else:
            transform = ds.window_transform(window)

        nrows, ncols = dem.shape
        cols = np.arange(ncols)
        rows = np.arange(nrows)
        cc, rr = np.meshgrid(cols, rows)
        xs_dem, ys_dem = rasterio.transform.xy(transform, rr, cc)
        xs_dem = np.asarray(xs_dem, dtype=np.float64)
        ys_dem = np.asarray(ys_dem, dtype=np.float64)

        if str(dem_crs).lower().find("epsg:4326") >= 0:
            lon_grid, lat_grid = xs_dem, ys_dem
        else:
            lon_grid, lat_grid = rio_transform(dem_crs, CRS.from_epsg(4326), xs_dem.ravel(), ys_dem.ravel())
            lon_grid = np.asarray(lon_grid, float).reshape(nrows, ncols)
            lat_grid = np.asarray(lat_grid, float).reshape(nrows, ncols)

        Z = dem / 1000.0
        Z = np.where(np.ma.getmaskarray(dem), np.nan, Z)

    return lon_grid, lat_grid, Z

import math
from typing import Tuple, List
import numpy as np
# ----------------------------
# 6) JGR-style 2x2 comparison plot
# ----------------------------
def _panel_label(ax, label: str):
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.2", alpha=0.85),
        zorder=20,
    )


def plot_map_profile_on_axes(
    ax_map, ax_prof,
    Xdem, Ydem, Zdem,
    x, y, dep,
    center, u, v,
    *,
    dem_cmap="terrain",
    dem_alpha=0.25,
    eq_size=3,
    profile_halfwidth_km=10.0,
    topo_profile_n=400,
    s_range=None,          # (smin, smax) forced for comparison
    map_limits=None,       # (xmin,xmax,ymin,ymax)
    depth_limits=None,     # (dmin, dmax) for profile y-limits
    draw_dem=True,
    draw_axis_line=True,
):
    """
    Draw ONE pair: map + profile onto given axes.
    - Uses the provided (center,u,v) for axis definition.
    - If s_range is provided, use it for axis line + topo sampling + xlim(profile).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dep = np.asarray(dep, float)

    # compute s,t in this shared coordinate system
    A = np.column_stack([x, y]) - center[None, :]
    s = A @ u
    t = A @ v

    # ---- Map ----
    if draw_dem:
        m = ax_map.pcolormesh(Xdem, Ydem, Zdem, shading="auto", cmap=dem_cmap, alpha=dem_alpha, zorder=1)
    else:
        m = None

    ax_map.scatter(x, y, s=eq_size, c="k", alpha=0.75, linewidths=0, zorder=5)

    # main axis line in map (from smin..smax)
    if s_range is None:
        smin, smax = float(np.percentile(s, 1)), float(np.percentile(s, 99))
    else:
        smin, smax = float(s_range[0]), float(s_range[1])

    if draw_axis_line:
        p0 = center + smin * u
        p1 = center + smax * u
        ax_map.plot([p0[0], p1[0]], [p0[1], p1[1]], "k-", lw=1.8, zorder=6)

    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("X (km)")
    ax_map.set_ylabel("Y (km)")

    if map_limits is not None:
        xmin, xmax, ymin, ymax = map_limits
        ax_map.set_xlim(xmin, xmax)
        ax_map.set_ylim(ymin, ymax)

    # ---- Profile ----
    hw = float(profile_halfwidth_km)
    sel = np.abs(t) <= hw

    if np.any(sel):
        ax_prof.scatter(s[sel], dep[sel], s=eq_size, c="k", alpha=0.55, linewidths=0, zorder=5)

    # topo along axis (shared smin..smax)
    ss = np.linspace(smin, smax, int(topo_profile_n))
    xs = center[0] + ss * u[0]
    ys = center[1] + ss * u[1]
    topo = bilinear_sample_grid(Xdem, Ydem, Zdem, xs, ys)  # km elevation
    ax_prof.plot(ss, -topo, "k-", lw=1.0, zorder=6)

    ax_prof.set_xlabel("Distance along main axis s (km)")
    ax_prof.set_ylabel("Depth (km)")
    ax_prof.grid(True, ls="--", lw=0.6, alpha=0.35)

    # profile limits
    ax_prof.set_xlim(smin, smax)

    if depth_limits is not None:
        dmin, dmax = depth_limits  # depth positive down
        # include topo a bit above 0
        top_pad = 2.0
        bot_pad = 1.0
        y_top = min(-np.nanmin(topo) - top_pad, 0.0)  # negative (above surface)
        y_bot = dmax + bot_pad
        ax_prof.set_ylim(y_bot, y_top)  # invert depth axis
    else:
        # fallback similar to your original
        y_top = min(-np.nanmin(topo) - 2.0, 0.0)
        y_bot = float(np.nanmax(dep) + 1.0)
        ax_prof.set_ylim(y_bot, y_top)

    return m  # dem mappable (for colorbar)


def plot_jgr_2x2_comparison(
    Xdem, Ydem, Zdem,
    x1, y1, dep1,
    x2, y2, dep2,
    center, u, v,
    *,
    out_png="fig_jgr_2x2.png",
    dem_alpha=0.25,
    eq_size=3,
    profile_halfwidth_km=10.0,
):
    # shared ranges for fair comparison
    # map limits from DEM extent (most stable)
    xmin, xmax = np.nanmin(Xdem), np.nanmax(Xdem)
    ymin, ymax = np.nanmin(Ydem), np.nanmax(Ydem)
    map_limits = (xmin, xmax, ymin, ymax)

    # shared s-range determined ONLY by first dataset (per your request “projection来自第一个图”)
    A1 = np.column_stack([x1, y1]) - center[None, :]
    s1 = A1 @ u
    smin, smax = float(np.percentile(s1, 1)), float(np.percentile(s1, 99))
    s_range = (smin, smax)

    # shared depth limits from BOTH datasets
    dmax = float(np.nanmax([np.nanmax(dep1), np.nanmax(dep2)]))
    depth_limits = (0.0, dmax)

    fig, axs = plt.subplots(
        2, 2, figsize=(12.5, 9.0),
        gridspec_kw=dict(wspace=0.18, hspace=0.18)
    )
    ax_a, ax_b = axs[0, 0], axs[0, 1]
    ax_c, ax_d = axs[1, 0], axs[1, 1]

    # (a)(b): original
    m0 = plot_map_profile_on_axes(
        ax_a, ax_b,
        Xdem, Ydem, Zdem,
        x1, y1, dep1,
        center, u, v,
        dem_alpha=dem_alpha,
        eq_size=eq_size,
        profile_halfwidth_km=profile_halfwidth_km,
        s_range=s_range,
        map_limits=map_limits,
        depth_limits=depth_limits,
        draw_dem=True,
        draw_axis_line=True,
    )

    # (c)(d): Catalog_3.Pha (same projection + same axis)
    m1 = plot_map_profile_on_axes(
        ax_c, ax_d,
        Xdem, Ydem, Zdem,
        x2, y2, dep2,
        center, u, v,
        dem_alpha=dem_alpha,
        eq_size=eq_size,
        profile_halfwidth_km=profile_halfwidth_km,
        s_range=s_range,
        map_limits=map_limits,
        depth_limits=depth_limits,
        draw_dem=True,
        draw_axis_line=True,
    )

    # panel labels
    _panel_label(ax_a, "(a)")
    _panel_label(ax_b, "(b)")
    _panel_label(ax_c, "(c)")
    _panel_label(ax_d, "(d)")

    # remove titles (JGR style)
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.set_title("")

    # One DEM colorbar for BOTH map panels (left column)
    # Use m0 (m1 same scale)
    cbar = fig.colorbar(m0, ax=[ax_a, ax_c], fraction=0.035, pad=0.02)
    cbar.set_label("Elevation (km)")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[OK] saved: {out_png}")


def read_events_real_newfmt_old(
    fname: str,
    bbox: Tuple[float, float, float, float],
    vmin_km: float = 0.0,
    vmax_km: float = 80.0,
    eh_max_km: float = 10.0,
    ez_max_km: float = 20.0,
):
    """
    Read REAL-like catalog in the *new* format:

    Event header line starts with '#', e.g.
    # 2022 09 05 03 30 33.33  29.5720  102.0226    3.74  1.21  0.0038  0.0042  0.0600         1
      YYYY MM DD HH MM SS.ss   LAT     LON        DEP   ...   e1      e2      e3        evid

    Following lines are picks (station tt weight phase), ignored here.

    Output: (lons, lats, deps) float64 arrays, same as read_events_real().
    """
    lon_min, lon_max, lat_min, lat_max = bbox

    lons: List[float] = []
    lats: List[float] = []
    deps: List[float] = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or not s.startswith("#"):
                continue

            # Split by any whitespace (new format is whitespace-separated)
            p = s.split()
            # Minimal tokens check: "# YYYY MM DD HH MM SS LAT LON DEP ..."
            # indexes: 0   1   2  3  4  5  6   7   8   9
            if len(p) < 10:
                continue

            # Parse lon/lat/dep
            try:
                lat = float(p[7])
                lon = float(p[8])
                dep = float(p[9])
            except Exception:
                continue

            # bbox & depth range screening
            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue
            if dep < vmin_km or dep > vmax_km:
                continue

            # uncertainty screening (optional)
            # Heuristic: try to read e1/e2/e3 right after (some extra field) following DEP:
            # Example: ... DEP 1.21 0.0038 0.0042 0.0600 ...
            # => e1,e2,e3 = p[11],p[12],p[13]
            try:
                if len(p) >= 14:
                    e1 = float(p[11])
                    e2 = float(p[12])
                    e3 = float(p[13])
                    eh = math.sqrt(e1 * e1 + e2 * e2)
                    ez = e3
                    if eh > eh_max_km or ez > ez_max_km:
                        continue
            except Exception:
                pass

            lons.append(lon)
            lats.append(lat)
            deps.append(dep)

    if len(lons) == 0:
        raise RuntimeError(f"No events after screening from: {fname}")

    return (
        np.asarray(lons, dtype=np.float64),
        np.asarray(lats, dtype=np.float64),
        np.asarray(deps, dtype=np.float64),
    )

# ----------------------------
# 2) AEQD 投影到 km
# ----------------------------
def make_aeqd_transformer(lon0: float, lat0: float) -> Transformer:
    crs_geo = CRS.from_epsg(4326)
    crs_loc = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=km +no_defs"
    )
    return Transformer.from_crs(crs_geo, crs_loc, always_xy=True)

def aeqd_project_km(lons, lats, lon0, lat0):
    tr = make_aeqd_transformer(lon0, lat0)
    x, y = tr.transform(lons, lats)  # km
    return np.asarray(x, float), np.asarray(y, float)
def _binned_depth_spread(s, dep, bin_km=5.0, s_range=None, stat="iqr"):
    """
    Return bin centers and spread (IQR or std) of depth in each s-bin.
    spread is computed on dep values in each bin.
    """
    s = np.asarray(s, float)
    dep = np.asarray(dep, float)

    if s_range is None:
        smin, smax = np.nanmin(s), np.nanmax(s)
    else:
        smin, smax = s_range

    # ensure ascending edges
    lo, hi = (smin, smax) if smin < smax else (smax, smin)
    edges = np.arange(lo, hi + bin_km, bin_km)
    if edges.size < 2:
        return np.array([]), np.array([])

    centers = 0.5 * (edges[:-1] + edges[1:])
    spread = np.full_like(centers, np.nan, dtype=float)

    for i in range(centers.size):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if np.count_nonzero(m) < 10:   # min samples per bin; adjust if needed
            continue
        d = dep[m]
        if stat == "iqr":
            q25, q75 = np.nanpercentile(d, [25, 75])
            spread[i] = q75 - q25
        elif stat == "std":
            spread[i] = np.nanstd(d)
        else:
            raise ValueError("stat must be 'iqr' or 'std'")
    return centers, spread


def _summarize_spread(spread, agg="median"):
    """Robust summary across bins, ignoring NaNs."""
    spread = np.asarray(spread, float)
    ok = np.isfinite(spread)
    if not np.any(ok):
        return np.nan
    if agg == "median":
        return float(np.nanmedian(spread[ok]))
    elif agg == "mean":
        return float(np.nanmean(spread[ok]))
    else:
        raise ValueError("agg must be 'median' or 'mean'")


import matplotlib.dates as mdates

def plot_jgr_2x2_timecolored(
    lon_dem, lat_dem, Zdem,
    lons1, lats1, dep1, t1,
    lons2, lats2, dep2, t2,
    *,
    center_xy, u, v,              # from dataset1 PCA in AEQD (km)
    aeqd_tr, aeqd_tr_inv,          # forward/inverse transformers
    s_range,                       # (smin,smax) from dataset1 (km)
    out_png="figures/figs/fig_map_profile_2x2.png",
    dem_alpha=0.25,
    eq_size=6,
    profile_halfwidth_km=10.0,
    topo_profile_n=400,
):
    # --- shared time colormap across BOTH datasets ---
    # Convert datetime objects to matplotlib date numbers
    tnum1 = mdates.date2num(list(t1))
    tnum2 = mdates.date2num(list(t2))
    tmin = float(np.nanmin([np.nanmin(tnum1), np.nanmin(tnum2)]))
    tmax = float(np.nanmax([np.nanmax(tnum1), np.nanmax(tnum2)]))
    norm = plt.Normalize(vmin=tmin, vmax=tmax)
    cmap = plt.get_cmap("viridis")

    # --- Create figure ---
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.2), gridspec_kw=dict(wspace=0.18, hspace=0.18))
    ax_a, ax_b = axs[0, 0], axs[0, 1]
    ax_c, ax_d = axs[1, 0], axs[1, 1]

    # --- Helper: map panel in lon/lat ---
    def draw_map(ax, lons, lats, tnum):
        m = ax.pcolormesh(lon_dem, lat_dem, Zdem, shading="auto", cmap="terrain", alpha=dem_alpha, zorder=1)
        sc = ax.scatter(lons, lats, s=eq_size, c=tnum, cmap=cmap, norm=norm, linewidths=0, alpha=0.85, zorder=5)

        # main axis line: use AEQD endpoints -> inverse to lon/lat
        smin, smax = s_range
        p0 = center_xy + smin * u
        p1 = center_xy + smax * u
        lon0, lat0 = aeqd_tr_inv.transform(p0[0], p0[1])
        lon1, lat1 = aeqd_tr_inv.transform(p1[0], p1[1])
        ax.plot([lon0, lon1], [lat0, lat1], "k-", lw=1.8, zorder=6)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_xlim(np.nanmin(lon_dem), np.nanmax(lon_dem))
        ax.set_ylim(np.nanmin(lat_dem), np.nanmax(lat_dem))
        return m, sc

    # --- Helper: profile panel (s-depth), colored by time ---
    def draw_profile(ax, lons, lats, dep, tnum):
        # project to AEQD to compute s,t in the SAME axis system
        x, y = aeqd_tr.transform(lons, lats)  # km
        x = np.asarray(x, float); y = np.asarray(y, float)
        A = np.column_stack([x, y]) - center_xy[None, :]
        s = A @ u
        t = A @ v

        hw = float(profile_halfwidth_km)
        sel = np.abs(t) <= hw

        if np.any(sel):
            ax.scatter(s[sel], dep[sel], s=eq_size, c=tnum[sel], cmap=cmap, norm=norm,
                       linewidths=0, alpha=0.65, zorder=5)

        # topo along axis (same for all panels)
        smin, smax = s_range
        ss = np.linspace(smin, smax, int(topo_profile_n))
        xs = center_xy[0] + ss * u[0]
        ys = center_xy[1] + ss * u[1]
        # sample topo using lon/lat DEM is inconvenient; use your existing AEQD DEM sampler if you prefer.
        # 这里我们直接不画 topo（或你想画 topo，需要用 AEQD 版 DEM 来 bilinear_sample_grid）
        # ——为了不大改你的 DEM 管线，我建议：继续用你原来的 AEQD 版 DEM 来画 topo。
        ax.grid(True, ls="--", lw=0.6, alpha=0.35)
        ax.set_xlabel("Distance along main axis s (km)")
        ax.set_ylabel("Depth (km)")
        ax.set_xlim(smax, smin)
        ax.set_ylim(25, -4.5)
        return

    # ---- panels ----
    m0, sc0 = draw_map(ax_a, lons1, lats1, tnum1)
    draw_profile(ax_b, lons1, lats1, dep1, tnum1)

    m1, sc1 = draw_map(ax_c, lons2, lats2, tnum2)
    draw_profile(ax_d, lons2, lats2, dep2, tnum2)

    # panel labels
    _panel_label(ax_a, "(a)")
    _panel_label(ax_b, "(b)")
    _panel_label(ax_c, "(c)")
    _panel_label(ax_d, "(d)")

    # one colorbar for time (applies to all panels)
    cbar = fig.colorbar(sc0, ax=[ax_a, ax_b, ax_c, ax_d], fraction=0.03, pad=0.02)
    cbar.set_label("Origin time")
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[OK] saved: {out_png}")

# ----------------------------
# 3) 读取 DEM (GeoTIFF) 并投影到 AEQD km
# ----------------------------
def read_dem_geotiff(
    dem_tif: str,
    bbox: Tuple[float, float, float, float],
    lon0: float,
    lat0: float,
    dem_downsample: int = 4,
):
    lon_min, lon_max, lat_min, lat_max = bbox

    with rasterio.open(dem_tif) as ds:
        dem_crs = ds.crs
        if dem_crs is None:
            raise RuntimeError("DEM has no CRS metadata (ds.crs is None).")

        lon_corners = [lon_min, lon_max, lon_max, lon_min]
        lat_corners = [lat_min, lat_min, lat_max, lat_max]

        if str(dem_crs).lower().find("epsg:4326") >= 0:
            xs, ys = lon_corners, lat_corners
        else:
            xs, ys = rio_transform(CRS.from_epsg(4326), dem_crs, lon_corners, lat_corners)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        row_min, col_min = ds.index(x_min, y_max)
        row_max, col_max = ds.index(x_max, y_min)

        row0, row1 = sorted([row_min, row_max])
        col0, col1 = sorted([col_min, col_max])

        window = rasterio.windows.Window.from_slices((row0, row1), (col0, col1))
        dem = ds.read(1, window=window, masked=True).astype(np.float64)

        if dem_downsample > 1:
            dem = dem[::dem_downsample, ::dem_downsample]
            transform = ds.window_transform(window) * rasterio.Affine.scale(dem_downsample, dem_downsample)
        else:
            transform = ds.window_transform(window)

        nrows, ncols = dem.shape
        cols = np.arange(ncols)
        rows = np.arange(nrows)
        cc, rr = np.meshgrid(cols, rows)
        xs_dem, ys_dem = rasterio.transform.xy(transform, rr, cc)
        xs_dem = np.asarray(xs_dem, dtype=np.float64)
        ys_dem = np.asarray(ys_dem, dtype=np.float64)

        if str(dem_crs).lower().find("epsg:4326") >= 0:
            lon_grid, lat_grid = xs_dem, ys_dem
        else:
            lon_grid, lat_grid = rio_transform(dem_crs, CRS.from_epsg(4326), xs_dem.ravel(), ys_dem.ravel())
            lon_grid = np.asarray(lon_grid, float).reshape(nrows, ncols)
            lat_grid = np.asarray(lat_grid, float).reshape(nrows, ncols)

        tr = make_aeqd_transformer(lon0, lat0)
        X, Y = tr.transform(lon_grid, lat_grid)  # km
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)

        Z = dem / 1000.0
        Z = np.where(np.ma.getmaskarray(dem), np.nan, Z)

    return X, Y, Z


# ----------------------------
# 4) HDBSCAN 聚类（在 AEQD km 坐标系）
# ----------------------------
def hdbscan_cluster_xyz(
    x_km: np.ndarray,
    y_km: np.ndarray,
    dep_km: np.ndarray,
    *,
    depth_weight: float = 0.5,
    min_cluster_size: int = 6,
    min_samples: int = 6, 
):
    import hdbscan
    from sklearn.cluster import DBSCAN 

    X = np.column_stack([x_km, y_km, dep_km * float(depth_weight)]).astype(np.float64)
    #mu = X.mean(axis=0, keepdims=True)
    #sd = X.std(axis=0, keepdims=True) + 1e-12
    #Xn = (X - mu) / sd
    Xn = X 
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(Xn)
    #clusterer = DBSCAN(
    #    eps=1.0,
    #    min_samples=50,
    #    metric="euclidean",
    #)
    labels = clusterer.fit_predict(Xn)
    return labels

def gmm_cluster_xyz(
    x_km: np.ndarray,
    y_km: np.ndarray,
    dep_km: np.ndarray,
    *,
    depth_weight: float = 0.5,
    n_components: int = 10,          # K
    covariance_type: str = "full",  # full / diag / tied / spherical
    random_state: int = 0,
):
    """
    用 GMM 在 (x, y, z) 上聚类。
    - z 用 dep_km（向下为正），用 depth_weight 缩放深度维（避免深度主导）。
    返回:
      labels: (N,) hard labels [0..K-1]
      probs:  (N,) 每个点属于其最可能簇的概率（可用于筛选“边界点”）
    """
    from sklearn.mixture import GaussianMixture

    X = np.column_stack([x_km, y_km, dep_km * float(depth_weight)]).astype(np.float64)

    # 标准化（建议保留，GMM 对尺度也敏感）
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    Xn = (X - mu) / sd

    gmm = GaussianMixture(
        n_components=int(n_components),
        covariance_type=covariance_type,
        random_state=int(random_state),
        reg_covar=1e-6,     # 数值稳定（很重要）
        n_init=5,           # 多次初始化，避免局部最优
        max_iter=500,
    )
    gmm.fit(Xn)

    resp = gmm.predict_proba(Xn)         # (N, K)
    labels = resp.argmax(axis=1).astype(np.int32)
    probs = resp.max(axis=1).astype(np.float32)
    return labels
def gmm_select_k_bic(
    Xn: np.ndarray,
    k_min: int = 2,
    k_max: int = 15,
    covariance_type: str = "full",
    random_state: int = 0,
):
    from sklearn.mixture import GaussianMixture

    best_k = None
    best_bic = np.inf
    best_model = None

    for k in range(k_min, k_max + 1):
        m = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            reg_covar=1e-6,
            n_init=5,
            max_iter=500,
        ).fit(Xn)
        bic = m.bic(Xn)
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_model = m

    return best_k, best_bic, best_model
def gmm_cluster_xyz_auto(
    x_km, y_km, dep_km,
    *,
    depth_weight=0.5,
    k_min=2,
    k_max=15,
    covariance_type="full",
    random_state=0,
):
    from sklearn.mixture import GaussianMixture

    X = np.column_stack([x_km, y_km, dep_km * depth_weight]).astype(np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    Xn = (X - mu) / sd

    best_k, best_bic, best_model = gmm_select_k_bic(
        Xn, k_min=k_min, k_max=k_max,
        covariance_type=covariance_type,
        random_state=random_state
    )
    resp = best_model.predict_proba(Xn)
    labels = resp.argmax(axis=1).astype(np.int32)
    probs = resp.max(axis=1).astype(np.float32)
    return labels, probs, best_k, best_bic

# ----------------------------
# 5) 主轴方向（PCA on x,y） + 剖面坐标
# ----------------------------
def pca_main_axis(x: np.ndarray, y: np.ndarray):
    """
    返回：
      center (x0,y0)
      unit direction u = (ux,uy)  (主轴方向)
      unit normal   v = (-uy,ux)
      s = along-axis coordinate (km)
      t = cross-axis coordinate (km)
    """
    pts = np.column_stack([x, y]).astype(np.float64)
    center = pts.mean(axis=0)

    A = pts - center
    C = (A.T @ A) / max(len(pts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    u = eigvecs[:, np.argmax(eigvals)]
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.array([-u[1], u[0]])

    s = A @ u
    t = A @ v
    return center, u, v, s, t


def bilinear_sample_grid(X, Y, Z, xs, ys):
    """
    简单双线性插值：假设 X,Y 是规则网格但不一定严格单调。
    这里用 nearest in each axis 的近似方式：把 X,Y 转成一维轴向坐标。
    对你的 DEM（来自 raster window + 等间隔）一般成立。
    """
    # 取一维坐标轴
    x_axis = X[0, :]
    y_axis = Y[:, 0]
    # 保证单调
    if x_axis[1] < x_axis[0]:
        x_axis = x_axis[::-1]
        Z = Z[:, ::-1]
    if y_axis[1] < y_axis[0]:
        y_axis = y_axis[::-1]
        Z = Z[::-1, :]

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    # 找到格点索引
    ix = np.searchsorted(x_axis, xs) - 1
    iy = np.searchsorted(y_axis, ys) - 1
    ix = np.clip(ix, 0, len(x_axis) - 2)
    iy = np.clip(iy, 0, len(y_axis) - 2)

    x0 = x_axis[ix]
    x1 = x_axis[ix + 1]
    y0 = y_axis[iy]
    y1 = y_axis[iy + 1]

    # 防止除零
    tx = (xs - x0) / (x1 - x0 + 1e-12)
    ty = (ys - y0) / (y1 - y0 + 1e-12)

    z00 = Z[iy, ix]
    z10 = Z[iy, ix + 1]
    z01 = Z[iy + 1, ix]
    z11 = Z[iy + 1, ix + 1]

    z0 = z00 * (1 - tx) + z10 * tx
    z1 = z01 * (1 - tx) + z11 * tx
    z = z0 * (1 - ty) + z1 * ty
    return z


# ----------------------------
# 6) 作图：水平 + 剖面
# ----------------------------
def plot_map_and_profile(
    Xdem, Ydem, Zdem,
    x, y, dep,
    labels,
    center, u, v, s, t,
    *,
    out_png="fig_2d.png",
    dem_cmap="terrain",
    dem_alpha=1.0,
    eq_size=2,
    profile_halfwidth_km=10.0,
    topo_profile_n=400,
    show_cluster_legend=False,
):
    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # ---- (A) Map with DEM ----
    # DEM 用 pcolormesh（支持非严格规则网格也更稳）
    # 注意 shading="auto" 让格点-像素对齐合理
    m = ax0.pcolormesh(Xdem, Ydem, Zdem, shading="auto", cmap=dem_cmap, alpha=dem_alpha)
    cb = fig.colorbar(m, ax=ax0, fraction=0.046, pad=0.02)
    cb.set_label("Elevation (km)")

    # cluster colors
    labels = np.asarray(labels, dtype=np.int32)
    noise = (labels == -1)
    core = ~noise

    # tab20 离散色
    base = plt.get_cmap("tab20")
    # remap core labels to 0..K-1
    if np.any(core):
        uniq = np.unique(labels[core])
        remap = {int(v): i for i, v in enumerate(uniq)}
        lab_mapped = np.full_like(labels, -1)
        for k, vv in remap.items():
            lab_mapped[labels == k] = vv
    else:
        uniq = np.array([], dtype=int)
        lab_mapped = np.full_like(labels, -1)

    # 先画噪声
    if np.any(noise):
        ax0.scatter(x[noise], y[noise], s=eq_size, c="0.6", alpha=0.35, linewidths=0, zorder=5)

    # 再画簇
    if np.any(core):
        colors = base(lab_mapped[core] % 20)
        ax0.scatter(x[core], y[core], s=eq_size, c=colors, alpha=0.9, linewidths=0, zorder=6)

    # 主轴直线（在 map 上画）
    # 取 s 的范围覆盖事件分布
    smin, smax = float(np.percentile(s, 1)), float(np.percentile(s, 99))
    p0 = center + smin * u
    p1 = center + smax * u
    ax0.plot([p0[0], p1[0]], [p0[1], p1[1]], "k-", lw=2.0, zorder=7)

    # 标注主轴方位角（用 obspy 根据两端点反投影回 lon/lat 再算 azimuth）
    # 先将 p0/p1 从 AEQD km -> lon/lat
    # 通过 inverse transformer：aeqd -> geo
    # 这里用 pyproj 直接反变换
    # 注意：我们的 transformer是 geo->aeqd，所以这里要创建反变换
    # 为简单，直接新建一对 transformer
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("X (km)")
    ax0.set_ylabel("Y (km)")
    ax0.set_title(f"Map view ({len(dep)} earthquakes)")

    # 估算方位角（从北顺时针）——在平面上用 dx,dy 也行：
    az = (math.degrees(math.atan2((p1[0]-p0[0]), (p1[1]-p0[1]))) + 360.0) % 360.0
    ax0.text(
        0.02, 0.98, f"Main axis azimuth ≈ {az:.1f}°",
        transform=ax0.transAxes, va="top", ha="left",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.3", alpha=0.85),
        zorder=10
    )

    # 可选：cluster legend（簇很多时不建议开）
    if show_cluster_legend and len(uniq) > 0 and len(uniq) <= 12:
        handles = []
        for k in range(len(uniq)):
            handles.append(plt.Line2D([0],[0], marker="o", color="w",
                                      markerfacecolor=base(k % 20), markersize=7,
                                      label=f"C{k}"))
        ax0.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.85)

    # ---- (B) Profile along main axis ----
    # 选取剖面带宽：|t| <= halfwidth
    hw = float(profile_halfwidth_km)
    sel = np.abs(t) <= hw

    # 剖面横轴：s（km）；纵轴：depth（km, positive down）
    # 颜色用与 map 相同的 cluster colors
    if np.any(sel):
        # noise
        if np.any(sel & noise):
            ax1.scatter(s[sel & noise], dep[sel & noise], s=eq_size, c="0.6",
                        alpha=0.35, linewidths=0, zorder=5)
        # core
        if np.any(sel & core):
            ax1.scatter(s[sel & core], dep[sel & core], s=eq_size,
                        c=base(lab_mapped[sel & core] % 20), alpha=0.6,
                        linewidths=0, zorder=6)

    # 地形剖面：沿主轴采样 DEM
    # 在 smin..smax 上采样
    ss = np.linspace(smin, smax, int(topo_profile_n))
    xs = center[0] + ss * u[0]
    ys = center[1] + ss * u[1]
    topo = bilinear_sample_grid(Xdem, Ydem, Zdem, xs, ys)  # km elevation
    # 在 depth 轴上画地表：用 -topo（让海拔在 0 上方）
    ax1.plot(ss, -topo, "k-", lw=1.2, zorder=7)

    ax1.set_ylim(max(dep.max(), 0) + 1, min(-np.nanmin(topo), 0) - 4.5)  # 深度向下
    ax1.set_xlabel("Distance along main axis s (km)")
    ax1.set_ylabel("Depth (km)")
    ax1.set_title(f"Profile (|t| ≤ {hw:.1f} km)")

    ax1.grid(True, ls="--", lw=0.6, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"[OK] saved: {out_png}")


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=str, default="run_fm3d/data/ours/reloc.real.True.v1.0.txt")
    ap.add_argument("--events2", type=str, default="data/Catalog_3.Pha")
    ap.add_argument("--dem", type=str, default="figures/dem.tif")
    ap.add_argument("--bbox", type=float, nargs=4, default=(101.6, 102.4, 29.2, 30.1),
                    metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    ap.add_argument("--vmin", type=float, default=0.0)
    ap.add_argument("--vmax", type=float, default=30.0)
    ap.add_argument("--ehmax", type=float, default=10.0)
    ap.add_argument("--ezmax", type=float, default=20.0)
    ap.add_argument("--dem_ds", type=int, default=4)
    ap.add_argument("--dem_alpha", type=float, default=0.25)
    ap.add_argument("--profile_halfwidth", type=float, default=10.0)
    ap.add_argument("--out", type=str, default="figures/figs/fig_jgr_compare_2x2.v1.0.png")
    args = ap.parse_args()

    bbox = tuple(args.bbox)
    lon_min, lon_max, lat_min, lat_max = bbox
    lon0 = 0.5 * (lon_min + lon_max)
    lat0 = 0.5 * (lat_min + lat_max)

    # --- DEM: both geo and AEQD (AEQD only needed if you want topo sampling) ---
    lon_dem, lat_dem, Zdem_geo = read_dem_geotiff_geo(args.dem, bbox=bbox, dem_downsample=args.dem_ds)
    Xdem, Ydem, Zdem_aeqd = read_dem_geotiff(args.dem, bbox=bbox, lon0=lon0, lat0=lat0, dem_downsample=args.dem_ds)

    # --- dataset1 reloc.* csv (with time) ---
    lons1, lats1, dep1, t1 = read_events_reloc_csv(
        args.events, bbox=bbox,
        vmin_km=args.vmin, vmax_km=args.vmax,
        eh_max_km=args.ehmax, ez_max_km=args.ezmax,
    )

    # --- dataset2 Catalog_3.Pha (with time) ---
    lons2, lats2, dep2, t2 = read_events_real_newfmt(
        args.events2, bbox=bbox,
        vmin_km=args.vmin, vmax_km=args.vmax,
        eh_max_km=args.ehmax, ez_max_km=args.ezmax,
    )

    # --- AEQD transformers (for PCA axis) ---
    aeqd_tr = make_aeqd_transformer(lon0, lat0)
    aeqd_tr_inv = Transformer.from_crs(aeqd_tr.target_crs, aeqd_tr.source_crs, always_xy=True)

    # --- PCA axis from dataset1 in AEQD km ---
    x1, y1 = aeqd_tr.transform(lons1, lats1)
    x1 = np.asarray(x1, float); y1 = np.asarray(y1, float)

    center_xy, u, v, s1, tt = pca_main_axis(x1, y1)
    smin, smax = float(np.percentile(s1, 1)), float(np.percentile(s1, 99))
    s_range = (smin, smax)

    print(f"[INFO] dataset1={len(dep1)} dataset2={len(dep2)} s_range=[{smin:.2f},{smax:.2f}] km")

    # --- plot 2x2 ---
    plot_jgr_2x2_timecolored(
        lon_dem, lat_dem, Zdem_geo,
        lons1, lats1, dep1, t1,
        lons2, lats2, dep2, t2,
        center_xy=center_xy, u=u, v=v,
        aeqd_tr=aeqd_tr, aeqd_tr_inv=aeqd_tr_inv,
        s_range=s_range,
        out_png=args.out,
        dem_alpha=float(args.dem_alpha),
        eq_size=6,
        profile_halfwidth_km=float(args.profile_halfwidth),
    )



if __name__ == "__main__":
    main()

