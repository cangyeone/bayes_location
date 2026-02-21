#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JGR-style 2x2 location comparison map (Cartopy) with grayscale DEM background,
depth-colored earthquakes (grayscale-friendly), and panel labels (a–d) with N.

Outputs:
  figures/figs/fig_location_comparison.v1.1_cartopy.pdf
  figures/figs/fig_location_comparison.v1.1_cartopy.png

Dependencies:
  numpy, matplotlib, cartopy, rasterio
"""

import os
import math
import datetime
from typing import Tuple, List, Optional

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import rasterio
from rasterio.warp import transform as rio_transform
from pyproj import CRS


# ----------------------------
# Time window (edit as needed)
# ----------------------------
btime1 = datetime.datetime(
                2022, 9, 5
            )
etime1 = datetime.datetime(
                2022, 9, 15
            )

# ----------------------------
# 1) Read catalogs (your original logic; minimally cleaned)
# ----------------------------
def read_event_file(file_path: str) -> np.ndarray:
    """
    Read Liu's relocated catalog-like file:
    - event header starts with '#'
    - time fields: header[0:6] => "%Y %m %d %H %M %S.%f"
    - lat/lon/depth at header[6:9]
    Output columns: [lon, lat, depth, delta_seconds]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    locs: List[List[float]] = []
    for line in lines:
        if not line.startswith("#"):
            continue

        header = line[1:].strip().split()
        try:
            origin_time = datetime.datetime.strptime(" ".join(header[0:6]), "%Y %m %d %H %M %S.%f")
        except Exception:
            # handle "60" seconds case
            origin_time = datetime.datetime.strptime(
                " ".join(header[0:6]).replace("60", "59"),
                "%Y %m %d %H %M %S.%f"
            )

        if origin_time < btime1 or origin_time > etime1:
            continue

        try:
            lat = float(header[6])
            lon = float(header[7])
            depth = float(header[8])
        except Exception:
            continue

        delta = (origin_time - btime1).total_seconds()
        locs.append([lon, lat, depth, delta])

    return np.asarray(locs, dtype=np.float64)


def read1(path: str, errors: List[float] = [10.0, 20.0]) -> np.ndarray:
    """
    Read NonLinLoc CSV like:
      event_id,time,lat,lon,depth,eh,ez,...
    You used:
      etime = datetime.strptime(header[1], "%Y-%m-%dT%H:%M:%S.%f")
      lat = header[2], lon = header[3], depth = header[4]
      e1 = header[5], e2 = header[6] (screen)
    Output: [lon, lat, depth]
    """
    locs: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        # skip header
        _ = f.readline()
        for line in f:
            header = line.strip().split(",")
            if len(header) < 7:
                continue
            if len(header[1]) == 0:
                continue

            try:
                etime = datetime.datetime.strptime(header[1], "%Y-%m-%dT%H:%M:%S.%f")
            except Exception:
                continue

            if etime < btime1 or etime > etime1:
                continue

            try:
                lat = float(header[2])
                lon = float(header[3])
                depth = float(header[4])
                e1 = float(header[5])
                e2 = float(header[6])
            except Exception:
                continue

            if e1 > float(errors[0]) or e2 > float(errors[1]):
                continue

            locs.append([lon, lat, depth])

    return np.asarray(locs, dtype=np.float64)


def readctlg() -> np.ndarray:
    """
    Read CENC/REAL .pha-like file: data/2022.pha
    Extract events from lines containing '#EVENT' with your bounding box filter.
    Output: [lon, lat, dep, delta, mag]
    """
    fname = "data/2022.pha"
    events: List[List[float]] = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            if "NONE" in line:
                continue
            if "#EVENT" not in line:
                continue

            sline = line.strip().split()
            if len(sline) < 16:
                continue

            try:
                mag = float(sline[-1])
                etime = datetime.datetime.strptime(
                    f"{sline[3]}-{sline[4]}-{sline[5]} {sline[7]}:{sline[8]}:{sline[9]}.{sline[10]}",
                    "%Y-%m-%d %H:%M:%S.%f"
                )
                elon, elat, edep = float(sline[12]), float(sline[13]), float(sline[15])
            except Exception:
                continue

            # your spatial filter
            if elat < 29.0 or elat > 30.4:
                continue
            if elon < 101.8 or elon > 102.4:
                continue

            if etime < btime1 or etime > etime1:
                continue

            delta = (etime - btime1).total_seconds()
            events.append([elon, elat, edep, delta, mag])

    return np.asarray(events, dtype=np.float64)


def read2(fname, erange: List[float] = [15.0, 30.0]) -> np.ndarray:
    """
    Read our reloc results:
      run_fm3d/data/ours/reloc.real.True.txt
    Each event header line starts with "#", comma-separated:
      #id,time,lon,lat,dep,..., e1,e2,e3 ...
    You used e1,e2,e3 from p[11:14] and screen by eh,ez.
    Output: [lon, lat, dep, delta]
    """
    #fname = "run_fm3d/data/ours/reloc.real.True.v1.0.txt"
    events: List[List[float]] = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                continue
            p = line[1:].strip().split(",")
            if len(p) < 15:
                continue

            try:
                lon, lat, dep = map(float, p[2:5])
            except Exception:
                continue

            # uncertainty screening
            try:
                e1, e2, e3 = map(float, p[11:14])
                eh = math.sqrt(e1 * e1 + e2 * e2)
                ez = e3
                if eh > float(erange[0]) or ez > float(erange[1]):
                    continue
            except Exception:
                pass

            # parse time
            try:
                etime = datetime.datetime.strptime(p[1].strip(), "%Y-%m-%d %H:%M:%S.%f")
            except Exception:
                # allow ISO-like
                try:
                    etime = datetime.datetime.fromisoformat(p[1].strip())
                except Exception:
                    continue

            if etime < btime1 or etime > etime1:
                continue

            delta = (etime - btime1).total_seconds()
            events.append([lon, lat, dep, delta])

    return np.asarray(events, dtype=np.float64)

def read_nlloc_timeaware(path, erange=[5, 10], btime1=None, etime1=None):
    locs = []
    with open(path, "r") as f:
        f.readline()  # skip header
        for line in f:
            header = line.strip().split(",")
            if len(header) < 7:
                continue
            if len(header[1]) == 0:
                continue
            etime = datetime.datetime.strptime(header[1], "%Y-%m-%dT%H:%M:%S.%f")
            if (btime1 is not None and etime < btime1) or (etime1 is not None and etime > etime1):
                continue

            lat = float(header[2])
            lon = float(header[3])
            depth = float(header[4])
            e1 = float(header[5])
            e2 = float(header[6])
            if e1 > erange[0] or e2 > erange[1]:
                continue

            delta = (etime - btime1).total_seconds()
            locs.append([lon, lat, depth, delta, e1, e2])
    return np.array(locs, dtype=float)

def read_ours_timeaware(path, erange=[15, 30], btime1=None, etime1=None):
    events = []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("#"):
                continue
            p = line[1:].strip().split(",")
            if len(p) < 14:
                continue

            lon, lat, dep = map(float, p[2:5])
            # your file: e1,e2,e3 at p[11:14] (as you used before)
            e1, e2, e3 = map(float, p[11:14])
            eh = np.sqrt(e1**2 + e2**2)
            ez = e3

            etime = datetime.datetime.strptime(p[1], "%Y-%m-%d %H:%M:%S.%f")
            if (btime1 is not None and etime < btime1) or (etime1 is not None and etime > etime1):
                continue
            if eh > erange[0] or ez > erange[1]:
                continue

            delta = (etime - btime1).total_seconds()
            events.append([lon, lat, dep, delta, eh, ez])
    return np.array(events, dtype=float)

# ----------------------------
# 2) DEM reader (geo lon/lat grid)
# ----------------------------
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.windows import Window
from pyproj import CRS

def read_dem_geotiff_geo(
    dem_tif: str,
    bbox: tuple[float, float, float, float],
    dem_downsample: int = 4,
):
    """
    Robust DEM reader:
    - Use rasterio.windows.from_bounds() (no negative index issues)
    - Clip window to dataset bounds
    - Return lon_grid, lat_grid, Z_km (elevation in km)
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(f"Invalid bbox={bbox} (expect lon_min<lon_max, lat_min<lat_max)")

    with rasterio.open(dem_tif) as ds:
        if ds.crs is None:
            raise RuntimeError("DEM has no CRS metadata (ds.crs is None).")

        # 1) bbox in DEM CRS (use transform_bounds for correctness)
        if CRS.from_user_input(ds.crs).to_epsg() == 4326:
            left, bottom, right, top = lon_min, lat_min, lon_max, lat_max
        else:
            left, bottom, right, top = transform_bounds(
                CRS.from_epsg(4326),
                ds.crs,
                lon_min, lat_min, lon_max, lat_max,
                densify_pts=21
            )

        # 2) Build window from bounds in DEM CRS
        win = from_bounds(left, bottom, right, top, transform=ds.transform)

        # 3) Clip to raster extent to avoid negative indexes / overflow
        full = Window(col_off=0, row_off=0, width=ds.width, height=ds.height)
        win = win.intersection(full)

        if win.width <= 1 or win.height <= 1:
            raise RuntimeError(
                "Requested bbox does not intersect DEM raster (after CRS transform). "
                f"bbox={bbox}, dem_bounds={ds.bounds}, dem_crs={ds.crs}"
            )

        # 4) Read data
        dem = ds.read(1, window=win, masked=True).astype(np.float64)

        # 5) Downsample (cheap stride). Keep transform consistent.
        transform = ds.window_transform(win)
        if dem_downsample and dem_downsample > 1:
            step = int(dem_downsample)
            dem = dem[::step, ::step]
            transform = transform * rasterio.Affine.scale(step, step)

        # 6) Build lon/lat grid for plotting
        nrows, ncols = dem.shape
        rows = np.arange(nrows)
        cols = np.arange(ncols)
        cc, rr = np.meshgrid(cols, rows)

        xs, ys = rasterio.transform.xy(transform, rr, cc)
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)

        # Convert DEM CRS -> lon/lat if needed
        if CRS.from_user_input(ds.crs).to_epsg() == 4326:
            lon_grid, lat_grid = xs, ys
        else:
            from rasterio.warp import transform as rio_transform
            lon, lat = rio_transform(ds.crs, CRS.from_epsg(4326), xs.ravel(), ys.ravel())
            lon_grid = np.asarray(lon, float).reshape(nrows, ncols)
            lat_grid = np.asarray(lat, float).reshape(nrows, ncols)

        # 7) Elevation in km, keep NaNs
        Z = dem / 1000.0
        Z = np.where(np.ma.getmaskarray(dem), np.nan, Z)

    return lon_grid, lat_grid, Z


# ----------------------------
# 3) JGR-ish plotting (Cartopy)
# ----------------------------
def set_jgr_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 300,
    })


def panel_label(ax, label: str, n: int):
    ax.text(
        0.02, 0.98, f"{label} N={n}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.2", alpha=0.85),
        zorder=50,
    )


def format_geo_axes(ax, xlim, ylim):
    ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=ccrs.PlateCarree())

    # subtle context features
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.4, edgecolor="0.35")
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.35, edgecolor="0.40")
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.25, edgecolor="0.55", alpha=0.30)

    # gridlines with labels (JGR-like subtle)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.4,
        color="0.6",
        alpha=0.35,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = MaxNLocator(4)
    gl.ylocator = MaxNLocator(4)

    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".2f"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".2f"))


def plot_dem_gray(ax, lon_dem, lat_dem, Z_km, dem_alpha=0.45):
    Z = np.asarray(Z_km, float)
    if not np.isfinite(Z).any():
        return None
    v0, v1 = np.nanpercentile(Z, [2, 98])
    im = ax.pcolormesh(
        lon_dem, lat_dem, Z,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap="Greys_r",
        vmin=v0, vmax=v1,
        alpha=dem_alpha,
        zorder=1,
    )
    return im


def plot_eq_depth(ax, lons, lats, deps, vmin=0.0, vmax=30.0, size=7.0):
    sc = ax.scatter(
        lons, lats,
        c=deps,
        s=size,
        cmap="cividis",     # grayscale-friendly depth
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        linewidths=0,
        alpha=0.85,
        zorder=10,
    )
    return sc


def fig_location_comparison_cartopy(
    *,
    dem_tif: str,
    bbox: Tuple[float, float, float, float],
    locs_a: np.ndarray,
    locs_b: np.ndarray,
    locs_c: np.ndarray,
    locs_d: np.ndarray,
    depth_range: Tuple[float, float] = (0.0, 30.0),
    dem_downsample: int = 4,
    dem_alpha: float = 0.45,
    out_pdf: str = "figures/figs/fig_location_comparison.v1.1_cartopy.pdf",
    out_png: str = "figures/figs/fig_location_comparison.v1.1_cartopy.png",
):
    set_jgr_style()

    lon_min, lon_max, lat_min, lat_max = bbox
    xlim = (lon_min, lon_max)
    ylim = (lat_min, lat_max)
    vmin, vmax = depth_range

    # DEM (geo grid)
    lon_dem, lat_dem, Zdem = read_dem_geotiff_geo(dem_tif, bbox=bbox, dem_downsample=int(dem_downsample))

    proj = ccrs.PlateCarree()

    # JGR-friendly size: ~double column width
    fig = plt.figure(figsize=(6.9, 6.5))
    gs = fig.add_gridspec(2, 2, wspace=0.08, hspace=0.10)

    panels = [
        ("(a)", locs_a, gs[0, 0]),
        ("(b)", locs_b, gs[0, 1]),
        ("(c)", locs_c, gs[1, 0]),
        ("(d)", locs_d, gs[1, 1]),
    ]

    last_sc = None
    for lab, locs, slot in panels:
        ax = fig.add_subplot(slot, projection=proj)

        # DEM background
        plot_dem_hillshade(ax, lon_dem, lat_dem, Zdem)

        # EQ points + N
        n_show = 0
        if locs is not None and len(locs) > 0:
            locs = np.asarray(locs, float)
            lons = locs[:, 0]
            lats = locs[:, 1]
            deps = locs[:, 2]

            m = (
                np.isfinite(lons) & np.isfinite(lats) & np.isfinite(deps) &
                (deps >= vmin) & (deps <= vmax)
            )
            n_show = int(len(deps))
            lons = lons[m]; lats = lats[m]; deps = deps[m]
            

            if n_show > 0:
                last_sc = plot_eq_depth(ax, lons, lats, deps, vmin=vmin, vmax=vmax, size=7.0)

        panel_label(ax, lab, n_show)

        format_geo_axes(ax, xlim, ylim)

        # clean labels (JGR-style): only left column y-label, bottom row x-label
        if lab in ["(a)", "(b)"]:
            ax.set_xlabel("")
        if lab in ["(b)", "(d)"]:
            ax.set_ylabel("")

    # Shared depth colorbar
    if last_sc is not None:
        cax = fig.add_axes([0.18, 0.05, 0.64, 0.025])
        cb = fig.colorbar(last_sc, cax=cax, orientation="horizontal")
        cb.set_label("Depth (km)")
        cb.set_ticks([vmin, 10, 20, vmax])

    # ensure output dirs exist
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_pdf}")
    print(f"[save] {out_png}")

from matplotlib.colors import LightSource

def plot_dem_hillshade(ax, lon_dem, lat_dem, Z_km, *, alpha=0.35):
    Z = np.asarray(Z_km, float)
    if not np.isfinite(Z).any():
        return None
    # 轻微拉伸，避免极端值影响阴影
    v0, v1 = np.nanpercentile(Z, [2, 98])
    Zc = np.clip(Z, v0, v1)

    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(Zc, vert_exag=1.5, dx=1.0, dy=1.0)

    im = ax.pcolormesh(
        lon_dem, lat_dem, shade,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap="cividis",
        vmin=0, vmax=1,
        alpha=alpha,
        zorder=1
    )
    return im
# ----------------------------
# 4) Main
# ----------------------------
def main():
    # ---- user config ----
    dem_tif = "figures/dem.tif"
    bbox = (101.6, 102.4, 29.2, 30.1)  # (lon_min, lon_max, lat_min, lat_max)
    depth_range = (0.0, 30.0)

    # ---- read catalogs ----
    # (a) CENC .pha-like (returns [lon,lat,dep,delta,mag])
    locs_a = readctlg()
    locs_a = locs_a[:, :3] if locs_a.size else locs_a

    # (b) Liu's relocated catalog (returns [lon,lat,dep,delta])
    locs_b = read_event_file("data/Catalog_1.Pha")
    locs_b = locs_b[:, :3] if locs_b.size else locs_b

    # (c) NonLinLoc (returns [lon,lat,dep])
    locs_c = read_nlloc_timeaware("run_fm3d/data/nlloc_sc/all.locfiles.csv", btime1=btime1, etime1=etime1, erange=[3.2, 6.4])
    locs_c = locs_c[:, :3] if locs_c.size else locs_c
    print(f"[read] {len(locs_c)} events")
    # (d) Our method (returns [lon,lat,dep,delta])
    locs_d = read_ours_timeaware("run_fm3d/data/ours/reloc.real.True.v1.0.txt", btime1=btime1, etime1=etime1, erange=[10, 20])
    locs_d = locs_d[:, :3] if locs_d.size else locs_d
    print(f"[read] {len(locs_d)} events")
    # ---- plot ----
    fig_location_comparison_cartopy(
        dem_tif=dem_tif,
        bbox=bbox,
        locs_a=locs_a,
        locs_b=locs_b,
        locs_c=locs_c,
        locs_d=locs_d,
        depth_range=depth_range,
        dem_downsample=4,
        dem_alpha=0.45,
        out_pdf="figures/figs/fig_location_comparison.v1.1_cartopy.pdf",
        out_png="figures/figs/fig_location_comparison.v1.1_cartopy.png",
    )


if __name__ == "__main__":
    main()