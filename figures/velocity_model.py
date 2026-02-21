#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

def load_axes(prefix: str):
    ax = np.load(prefix + "_axes_km.npz")
    return ax["x_km"], ax["y_km"], ax["z_km"], float(ax["lon0"]), float(ax["lat0"])

def load_xyz(prefix: str):
    # (N,5): x_km, y_km, z_km, vp, vs
    return np.load(prefix + ".npy")

def find_nearest_depth(z_km_axis, z_target):
    iz = int(np.argmin(np.abs(z_km_axis - z_target)))
    return iz, float(z_km_axis[iz])

def reshape_layer(arr_xyz, x_km, y_km, z_sel):
    """
    从 (N,5) 中抽出 z=z_sel 的层，reshape 成 [Ny, Nx] 的 vp/vs。
    注意：build_velocity_grid 里是按 z 循环，每层按 meshgrid(x,y) 顺序写入。
    """
    Nx = len(x_km); Ny = len(y_km)
    # 允许浮点误差：用 isclose
    m = np.isclose(arr_xyz[:, 2], z_sel, rtol=0, atol=1e-6)
    layer = arr_xyz[m]
    if layer.shape[0] != Nx * Ny:
        raise RuntimeError(f"Layer size mismatch at z={z_sel}: got {layer.shape[0]}, expected {Nx*Ny}")
    vp = layer[:, 3].reshape(Ny, Nx)
    vs = layer[:, 4].reshape(Ny, Nx)
    return vp, vs

def make_lonlat_grid(x_km, y_km, lon0, lat0):
    crs_geod = CRS.from_epsg(4326)
    crs_aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    inv = Transformer.from_crs(crs_aeqd, crs_geod, always_xy=True)

    xx, yy = np.meshgrid(x_km, y_km, indexing="xy")
    lon, lat = inv.transform(xx * 1000.0, yy * 1000.0)
    return lon, lat

def jgr_axes_style(ax):
    # JGR 常见：简洁、线宽适中、刻度向内、上右边框可保留或去掉
    ax.tick_params(direction="in", top=True, right=True, length=4, width=0.8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

def plot_slice(prefix, z_target, field="vs", use_lonlat=True, panel="(a)", out_png=None, dpi=300):
    x_km, y_km, z_km, lon0, lat0 = load_axes(prefix)
    arr = load_xyz(prefix)

    iz, z_sel = find_nearest_depth(z_km, z_target)
    vp, vs = reshape_layer(arr, x_km, y_km, z_sel)

    data = vs if field.lower() == "vs" else vp
    cbar_label = "Vs (km/s)" if field.lower() == "vs" else "Vp (km/s)"

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    # 单栏宽 ~3.5 in；双栏宽 ~7.2 in。这里给双栏风格更通用
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    if use_lonlat:
        lon, lat = make_lonlat_grid(x_km, y_km, lon0, lat0)
        # pcolormesh 需要网格点；这里用中心点也可，视觉上足够
        im = ax.pcolormesh(lon, lat, data, shading="auto")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
    else:
        xx, yy = np.meshgrid(x_km, y_km, indexing="xy")
        im = ax.pcolormesh(xx, yy, data, shading="auto")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")

    # 等比例（JGR 地图类图一般建议等比例或接近等比例）
    ax.set_aspect("equal", adjustable="box")

    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
    cb.set_label(cbar_label)

    # panel label + depth 注记（JGR 常用左上角标注）
    ax.text(0.01, 0.99, f"{panel}  z = {z_sel:.1f} km",
            transform=ax.transAxes, ha="left", va="top")

    jgr_axes_style(ax)
    fig.tight_layout()

    if out_png is None:
        out_png = f"{prefix}_{field}_z{z_sel:.1f}km.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="run_fm3d/data/xyz_vp_vs",
                    help="path prefix without extension (expects .npy and _axes_km.npz)")
    ap.add_argument("--z", type=float, default=12, help="target depth in km (will snap to nearest z_km grid)")
    ap.add_argument("--field", choices=["vp", "vs"], default="vs")
    ap.add_argument("--lonlat", action="store_true", help="plot in lon/lat (recommended for JGR)")
    ap.add_argument("--panel", default="")
    ap.add_argument("--out", default="figures/figs/slice_vs_z12.png")
    args = ap.parse_args()

    plot_slice(args.prefix, args.z, field=args.field, use_lonlat=args.lonlat, panel=args.panel, out_png=args.out)
