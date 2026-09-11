"""Convert SWChinaCVM-style profiles to a Cartesian velocity grid for FMM."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError

from .io import read_json, sha256, transformers, validate_geometry, write_json


def read_model(path):
    """Read lon/lat/dep/Vp/Vs, requiring complete shared-depth profiles.

    Horizontal locations may have an irregular footprint. Input ordering is
    irrelevant; duplicate points and missing profile layers are rejected.
    Error messages deliberately omit geographic values and raw input rows.
    """
    rows = []
    with open(path, encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            if not rows and [v.lower() for v in fields] == ["lon", "lat", "dep", "vp", "vs"]:
                continue
            if len(fields) != 5:
                raise ValueError(f"model line {number}: expected five whitespace-separated columns")
            try:
                rows.append([float(value) for value in fields])
            except ValueError:
                raise ValueError(f"model line {number}: expected numeric lon/lat/dep/Vp/Vs") from None
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or len(data) < 6 or not np.isfinite(data).all():
        raise ValueError("model must contain at least six finite five-column rows")
    if np.any(np.abs(data[:, 0]) > 180) or np.any(np.abs(data[:, 1]) >= 90):
        raise ValueError("model contains invalid geographic coordinates")
    if np.ptp(data[:, 0]) >= 180:
        raise ValueError("dateline-spanning models require a separate longitude-unwrapping policy")
    if np.any(data[:, 3:] <= 0) or np.any(data[:, 3] <= data[:, 4]):
        raise ValueError("model requires positive km/s speeds with Vp greater than Vs")
    lonlat, horizontal_index = np.unique(data[:, :2], axis=0, return_inverse=True)
    depths, depth_index = np.unique(data[:, 2], return_inverse=True)
    if len(lonlat) < 3 or len(depths) < 2:
        raise ValueError("model requires at least three horizontal profiles and two depths")
    index = horizontal_index * len(depths) + depth_index
    if len(np.unique(index)) != len(data):
        raise ValueError("model contains duplicate horizontal-location/depth rows")
    if len(data) != len(lonlat) * len(depths):
        raise ValueError("model requires the same complete depth levels at every horizontal location")
    speeds = np.empty((len(lonlat), len(depths), 2), dtype=float)
    speeds[horizontal_index, depth_index] = data[:, 3:]
    return lonlat, depths, speeds


def axis_count(bounds, spacing):
    lower, upper = map(float, bounds)
    if not np.isfinite([lower, upper, spacing]).all() or spacing <= 0 or lower >= upper:
        raise ValueError("grid bounds must increase and spacing must be finite and positive")
    cells = (upper - lower) / spacing
    if not math.isfinite(cells) or cells < 2 or not np.isclose(cells, round(cells), rtol=0, atol=1e-8):
        raise ValueError("each grid extent must be an integer multiple of spacing with at least three nodes")
    return int(round(cells)) + 1


def interpolate_grid(lonlat, depths, speeds, geometry, x, y, z):
    """Linear interpolation of velocity in projected x/y and then in depth.

    No extrapolation or nearest-neighbour fill is applied. The supported
    horizontal area is the convex hull, which can bridge unsampled gaps.
    """
    if z[0] < depths[0] or z[-1] > depths[-1]:
        raise ValueError("requested depths exceed the model; vertical extrapolation is disabled")
    east, north = transformers(geometry)[0].transform(lonlat[:, 0], lonlat[:, 1])
    xy = np.column_stack([east, north]) / 1000.0
    if not np.isfinite(xy).all():
        raise ValueError("model projection produced nonfinite coordinates")
    try:
        horizontal = LinearNDInterpolator(xy, speeds, fill_value=np.nan)
    except QhullError:
        raise ValueError("model horizontal locations do not support a two-dimensional triangulation") from None
    low = np.clip(np.searchsorted(depths, z, side="right") - 1, 0, len(depths) - 2)
    weight = ((z - depths[low]) / (depths[low + 1] - depths[low]))[None, :, None]
    result = np.empty((len(z), len(y) * len(x), 2), dtype=float)
    for start in range(0, len(y) * len(x), 16384):
        flat = np.arange(start, min(start + 16384, len(y) * len(x)))
        points = np.column_stack([x[flat % len(x)], y[flat // len(x)]])
        profiles = horizontal(points)
        if not np.isfinite(profiles).all():
            raise ValueError("requested x/y grid exceeds the model convex hull; shrink bounds or change geometry")
        values = profiles[:, low] * (1 - weight) + profiles[:, low + 1] * weight
        result[:, flat, :] = values.transpose(1, 0, 2)
    result = result.reshape(len(z), len(y), len(x), 2)
    if not np.isfinite(result).all() or np.any(result <= 0) or np.any(result[..., 0] <= result[..., 1]):
        raise ValueError("interpolated grid does not satisfy finite positive Vp > Vs")
    return result[..., 0], result[..., 1]


def build_grid(args):
    """Write velocity.npz, matching geometry.json, and conversion provenance."""
    if args.depth_reference != "sea-level":
        raise ValueError("use sea-level depths; surface-relative depths need an explicit topography conversion")
    if Path(args.model).name == "SWChinaCVMv2.0.txt":
        raise ValueError("SWChinaCVMv2.0.txt uses surface-relative depth; choose SWChinaCVMv2.0.txt.wrst.sea_level")
    out = Path(args.output)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("grid output must be a new or empty directory")
    bounds = [args.xlim_km, args.ylim_km, args.zlim_km]
    counts = [axis_count(b, h) for b, h in zip(bounds, args.spacing_km)]
    if args.max_nodes < 1 or math.prod(counts) > args.max_nodes:
        raise ValueError("requested grid exceeds --max-nodes; coarsen spacing or reduce bounds")
    x, y, z = [np.linspace(b[0], b[1], n) for b, n in zip(bounds, counts)]
    lonlat, depths, speeds = read_model(args.model)
    if args.geometry:
        geometry = validate_geometry(read_json(args.geometry))
        if geometry.get("vertical_datum") != "mean_sea_level":
            raise ValueError("geometry.vertical_datum must be mean_sea_level for this model")
    else:
        center = (lonlat.min(axis=0) + lonlat.max(axis=0)) / 2
        geometry = {
            "projection": "AEQD", "lon0": float(center[0]), "lat0": float(center[1]),
            "vertical_datum": "mean_sea_level",
            "source_bounds_km": [list(map(float, b)) for b in bounds],
            "receiver_bounds_km": [list(map(float, b)) for b in bounds[:2]] + [[0.0, 0.0]],
        }
        validate_geometry(geometry)
    for name in ("source_bounds_km", "receiver_bounds_km"):
        for axis, (lower, upper) in zip((x, y, z), geometry[name]):
            if lower < axis[0] or upper > axis[-1] or not np.any((axis >= lower) & (axis <= upper)):
                raise ValueError(f"{name} must fit inside the grid and include grid nodes on every axis")
    vp, vs = interpolate_grid(lonlat, depths, speeds, geometry, x, y, z)
    provenance = {
        "source_file": Path(args.model).name, "source_sha256": sha256(args.model),
        "source_columns": ["longitude_deg", "latitude_deg", "depth_km", "vp_km_s", "vs_km_s"],
        "source_depth_reference": "mean_sea_level", "source_rows": int(speeds.shape[0] * speeds.shape[1]),
        "source_profiles": len(lonlat), "source_depth_levels": len(depths),
        "projection_origin": "provided_geometry" if args.geometry else "source_geographic_bounding_box_midpoint",
        "geometry": geometry, "shape_zyx": list(vp.shape), "spacing_xyz_km": list(args.spacing_km),
        "interpolation": "linear velocity on projected 2-D Delaunay triangles, then linear depth interpolation",
        "extrapolation": "disabled", "horizontal_coverage": "convex hull; unsampled gaps may be bridged",
        "terrain_mask": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "velocity.npz", x_km=x, y_km=y, z_km=z, vp=vp, vs=vs,
                        geometry_json=json.dumps(geometry))
    write_json(out / "geometry.json", geometry)
    write_json(out / "provenance.json", provenance)
    print(f"Saved velocity.npz, geometry.json, provenance.json: {out} (Nz,Ny,Nx)={vp.shape}")
