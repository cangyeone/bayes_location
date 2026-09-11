"""Generate supervised labels on a Cartesian 3-D velocity grid, with skfmm."""

from __future__ import annotations

import json

import numpy as np

from .io import read_json, save_pairs, sha256, validate_geometry, write_json


def read_grid(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k], dtype=np.float64) for k in ("x_km", "y_km", "z_km", "vp", "vs")}
        if "geometry_json" in data:
            arrays["geometry"] = validate_geometry(json.loads(str(data["geometry_json"].item())))
    spacing = []
    for name in ("z_km", "y_km", "x_km"):
        axis = arrays[name]
        if axis.ndim != 1 or len(axis) < 3 or not np.isfinite(axis).all():
            raise ValueError(f"{name}: at least 3 finite grid nodes required")
        differences = np.diff(axis)
        if differences[0] <= 0 or not np.allclose(differences, differences[0], rtol=1e-5, atol=1e-6):
            raise ValueError(f"{name}: axis must be increasing and evenly spaced")
        spacing.append(float(differences[0]))
    expected = tuple(len(arrays[k]) for k in ("z_km", "y_km", "x_km"))
    for phase in ("vp", "vs"):
        speed = arrays[phase]
        if speed.shape != expected or not np.isfinite(speed).all() or np.any(speed <= 0):
            raise ValueError(f"{phase}: require positive finite km/s speeds with shape (Nz,Ny,Nx)={expected}")
    if np.any(arrays["vp"] <= arrays["vs"]):
        raise ValueError("vp must exceed vs at each grid node")
    return arrays, tuple(spacing)


def travel_time_field(speed, source_zyx, spacing):
    try:
        import skfmm
    except ImportError as exc:
        raise RuntimeError('FMM requires: pip install -e ".[fmm]"') from exc
    phi = np.ones(speed.shape, dtype=np.float64)
    # A zero-valued source node explicitly sets T(source)=0. Legacy generators
    # use one negative voxel and hence a different, half-cell zero interface.
    phi[tuple(source_zyx)] = 0.0
    return np.asarray(skfmm.travel_time(phi, speed=speed, dx=spacing), dtype=float)


def generate(args):
    if args.n_sources < 2 or args.receivers_per_source < 1 or args.noise_s < 0:
        raise ValueError("require >=2 sources, >=1 receiver/source and nonnegative label noise")
    grid, spacing = read_grid(args.grid)
    meta = validate_geometry(read_json(args.geometry))
    if "geometry" in grid:
        for key in ("projection", "lon0", "lat0", "vertical_datum"):
            if grid["geometry"].get(key) != meta.get(key):
                raise ValueError("velocity grid and geometry use different projections or vertical datums")
    axes = [grid[k] for k in ("z_km", "y_km", "x_km")]
    source_bounds = np.asarray(meta["source_bounds_km"])[::-1]
    receiver_bounds = np.asarray(meta["receiver_bounds_km"])[::-1]
    for b in (source_bounds, receiver_bounds):
        for axis, bounds in zip(axes, b):
            if bounds[0] < axis[0] or bounds[1] > axis[-1]:
                raise ValueError("configured source/receiver bounds exceed the velocity grid")
    source_axes = [np.flatnonzero((a >= b[0]) & (a <= b[1])) for a, b in zip(axes, source_bounds)]
    receiver_axes = [np.flatnonzero((a >= b[0]) & (a <= b[1])) for a, b in zip(axes, receiver_bounds)]
    ns = int(np.prod([len(a) for a in source_axes]))
    nr = int(np.prod([len(a) for a in receiver_axes]))
    if args.n_sources > ns or args.receivers_per_source > nr:
        raise ValueError(f"not enough distinct grid nodes: sources={ns}, receivers={nr}")
    rng = np.random.default_rng(args.seed)

    def decode(flat, choices):
        indices = np.unravel_index(flat, tuple(len(a) for a in choices))
        return np.column_stack([a[i] for a, i in zip(choices, indices)])

    source_nodes = decode(rng.choice(ns, args.n_sources, replace=False), source_axes)
    output_xr, output_xs, output_t, event_ids = [], [], [], []
    for i, source in enumerate(source_nodes):
        receivers = decode(rng.choice(nr, args.receivers_per_source, replace=False), receiver_axes)
        times = np.column_stack([
            travel_time_field(grid[p], source, spacing)[tuple(receivers.T)] for p in ("vp", "vs")
        ])
        times += rng.normal(0.0, args.noise_s, times.shape)
        # Skip coincident source/receiver pairs; no artificial zero/noise clipping.
        valid = ~np.all(receivers == source, axis=1) & (times >= 0).all(axis=1)
        receivers, times = receivers[valid], times[valid]
        xyz_source = np.array([axes[j][source[j]] for j in (2, 1, 0)])
        xyz_receivers = np.column_stack([axes[j][receivers[:, j]] for j in (2, 1, 0)])
        output_xr.append(xyz_receivers)
        output_xs.append(np.broadcast_to(xyz_source, (len(times), 3)))
        output_t.append(times)
        event_ids.extend([f"fmm-{i:06d}"] * len(times))
        if (i + 1) % max(1, args.n_sources // 10) == 0:
            print(f"FMM sources {i + 1}/{args.n_sources}", flush=True)
    labels = np.concatenate(output_t)
    if not len(labels):
        raise ValueError("no noncoincident, nonnegative travel-time pairs were generated")
    save_pairs(args.output, xr=np.concatenate(output_xr).astype(np.float32),
               xs=np.concatenate(output_xs).astype(np.float32),
               tp=labels[:, 0].astype(np.float32), ts=labels[:, 1].astype(np.float32),
               event_id=event_ids, geometry=meta)
    write_json(str(args.output) + ".provenance.json", {
        "grid_sha256": sha256(args.grid), "geometry": meta, "seed": args.seed,
        "n_sources": args.n_sources, "pairs": len(labels), "noise_sd_s": args.noise_s,
        "solver": "skfmm.travel_time", "source_boundary": "single zero node, T(source)=0",
        "spacing_zyx_km": list(spacing), "gross_error_fraction": 0,
    })
