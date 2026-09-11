"""Portable location, event-level selection and geographic export."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from .io import (read_csv, read_json, read_observations, sha256, transformers,
                 validate_geometry, write_csv, write_json)
from .locator import MISSING, TravelTimeNet, run_sampler
from .multistart import run_multistart_initializer


def load_model(checkpoint, device, geometry_path=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    meta = payload.get("geometry")
    supplied = validate_geometry(read_json(geometry_path)) if geometry_path else None
    if meta is None:
        if supplied is None:
            raise ValueError("legacy checkpoint has no domain metadata; supply --geometry explicitly")
        meta = supplied
        legacy_projection = payload.get("meta", {}).get("projection_meta", {})
        for key in ("lon0", "lat0"):
            if key in legacy_projection and float(legacy_projection[key]) != float(meta[key]):
                raise ValueError("geometry conflicts with the legacy checkpoint projection")
    elif supplied is not None and meta != supplied:
        raise ValueError("geometry differs from checkpoint metadata; retrain for another domain")
    validate_geometry(meta)
    model = TravelTimeNet(int(payload.get("model_hidden_dim", 256)))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, meta


def split_rhat(values):
    """Classical split R-hat; values=(chains,draws,events,parameters).

    This inexpensive diagnostic is not rank-normalized R-hat and is not ESS.
    NaN indicates insufficient draws; a zero within-chain variance gives inf.
    """
    chains, draws = values.shape[:2]
    if chains < 2 or draws < 4:
        return np.full(values.shape[2:], np.nan)
    n = draws // 2
    split = np.concatenate([values[:, :n], values[:, -n:]], axis=0).astype(float)
    w = split.var(axis=1, ddof=1).mean(axis=0)
    b = n * split.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.sqrt(((n - 1) / n * w + b / n) / w)
    return np.where(w > 0, result, np.inf)


def locate(args, device):
    if args.chains < 1 or args.min_stations < 1 or args.min_phases < 1:
        raise ValueError("chains/min_stations/min_phases must be positive")
    if args.samples <= args.burn or args.burn < 0 or args.thin < 1:
        raise ValueError("require samples > burn >= 0 and thin >= 1")
    if args.pool_size < 6 or not 1 <= args.refine_top <= args.pool_size or args.init_steps < 0:
        raise ValueError("require pool_size >= 6, 1 <= refine_top <= pool_size and init_steps >= 0")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty; choose a new run directory: {out}")
    model, meta = load_model(args.checkpoint, device, args.geometry)
    data = read_observations(args.stations, args.events, args.picks, meta)
    counts = np.bincount(data["event_ids"], minlength=len(data["events"]))
    phase_counts = np.bincount(
        data["event_ids"], weights=(data["tp"] != MISSING).astype(int) + (data["ts"] != MISSING),
        minlength=len(data["events"]),
    )
    accepted = (counts >= args.min_stations) & (phase_counts >= args.min_phases)
    rejected = []
    for i in np.flatnonzero(~accepted):
        rejected.append(dict(event_id=data["events"][i]["event_id"], n_stations=int(counts[i]),
                             n_phases=int(phase_counts[i]), reason="insufficient_observations"))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "rejected_events.csv", rejected, ["event_id", "n_stations", "n_phases", "reason"])
    if not accepted.any():
        raise ValueError(f"no events pass input-count criteria; see {out / 'rejected_events.csv'}")
    original_ids = np.flatnonzero(accepted)
    keep = accepted[data["event_ids"]]
    remap = np.cumsum(accepted) - 1
    ids = remap[data["event_ids"][keep]].astype(np.int64)
    obs = {k: data[k][keep] for k in ("xr", "tp", "ts")}
    bounds = np.asarray(meta["source_bounds_km"], dtype=float)
    initial = run_multistart_initializer(
        model=model, **obs, event_ids=ids, n_events=len(original_ids), seed=args.seed,
        pool_size=args.pool_size, candidate_strategy="hybrid", refine_top_k=args.refine_top,
        return_ranks=min(3, args.refine_top), steps=args.init_steps,
        x_bounds=tuple(bounds[0]), y_bounds=tuple(bounds[1]), z_bounds=tuple(bounds[2]),
    )
    np.savez_compressed(out / "initialization.npz", **{
        k: initial[k] for k in ("ranked_sources", "ranked_t0", "ranked_scores", "ranked_raw_candidate_index")
    })
    chains = []
    names = [data["events"][i]["event_id"] for i in original_ids]
    timings = []
    # Independent RNGs in the best selected basin, as in the controlling local
    # workflow. This is not a claim of exploring/mixing all remote modes.
    for chain in range(args.chains):
        result = run_sampler(
            model=model, **obs, event_ids=ids, n_events=len(original_ids),
            xs_init=initial["ranked_sources"][0], t0_init=initial["ranked_t0"][0],
            mode=args.mode, seed=args.seed + chain + 1, n_samples=args.samples,
            burn=args.burn, thin=args.thin, physical_bounds=tuple(map(tuple, bounds)),
            nu=args.nu, prop_scale=args.proposal_km,
        )
        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(out / f"chain_{chain:02d}.npz", **arrays, event_id=np.asarray(names),
                            row_event_index=ids, row_station_index=data["station_indices"][keep])
        chains.append(result)
        timings.append(result["elapsed_seconds"])
    xs = np.stack([c["xs_samples"] for c in chains])
    t0 = np.stack([c["t0_samples"] for c in chains])
    flat = xs.reshape(-1, len(names), 3)
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    q05, q95 = np.quantile(flat, [0.05, 0.95], axis=0)
    widths = q95 - q05
    rhat = split_rhat(np.concatenate([xs, t0[..., None]], axis=-1))
    lon, lat = transformers(meta)[1].transform((mean[:, 0] * 1000.0).tolist(), (mean[:, 1] * 1000.0).tolist())
    rows = []
    for i, original in enumerate(original_ids):
        time_mean = float(t0[:, :, i].mean())
        row = dict(event_id=names[i], reference_time=data["reference_times"][original].isoformat(),
                   origin_time=(data["reference_times"][original] + dt.timedelta(seconds=time_mean)).isoformat(),
                   longitude=float(lon[i]), latitude=float(lat[i]), depth_km=float(mean[i, 2]),
                   t0_mean_s=time_mean, t0_std_s=float(t0[:, :, i].std()),
                   n_stations=int(counts[original]), n_phases=int(phase_counts[original]),
                   acceptance_rate=float(np.mean([c["acceptance_rate"][i] for c in chains])))
        for j, axis in enumerate("xyz"):
            row.update({f"{axis}_mean_km": float(mean[i, j]), f"std_{axis}_km": float(std[i, j]),
                        f"q05_{axis}_km": float(q05[i, j]), f"q95_{axis}_km": float(q95[i, j]),
                        f"width_{axis}90_km": float(widths[i, j])})
        for j, parameter in enumerate(("x", "y", "z", "t0")):
            row[f"rhat_split_{parameter}"] = float(rhat[i, j]) if not np.isnan(rhat[i, j]) else ""
        row["width_h90_km"] = float(np.hypot(widths[i, 0], widths[i, 1]))
        rows.append(row)
    write_csv(out / "catalog.csv", rows)
    pick_rows = []
    pin = np.mean([c["inlier_probability_p"] for c in chains], axis=0)
    sin = np.mean([c["inlier_probability_s"] for c in chains], axis=0)
    for row_index, original_row in enumerate(np.flatnonzero(keep)):
        for phase, key, probabilities in (("P", "tp", pin), ("S", "ts", sin)):
            if obs[key][row_index] == MISSING:
                continue
            pick_rows.append(dict(event_id=names[ids[row_index]],
                                  station_id=data["stations"][data["station_indices"][original_row]]["station_id"],
                                  phase=phase, relative_arrival_s=float(obs[key][row_index]),
                                  inlier_probability=float(probabilities[row_index]) if args.mode == "student_t_z" else ""))
    write_csv(out / "pick_quality.csv", pick_rows)
    write_json(out / "run.json", {
        "arguments": vars(args), "geometry": meta, "device": str(device),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "input_sha256": {k: sha256(getattr(args, k)) for k in ("checkpoint", "stations", "events", "picks")},
        "input_events": len(data["events"]), "located_events": len(names), "rejected_events": len(rejected),
        "seed_initializer": args.seed, "seeds_chains": [args.seed + i + 1 for i in range(args.chains)],
        "retained_draws_per_chain": len(chains[0]["xs_samples"]),
        "initialization_seconds": initial["runtime"]["total_seconds"], "sampling_seconds": timings,
        "summary": "pooled chains initialized in the best ranked basin per event; global modes not guaranteed",
    })
    print(f"Located {len(names)} events; rejected {len(rejected)}. Results: {out / 'catalog.csv'}")


def score_and_select(rows, *, top_k=None, max_h=None, max_z=None, max_rhat=None):
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be nonnegative")
    for value in (max_h, max_z, max_rhat):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError("QC thresholds must be finite and positive")
    if len({r["event_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate event IDs in QC catalog")
    valid = []
    for row in rows:
        row.update(qc_score="", retained=False, rejection_reason="", width_h90_km="")
        try:
            widths = [float(row[f"width_{a}90_km"]) for a in "xyz"]
        except (ValueError, KeyError):
            widths = [np.nan] * 3
        if not np.isfinite(widths).all() or min(widths) < 0:
            row["rejection_reason"] = "invalid_width"
            continue
        row["width_h90_km"] = float(np.hypot(*widths[:2]))
        valid.append(row)
    n = len(valid)
    if n:
        h = rankdata([r["width_h90_km"] for r in valid], method="average") / n
        z = rankdata([float(r["width_z90_km"]) for r in valid], method="average") / n
        for row, score in zip(valid, np.maximum(h, z)):
            row["qc_score"] = float(score)
            reasons = []
            if max_h is not None and row["width_h90_km"] > max_h:
                reasons.append("horizontal_width")
            if max_z is not None and float(row["width_z90_km"]) > max_z:
                reasons.append("depth_width")
            if max_rhat is not None:
                rhats = [float(row.get(f"rhat_split_{p}", "") or "nan") for p in ("x", "y", "z", "t0")]
                if not np.isfinite(rhats).all() or max(rhats) > max_rhat:
                    reasons.append("rhat")
            row["rejection_reason"] = ";".join(reasons)
    eligible = sorted([r for r in valid if not r["rejection_reason"]], key=lambda r: (r["qc_score"], r["event_id"]))
    for rank, row in enumerate(eligible):
        row["retained"] = top_k is None or rank < top_k
        if not row["retained"]:
            row["rejection_reason"] = "above_top_k"
    return rows


def filter_catalog(args):
    rows = read_csv(args.catalog, ["event_id", "width_x90_km", "width_y90_km", "width_z90_km"])
    scored = score_and_select(rows, top_k=args.top_k, max_h=args.max_horizontal_km,
                              max_z=args.max_depth_km, max_rhat=args.max_rhat)
    selected = sorted([r for r in scored if r["retained"]], key=lambda r: (r["qc_score"], r["event_id"]))
    fields = list(scored[0])
    write_csv(args.output, selected, fields)
    write_csv(str(args.output) + ".audit.csv", scored, fields)
    write_json(str(args.output) + ".qc.json", {"arguments": vars(args), "input_sha256": sha256(args.catalog),
                                               "input_events": len(rows), "retained_events": len(selected)})
    print(f"Retained {len(selected)}/{len(rows)} events: {args.output}")


def reproject(args):
    meta = validate_geometry(read_json(args.geometry))
    rows = read_csv(args.catalog, ["x_mean_km", "y_mean_km", "z_mean_km"])
    xyz = np.array([[float(r[f"{axis}_mean_km"]) for axis in "xyz"] for r in rows])
    if not np.isfinite(xyz).all():
        raise ValueError("nonfinite projected coordinates")
    lon, lat = transformers(meta)[1].transform((xyz[:, 0] * 1000.0).tolist(), (xyz[:, 1] * 1000.0).tolist())
    for i, row in enumerate(rows):
        row.update(longitude=float(lon[i]), latitude=float(lat[i]), depth_km=float(xyz[i, 2]))
    write_csv(args.output, rows)
