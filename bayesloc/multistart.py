"""Grid-free, robust multi-start initialization for the revision locator.

The initializer never uses event truth.  It builds a small pool of
station-informed continuous starting points, profiles origin time with a
median, refines only the best few starts with a fixed Student-t objective, and
returns ranked candidate solutions. Different ranks need not be distinct basins.

Ported from robust_bayes_location_public/revision_experiments/multistart.py.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch


MISSING = -12345.0


def _event_slices(event_ids: np.ndarray, n_events: int) -> list[slice]:
    ids = np.asarray(event_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("event_ids must be a non-empty one-dimensional array")
    if ids.min() != 0 or ids.max() != n_events - 1:
        raise ValueError("event_ids are not dense in [0, n_events)")
    if np.any(ids[1:] < ids[:-1]):
        raise ValueError("multi-start initialization requires event-contiguous rows")
    counts = np.bincount(ids, minlength=n_events)
    offsets = np.r_[0, np.cumsum(counts)]
    return [slice(int(offsets[i]), int(offsets[i + 1])) for i in range(n_events)]


def build_blind_candidate_pool(
    *,
    xr: np.ndarray,
    tp: np.ndarray,
    ts: np.ndarray,
    event_ids: np.ndarray,
    n_events: int,
    pool_size: int,
    seed: int,
    strategy: str = "annulus",
    x_bounds: tuple[float, float] = (-685.0, 685.0),
    y_bounds: tuple[float, float] = (-815.0, 850.0),
    z_bounds: tuple[float, float] = (0.0, 70.0),
) -> tuple[np.ndarray, list[str]]:
    """Generate station-informed candidates without event truth or a spatial grid."""
    if pool_size < 6:
        raise ValueError("pool_size must be at least 6")
    if strategy not in {"annulus", "centroid_jitter", "hybrid"}:
        raise ValueError(f"unknown candidate strategy: {strategy}")
    xr = np.asarray(xr, dtype=np.float32)
    tp = np.asarray(tp, dtype=np.float32)
    ts = np.asarray(ts, dtype=np.float32)
    slices = _event_slices(event_ids, n_events)
    rng = np.random.default_rng(seed)

    centroid = np.empty((n_events, 2), dtype=np.float32)
    earliest_p = np.empty_like(centroid)
    earliest_s = np.empty_like(centroid)
    early_weighted = np.empty_like(centroid)
    exploration_xy = np.empty((max(pool_size - 6, 0), n_events, 2), dtype=np.float32)
    exploration_depth = np.empty((max(pool_size - 6, 0), n_events), dtype=np.float32)
    if strategy == "annulus":
        annulus_slots = list(range(exploration_xy.shape[0]))
    elif strategy == "hybrid":
        annulus_slots = [index for index in range(exploration_xy.shape[0]) if index % 2 == 0]
    else:
        annulus_slots = []
    annulus_slot_rank = {slot: rank for rank, slot in enumerate(annulus_slots)}

    for event_index, slc in enumerate(slices):
        receiver = xr[slc]
        p = tp[slc]
        s = ts[slc]
        mask_p = p != MISSING
        mask_s = s != MISSING
        valid = mask_p | mask_s
        if not np.any(valid):
            raise ValueError(f"event {event_index} has no valid phase")
        receiver = receiver[valid]
        p = p[valid]
        s = s[valid]
        mask_p = p != MISSING
        mask_s = s != MISSING
        xy = receiver[:, :2]

        centroid[event_index] = xy.mean(axis=0)
        earliest_p[event_index] = (
            xy[np.flatnonzero(mask_p)[np.argmin(p[mask_p])]]
            if np.any(mask_p)
            else centroid[event_index]
        )
        earliest_s[event_index] = (
            xy[np.flatnonzero(mask_s)[np.argmin(s[mask_s])]]
            if np.any(mask_s)
            else earliest_p[event_index]
        )

        weight = np.zeros(len(xy), dtype=np.float64)
        if np.any(mask_p):
            weight[mask_p] += np.exp(-(p[mask_p] - p[mask_p].min()) / 8.0)
        if np.any(mask_s):
            weight[mask_s] += np.exp(-(s[mask_s] - s[mask_s].min()) / 12.0)
        early_weighted[event_index] = np.average(xy, axis=0, weights=weight)

        paired = mask_p & mask_s
        plausible_pair = paired & ((s - p) > 0.0) & ((s - p) < 70.0)
        if np.any(plausible_pair):
            paired_indexes = np.flatnonzero(plausible_pair)
            anchor_index = paired_indexes[np.argmin(p[paired_indexes])]
            anchor = xy[anchor_index]
            # A P/S differential removes origin time.  These average velocities
            # only set the radius of diverse starts; the 3-D neural travel-time
            # model supplies the actual objective during refinement.
            inverse_velocity_difference = 1.0 / 3.5 - 1.0 / 6.0
            radius = float((s[anchor_index] - p[anchor_index]) / inverse_velocity_difference)
            radius = float(np.clip(radius, 20.0, 250.0))
        else:
            anchor = earliest_p[event_index]
            radius = 150.0
        phase_offset = float(rng.uniform(0.0, 2.0 * math.pi))
        jitter_index = 0
        for exploration_index in range(exploration_xy.shape[0]):
            if exploration_index in annulus_slot_rank:
                ring_index = annulus_slot_rank[exploration_index]
                angle = phase_offset + 2.0 * math.pi * ring_index / len(annulus_slots)
                exploration_xy[exploration_index, event_index] = anchor + radius * np.array(
                    [math.cos(angle), math.sin(angle)], dtype=np.float32
                )
                exploration_depth[exploration_index, event_index] = 10.0
            else:
                scale = (25.0, 75.0, 150.0, 250.0)[jitter_index % 4]
                exploration_xy[exploration_index, event_index] = centroid[event_index] + rng.normal(
                    0.0, scale, size=2
                ).astype(np.float32)
                exploration_depth[exploration_index, event_index] = float(
                    rng.uniform(0.0, 40.0)
                )
                jitter_index += 1

    definitions: list[tuple[str, np.ndarray, float]] = [
        ("centroid_z5", centroid, 5.0),
        ("centroid_z20", centroid, 20.0),
        ("earliest_p_z5", earliest_p, 5.0),
        ("earliest_p_z20", earliest_p, 20.0),
        ("earliest_s_z10", earliest_s, 10.0),
        ("early_weighted_z10", early_weighted, 10.0),
    ]
    candidates = np.empty((pool_size, n_events, 3), dtype=np.float32)
    labels: list[str] = []
    for candidate_index, (label, xy, depth) in enumerate(definitions):
        candidates[candidate_index, :, :2] = xy
        candidates[candidate_index, :, 2] = depth
        labels.append(label)
    jitter_index = 0
    for exploration_index in range(exploration_xy.shape[0]):
        candidate_index = 6 + exploration_index
        candidates[candidate_index, :, :2] = exploration_xy[exploration_index]
        candidates[candidate_index, :, 2] = exploration_depth[exploration_index]
        if exploration_index in annulus_slot_rank:
            labels.append(f"sp_annulus_direction_{annulus_slot_rank[exploration_index] + 1}")
        else:
            scale = (25, 75, 150, 250)[jitter_index % 4]
            labels.append(f"centroid_jitter_sigma{scale}_{jitter_index + 1}")
            jitter_index += 1

    candidates[:, :, 0] = np.clip(candidates[:, :, 0], *x_bounds)
    candidates[:, :, 1] = np.clip(candidates[:, :, 1], *y_bounds)
    candidates[:, :, 2] = np.clip(candidates[:, :, 2], *z_bounds)
    return candidates, labels


def _predict_numpy(
    model,
    xr_t: torch.Tensor,
    ids_t: torch.Tensor,
    source: np.ndarray,
    row_batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    source_t = torch.as_tensor(source, dtype=torch.float32, device=device)
    parts = []
    with torch.no_grad():
        for start in range(0, len(xr_t), row_batch_size):
            stop = min(start + row_batch_size, len(xr_t))
            parts.append(model(xr_t[start:stop], source_t[ids_t[start:stop]]).cpu())
    return torch.cat(parts, dim=0).numpy()


def profile_student_t_score(
    *,
    prediction: np.ndarray,
    tp: np.ndarray,
    ts: np.ndarray,
    event_slices: list[slice],
    nu: float = 4.0,
    sigma_p: float = 0.5,
    sigma_s: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-event profiled robust loss and median origin time."""
    scores = np.empty(len(event_slices), dtype=np.float64)
    t0 = np.empty(len(event_slices), dtype=np.float32)
    for event_index, slc in enumerate(event_slices):
        p = tp[slc]
        s = ts[slc]
        pred = prediction[slc]
        mask_p = p != MISSING
        mask_s = s != MISSING
        offsets = []
        if np.any(mask_p):
            offsets.append(p[mask_p] - pred[mask_p, 0])
        if np.any(mask_s):
            offsets.append(s[mask_s] - pred[mask_s, 1])
        combined = np.concatenate(offsets)
        t0_value = float(np.median(combined))
        t0[event_index] = t0_value
        losses = []
        if np.any(mask_p):
            residual = p[mask_p] - pred[mask_p, 0] - t0_value
            losses.append(0.5 * (nu + 1.0) * np.log1p(residual**2 / (nu * sigma_p**2)))
        if np.any(mask_s):
            residual = s[mask_s] - pred[mask_s, 1] - t0_value
            losses.append(0.5 * (nu + 1.0) * np.log1p(residual**2 / (nu * sigma_s**2)))
        scores[event_index] = float(np.mean(np.concatenate(losses)))
    return scores, t0


def _refine_one_start(
    *,
    model,
    xr_t: torch.Tensor,
    tp_t: torch.Tensor,
    ts_t: torch.Tensor,
    ids_t: torch.Tensor,
    counts_t: torch.Tensor,
    source_init: np.ndarray,
    t0_init: np.ndarray,
    steps: int,
    lr_xyz: float,
    lr_t0: float,
    row_batch_size: int,
    nu: float,
    sigma_p: float,
    sigma_s: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    device = next(model.parameters()).device
    source = torch.nn.Parameter(
        torch.as_tensor(source_init, dtype=torch.float32, device=device).clone()
    )
    t0 = torch.nn.Parameter(
        torch.as_tensor(t0_init, dtype=torch.float32, device=device).clone()
    )
    optimizer = torch.optim.Adam(
        [{"params": [source], "lr": lr_xyz}, {"params": [t0], "lr": lr_t0}]
    )
    mask_p = tp_t != MISSING
    mask_s = ts_t != MISSING
    started = time.perf_counter()
    n_events = len(source_init)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(xr_t), row_batch_size):
            stop = min(start + row_batch_size, len(xr_t))
            ids_batch = ids_t[start:stop]
            pred = model(xr_t[start:stop], source[ids_batch])
            loss_values = torch.zeros(stop - start, dtype=torch.float32, device=device)
            p_mask = mask_p[start:stop]
            s_mask = mask_s[start:stop]
            if bool(p_mask.any()):
                residual = tp_t[start:stop][p_mask] - t0[ids_batch[p_mask]] - pred[p_mask, 0]
                loss_values[p_mask] += 0.5 * (nu + 1.0) * torch.log1p(
                    residual.square() / (nu * sigma_p**2)
                )
            if bool(s_mask.any()):
                residual = ts_t[start:stop][s_mask] - t0[ids_batch[s_mask]] - pred[s_mask, 1]
                loss_values[s_mask] += 0.5 * (nu + 1.0) * torch.log1p(
                    residual.square() / (nu * sigma_s**2)
                )
            weighted = loss_values / counts_t[ids_batch]
            (weighted.sum() / n_events).backward()
        optimizer.step()
        with torch.no_grad():
            source[:, 0].clamp_(*x_bounds)
            source[:, 1].clamp_(*y_bounds)
            source[:, 2].clamp_(*z_bounds)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return source.detach().cpu().numpy(), t0.detach().cpu().numpy(), elapsed


def run_multistart_initializer(
    *,
    model,
    xr: np.ndarray,
    tp: np.ndarray,
    ts: np.ndarray,
    event_ids: np.ndarray,
    n_events: int,
    seed: int = 93017,
    pool_size: int = 16,
    candidate_strategy: str = "annulus",
    refine_top_k: int = 4,
    return_ranks: int = 3,
    steps: int = 100,
    lr_xyz: float = 2.0,
    lr_t0: float = 0.25,
    row_batch_size: int = 20000,
    nu: float = 4.0,
    sigma_p: float = 0.5,
    sigma_s: float = 0.75,
    x_bounds: tuple[float, float] = (-685.0, 685.0),
    y_bounds: tuple[float, float] = (-815.0, 850.0),
    z_bounds: tuple[float, float] = (0.0, 70.0),
    raw_candidates: np.ndarray | None = None,
    candidate_labels: list[str] | None = None,
) -> dict:
    """Run robust continuous multi-start refinement and rank basins per event."""
    if not 1 <= return_ranks <= refine_top_k <= pool_size:
        raise ValueError("require 1 <= return_ranks <= refine_top_k <= pool_size")
    if steps < 0 or row_batch_size < 1:
        raise ValueError("steps must be nonnegative and row_batch_size must be positive")
    device = next(model.parameters()).device
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    overall_started = time.perf_counter()
    if (raw_candidates is None) != (candidate_labels is None):
        raise ValueError(
            "raw_candidates and candidate_labels must be supplied together"
        )
    if raw_candidates is None:
        candidates, labels = build_blind_candidate_pool(
            xr=xr,
            tp=tp,
            ts=ts,
            event_ids=event_ids,
            n_events=n_events,
            pool_size=pool_size,
            seed=seed,
            strategy=candidate_strategy,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
        )
    else:
        candidates = np.asarray(raw_candidates, dtype=np.float32)
        labels = list(candidate_labels)
        expected_shape = (pool_size, n_events, 3)
        if candidates.shape != expected_shape:
            raise ValueError(
                f"raw_candidates must have shape {expected_shape}, got {candidates.shape}"
            )
        if len(labels) != pool_size or len(set(labels)) != pool_size:
            raise ValueError("candidate_labels must be unique and match pool_size")
        if not np.all(np.isfinite(candidates)):
            raise ValueError("raw_candidates contain non-finite values")
        if (
            np.any((candidates[..., 0] < x_bounds[0]) | (candidates[..., 0] > x_bounds[1]))
            or np.any((candidates[..., 1] < y_bounds[0]) | (candidates[..., 1] > y_bounds[1]))
            or np.any((candidates[..., 2] < z_bounds[0]) | (candidates[..., 2] > z_bounds[1]))
        ):
            raise ValueError("raw_candidates lie outside the declared bounds")
    slices = _event_slices(event_ids, n_events)
    xr_t = torch.as_tensor(xr, dtype=torch.float32, device=device)
    tp_t = torch.as_tensor(tp, dtype=torch.float32, device=device)
    ts_t = torch.as_tensor(ts, dtype=torch.float32, device=device)
    ids_t = torch.as_tensor(event_ids, dtype=torch.long, device=device)
    phase_counts = np.bincount(
        event_ids,
        weights=((tp != MISSING).astype(np.float32) + (ts != MISSING).astype(np.float32)),
        minlength=n_events,
    ).astype(np.float32)
    counts_t = torch.as_tensor(phase_counts, dtype=torch.float32, device=device)

    raw_scores = np.empty((pool_size, n_events), dtype=np.float64)
    raw_t0 = np.empty((pool_size, n_events), dtype=np.float32)
    scoring_started = time.perf_counter()
    for candidate_index in range(pool_size):
        prediction = _predict_numpy(
            model, xr_t, ids_t, candidates[candidate_index], row_batch_size
        )
        raw_scores[candidate_index], raw_t0[candidate_index] = profile_student_t_score(
            prediction=prediction,
            tp=tp,
            ts=ts,
            event_slices=slices,
            nu=nu,
            sigma_p=sigma_p,
            sigma_s=sigma_s,
        )
    raw_scoring_seconds = time.perf_counter() - scoring_started
    raw_order = np.argsort(raw_scores, axis=0)[:refine_top_k]
    event_index = np.arange(n_events)

    refined_sources = np.empty((refine_top_k, n_events, 3), dtype=np.float32)
    refined_t0 = np.empty((refine_top_k, n_events), dtype=np.float32)
    refined_scores = np.empty((refine_top_k, n_events), dtype=np.float64)
    refine_seconds = []
    for rank in range(refine_top_k):
        source_init = candidates[raw_order[rank], event_index]
        t0_init = raw_t0[raw_order[rank], event_index]
        source_value, _, elapsed = _refine_one_start(
            model=model,
            xr_t=xr_t,
            tp_t=tp_t,
            ts_t=ts_t,
            ids_t=ids_t,
            counts_t=counts_t,
            source_init=source_init,
            t0_init=t0_init,
            steps=steps,
            lr_xyz=lr_xyz,
            lr_t0=lr_t0,
            row_batch_size=row_batch_size,
            nu=nu,
            sigma_p=sigma_p,
            sigma_s=sigma_s,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
        )
        prediction = _predict_numpy(model, xr_t, ids_t, source_value, row_batch_size)
        score_value, t0_value = profile_student_t_score(
            prediction=prediction,
            tp=tp,
            ts=ts,
            event_slices=slices,
            nu=nu,
            sigma_p=sigma_p,
            sigma_s=sigma_s,
        )
        refined_sources[rank] = source_value
        refined_t0[rank] = t0_value
        refined_scores[rank] = score_value
        refine_seconds.append(elapsed)

    final_order = np.argsort(refined_scores, axis=0)[:return_ranks]
    ranked_sources = refined_sources[final_order, event_index]
    ranked_t0 = refined_t0[final_order, event_index]
    ranked_scores = refined_scores[final_order, event_index]
    ranked_raw_candidate_index = raw_order[final_order, event_index]
    return {
        "ranked_sources": ranked_sources.astype(np.float32),
        "ranked_t0": ranked_t0.astype(np.float32),
        "ranked_scores": ranked_scores,
        "ranked_raw_candidate_index": ranked_raw_candidate_index.astype(np.int64),
        "candidate_labels": labels,
        "raw_scores": raw_scores,
        "raw_best_score": raw_scores.min(axis=0),
        "refined_best_score": ranked_scores[0],
        "runtime": {
            "raw_scoring_seconds": raw_scoring_seconds,
            "refine_seconds_by_rank": refine_seconds,
            "total_seconds": time.perf_counter() - overall_started,
        },
        "config": {
            "seed": seed,
            "pool_size": pool_size,
            "candidate_strategy": candidate_strategy,
            "refine_top_k": refine_top_k,
            "return_ranks": return_ranks,
            "steps": steps,
            "lr_xyz": lr_xyz,
            "lr_t0": lr_t0,
            "row_batch_size": row_batch_size,
            "nu": nu,
            "sigma_p": sigma_p,
            "sigma_s": sigma_s,
            "x_bounds": list(x_bounds),
            "y_bounds": list(y_bounds),
            "z_bounds": list(z_bounds),
            "uses_truth": False,
            "uses_spatial_grid": False,
        },
    }
