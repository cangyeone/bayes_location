"""Masked, purely supervised travel-time regression, grouped by source for validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .io import load_pairs, sha256, write_json
from .locator import TravelTimeNet


def split_sources(xs, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must lie between 0 and 1")
    _, groups = np.unique(xs, axis=0, return_inverse=True)
    n = int(groups.max()) + 1
    if n < 2:
        raise ValueError("validation requires at least two distinct source coordinates")
    order = np.random.default_rng(seed).permutation(n)
    n_valid = min(n - 1, max(1, int(round(n * fraction))))
    valid = np.isin(groups, order[:n_valid])
    return np.flatnonzero(~valid), np.flatnonzero(valid)


def masked_mse(prediction, targets):
    mask = torch.isfinite(targets)
    safe = torch.nan_to_num(targets)
    return ((prediction - safe).square() * mask).sum() / mask.sum().clamp_min(1)


def loader(data, indices, batch_size, shuffle=False):
    labels = np.column_stack([data["tp"], data["ts"]])
    dataset = TensorDataset(*[
        torch.as_tensor(v[indices], dtype=torch.float32)
        for v in (data["xr"], data["xs"], labels)
    ])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def metrics(model, batches, device):
    sums = np.zeros((3, 2), dtype=float)  # absolute, square, count
    model.eval()
    for xr, xs, target in batches:
        pred = model(xr.to(device), xs.to(device)).cpu().numpy()
        target = target.numpy()
        mask = np.isfinite(target)
        residual = np.where(mask, pred - target, 0.0)
        sums[0] += np.abs(residual).sum(axis=0)
        sums[1] += np.square(residual).sum(axis=0)
        sums[2] += mask.sum(axis=0)
    result = {"mse_s2": float(sums[1].sum() / sums[2].sum())}
    for i, phase in enumerate(("p", "s")):
        count = sums[2, i]
        result[f"n_{phase}"] = int(count)
        result[f"mae_{phase}_s"] = float(sums[0, i] / count) if count else None
        result[f"rmse_{phase}_s"] = float(np.sqrt(sums[1, i] / count)) if count else None
    return result


def train(args, device):
    if min(args.epochs, args.batch_size, args.hidden_dim) < 1 or args.lr <= 0:
        raise ValueError("epochs, batch size, hidden dimension and learning rate must be positive")
    torch.manual_seed(args.seed)
    data, meta = load_pairs(args.pairs)
    if args.validation:
        valid_data, valid_meta = load_pairs(args.validation)
        if meta != valid_meta:
            raise ValueError("training and validation geometry must match")
        train_sources = {tuple(x) for x in data["xs"]}
        if any(tuple(x) in train_sources for x in valid_data["xs"]):
            raise ValueError("validation shares source coordinates with training")
        train_ids, valid_ids = np.arange(len(data["xr"])), np.arange(len(valid_data["xr"]))
    else:
        train_ids, valid_ids = split_sources(data["xs"], args.validation_fraction, args.seed)
        valid_data = data
    batches = loader(data, train_ids, args.batch_size, shuffle=True)
    if any(not np.isfinite(data[k][train_ids]).any() for k in ("tp", "ts")):
        raise ValueError("the two-output model requires training labels for both P and S")
    validation = loader(valid_data, valid_ids, args.batch_size)
    model = TravelTimeNet(args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, history = float("inf"), []
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "training_sha256": sha256(args.pairs),
        "validation_sha256": sha256(args.validation) if args.validation else None,
        "seed": args.seed, "train_rows": len(train_ids), "validation_rows": len(valid_ids),
        "split": "external_source_disjoint" if args.validation else "distinct_source_coordinates",
        "optimizer": "AdamW", "lr": args.lr, "weight_decay": 0.01,
        "batch_size": args.batch_size, "selection": "minimum validation masked MSE",
        "loss": "masked P/S MSE in seconds squared; no physics loss",
    }
    np.savez_compressed(str(checkpoint) + ".split.npz", train_rows=train_ids, validation_rows=valid_ids)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for xr, xs, targets in batches:
            xr, xs, targets = xr.to(device), xs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(xr, xs), targets)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            n_labels = int(torch.isfinite(targets).sum())
            total += float(loss.detach()) * n_labels
            count += n_labels
        measured = metrics(model, validation, device)
        record = dict(epoch=epoch, train_mse_s2=total / count, validation=measured)
        history.append(record)
        print(json.dumps(record), flush=True)
        if measured["mse_s2"] < best:
            best = measured["mse_s2"]
            torch.save({
                "schema_version": 1, "model_hidden_dim": args.hidden_dim,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "geometry": meta, "epoch": epoch, "training": provenance,
                "validation": measured,
            }, checkpoint)
        write_json(str(checkpoint) + ".history.json", history)
    print(f"Saved best validation checkpoint: {checkpoint}")
