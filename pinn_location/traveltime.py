"""High-level helpers for generating dense travel-time tables."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import Tensor

from .features import build_feats
from .model import TauPSNet


@torch.no_grad()
def compute_travel_time_table(
    model: TauPSNet,
    stations: Tensor,
    grid_points: Tensor,
    *,
    batch_size: int = 4096,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Tensor, Tensor]:
    """Evaluate P- and S-wave travel times on a 3-D grid."""

    model = model.to(device)
    stations = stations.to(device)
    grid_points = grid_points.to(device)

    tau_p = []
    tau_s = []
    for i in range(0, grid_points.shape[0], batch_size):
        xs = grid_points[i : i + batch_size]
        xs_expand = xs[:, None, :].expand(-1, stations.shape[0], -1)
        stations_expand = stations[None, :, :].expand(xs.shape[0], -1, -1)
        feats = build_feats(stations_expand.reshape(-1, 3), xs_expand.reshape(-1, 3))
        pred_p, pred_s = model(feats)
        pred_p = pred_p.view(xs.shape[0], stations.shape[0])
        pred_s = pred_s.view(xs.shape[0], stations.shape[0])
        tau_p.append(pred_p.cpu())
        tau_s.append(pred_s.cpu())
    return torch.cat(tau_p, dim=0), torch.cat(tau_s, dim=0)


__all__ = ["compute_travel_time_table"]
