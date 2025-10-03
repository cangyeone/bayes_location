"""Utilities for training the travel-time neural network."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader

from .features import build_feats
from .model import TauPSNet, TauPSNetConfig


class TraveltimeDataset(Dataset[Tuple[Tensor, Tensor, Tensor]]):
    """Dataset of observed P- and S-wave travel times.

    Each item consists of ``(station_xyz, source_xyz, travel_times)`` where the
    first two tensors have shape ``[3]`` and travel times is ``[2]`` containing
    ``(tau_p, tau_s)`` in seconds. Missing arrivals can be indicated using
    ``NaN`` entries, which are automatically masked during training.
    """

    def __init__(
        self,
        stations: Tensor,
        sources: Tensor,
        travel_times: Tensor,
    ) -> None:
        if stations.shape != sources.shape:
            raise ValueError("Station and source tensors must have identical shapes")
        if travel_times.shape != (*stations.shape[:-1], 2):
            raise ValueError("Travel times must contain both P and S components")
        self.stations = stations
        self.sources = sources
        self.travel_times = travel_times

    def __len__(self) -> int:  # pragma: no cover - simple passthrough
        return self.stations.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self.stations[idx],
            self.sources[idx],
            self.travel_times[idx],
        )


@dataclass
class TrainingConfig:
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-6
    epochs: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: Optional[float] = 10.0


def _loss(pred_p: Tensor, pred_s: Tensor, target: Tensor) -> Tensor:
    mask = torch.isfinite(target)
    if not mask.any():
        return torch.tensor(0.0, device=target.device, requires_grad=True)
    loss = 0.0
    for i, pred in enumerate((pred_p, pred_s)):
        component_mask = mask[:, i]
        if component_mask.any():
            loss = loss + nn.functional.smooth_l1_loss(
                pred[component_mask], target[component_mask, i]
            )
    return loss


def train_taunet(
    model: TauPSNet,
    dataset: Dataset[Tuple[Tensor, Tensor, Tensor]],
    config: Optional[TrainingConfig] = None,
    *,
    num_workers: int = 0,
) -> TauPSNet:
    """Train :class:`TauPSNet` on the provided dataset."""
    config = config or TrainingConfig()
    model = model.to(config.device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)

    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for stations, sources, travel_times in loader:
            stations = stations.to(config.device)
            sources = sources.to(config.device)
            travel_times = travel_times.to(config.device)

            feats = build_feats(stations, sources)
            pred_p, pred_s = model(feats)
            loss = _loss(pred_p, pred_s, travel_times)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optim.step()

            running_loss += loss.detach().item() * stations.shape[0]
        running_loss /= len(dataset)
        # Lightweight progress logging
        if (epoch + 1) % max(1, config.epochs // 10) == 0:
            print(f"Epoch {epoch+1:04d}: loss={running_loss:.4f}")
    return model


__all__ = ["TraveltimeDataset", "TrainingConfig", "train_taunet"]
