"""Gibbs-within-Metropolis-Hastings sampler for earthquake locations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor

from .features import build_feats

TraveltimeModel = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]


@dataclass
class SamplerConfig:
    """Configuration controlling the Gibbs-MH sampling procedure."""

    step_std: Tensor
    num_iterations: int = 2_000
    burn_in: int = 500
    thinning: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.step_std = torch.as_tensor(self.step_std, dtype=torch.float32, device=self.device)
        if self.step_std.ndim != 1 or self.step_std.shape[0] != 3:
            raise ValueError("step_std must be broadcastable to a 3-vector [σx, σy, σz]")


@dataclass
class Observations:
    station_xyz: Tensor
    travel_times: Tensor
    sigma_p: float
    sigma_s: float

    def to(self, device: str) -> "Observations":  # pragma: no cover - simple
        return Observations(
            station_xyz=self.station_xyz.to(device),
            travel_times=self.travel_times.to(device),
            sigma_p=self.sigma_p,
            sigma_s=self.sigma_s,
        )


def _log_likelihood(
    model: TraveltimeModel,
    obs: Observations,
    hypo: Tensor,
) -> Tensor:
    try:
        pred_p, pred_s = model(obs.station_xyz, hypo)
    except TypeError:
        # Fall back to modules that expect pre-built features
        feats = build_feats(obs.station_xyz, hypo)
        pred_p, pred_s = model(feats)

    mask = torch.isfinite(obs.travel_times)
    logp = torch.tensor(0.0, device=hypo.device)
    if mask[:, 0].any():
        diff = obs.travel_times[mask[:, 0], 0] - pred_p[mask[:, 0]]
        logp = logp - 0.5 * torch.sum(diff.pow(2) / (obs.sigma_p ** 2))
    if mask[:, 1].any():
        diff = obs.travel_times[mask[:, 1], 1] - pred_s[mask[:, 1]]
        logp = logp - 0.5 * torch.sum(diff.pow(2) / (obs.sigma_s ** 2))
    return logp


def gibbs_mh_location_sampler(
    model: TraveltimeModel,
    observations: Observations,
    initial_hypo: Tensor,
    config: SamplerConfig,
    *,
    progress: bool = False,
) -> Tensor:
    """Draw posterior samples of the earthquake hypocentre."""

    obs = observations.to(config.device)
    x = initial_hypo.to(config.device)
    step_std = config.step_std.to(config.device)

    samples = []
    log_prob = _log_likelihood(model, obs, x)
    total = config.num_iterations

    for it in range(total):
        for dim in range(3):
            proposal = x.clone()
            proposal[dim] = proposal[dim] + torch.randn((), device=config.device) * step_std[dim]
            log_prob_prop = _log_likelihood(model, obs, proposal)
            log_alpha = log_prob_prop - log_prob
            if torch.log(torch.rand((), device=config.device)) < log_alpha:
                x = proposal
                log_prob = log_prob_prop
        if it >= config.burn_in and (it - config.burn_in) % config.thinning == 0:
            samples.append(x.detach().cpu())
        if progress and (it + 1) % max(1, total // 10) == 0:
            print(f"Iter {it+1}/{total}")
    return torch.stack(samples, dim=0)


__all__ = [
    "SamplerConfig",
    "Observations",
    "gibbs_mh_location_sampler",
]
