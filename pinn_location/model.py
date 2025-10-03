"""Neural network for predicting P- and S-wave travel times."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class TauPSNetConfig:
    """Configuration for :class:`TauPSNet`.

    Attributes
    ----------
    in_dim:
        Dimensionality of the input features. Defaults to ``13`` which matches
        :func:`pinn_location.features.build_feats`.
    hidden:
        Width of the hidden layers.
    layers:
        Number of hidden layers.
    use_fourier_features:
        Whether to apply Fourier feature mapping prior to the MLP.
    fourier_features:
        Number of random Fourier features per input dimension if
        ``use_fourier_features`` is enabled.
    fourier_sigma:
        Bandwidth of the Gaussian used to sample Fourier feature frequencies.
    dropout:
        Optional dropout probability applied after each hidden activation.
    """

    in_dim: int = 13
    hidden: int = 256
    layers: int = 6
    use_fourier_features: bool = True
    fourier_features: int = 64
    fourier_sigma: float = 3.0
    dropout: float = 0.0


class FourierFeatures(nn.Module):
    """Random Fourier feature mapping for coordinate inputs."""

    def __init__(self, in_dim: int, features: int, sigma: float = 10.0) -> None:
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, features) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class TauPSNet(nn.Module):
    """Predict joint P- and S-wave travel times for a given event."""

    def __init__(self, config: Optional[TauPSNetConfig] = None) -> None:
        super().__init__()
        config = config or TauPSNetConfig()
        self.config = config

        self.ff = (
            FourierFeatures(config.in_dim, config.fourier_features, config.fourier_sigma)
            if config.use_fourier_features
            else None
        )
        d_in = (
            2 * config.fourier_features
            if self.ff is not None
            else config.in_dim
        )

        layers = []
        for i in range(config.layers):
            layers.append(nn.Linear(d_in if i == 0 else config.hidden, config.hidden))
            layers.append(nn.SiLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
        self.trunk = nn.Sequential(*layers)
        self.head_p = nn.Linear(config.hidden, 1)
        self.head_s = nn.Linear(config.hidden, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in list(self.trunk) + [self.head_p, self.head_s]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ff is not None:
            feats = self.ff(feats)
        h = self.trunk(feats)
        tau_p = self.head_p(h)
        tau_s = self.head_s(h)
        return tau_p.squeeze(-1), tau_s.squeeze(-1)


__all__ = ["TauPSNet", "TauPSNetConfig", "FourierFeatures"]
