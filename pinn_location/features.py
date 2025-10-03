"""Feature engineering helpers for seismic travel-time prediction."""
from __future__ import annotations

from typing import Tuple

import torch


def build_feats(x: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Construct conditional features for the (station, source) pair.

    Parameters
    ----------
    x:
        Tensor of shape ``[N, 3]`` containing station coordinates.
    xs:
        Tensor of shape ``[3]`` or ``[N, 3]`` containing earthquake hypocentres.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[N, 13]`` containing the concatenated features
        ``[x, xs, r, u, d]`` where ``r`` is the offset vector, ``u`` is the
        normalised ray direction and ``d`` is the epicentral distance.
    """
    if xs.dim() == 1:
        xs = xs.unsqueeze(0).expand_as(x)
    elif xs.shape[0] != x.shape[0]:
        raise ValueError("Station and source tensors must have matching length")

    r = x - xs
    d = torch.linalg.norm(r, dim=-1, keepdim=True).clamp_min(1e-6)
    u = r / d
    return torch.cat([x, xs, r, u, d], dim=-1)


__all__ = ["build_feats"]
