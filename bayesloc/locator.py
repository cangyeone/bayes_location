"""Accelerator-aware MH-within-Gibbs locator from the corrected local workflow.

Ported from robust_bayes_location_public/revision_experiments/locator.py.
The probabilistic updates are preserved; see docs/ALGORITHM.md for provenance.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


MISSING = -12345.0
PHYSICAL_BOUNDS_KM = (
    (-685.0, 685.0),
    (-815.0, 850.0),
    (0.0, 70.0),
)


class TravelTimeNet(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(6, hidden_dim), nn.Tanh()]
        for _ in range(6):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.extend([nn.Linear(hidden_dim, 2), nn.Softplus()])
        self.net_merge = nn.Sequential(*layers)

    def forward(self, receiver: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        values = torch.cat([receiver, source], dim=-1) / 1000.0
        return self.net_merge(values) * 10.0


def require_device(name: str) -> torch.device:
    """Return a requested accelerator after an explicit availability check."""
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {name}")


def require_mps() -> torch.device:
    """Backward-compatible helper for the original Apple-MPS experiments."""
    return require_device("mps")


def load_travel_time_model(checkpoint: str | Path, device: torch.device) -> TravelTimeNet:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hidden = int(payload.get("model_hidden_dim", 256))
    model = TravelTimeNet(hidden_dim=hidden)
    model.load_state_dict(payload["model_state"])
    return model.to(device=device, dtype=torch.float32).eval()


def _gamma_sample_shape_rate(shape: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """Draw Gamma(shape, rate), using the CPU only for the MPS workaround."""
    shape_b = shape.expand_as(rate) if shape.ndim == 0 else torch.broadcast_to(shape, rate.shape)
    if rate.device.type == "mps":
        draw = torch.distributions.Gamma(
            shape_b.detach().cpu(), rate.detach().cpu()
        ).sample()
        return draw.to(device=rate.device, dtype=rate.dtype)
    return torch.distributions.Gamma(shape_b, rate).sample()


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _invgamma(shape: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    return 1.0 / _gamma_sample_shape_rate(shape, rate).clamp_min(1.0e-12)


def _beta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ones_a = torch.ones_like(a)
    ones_b = torch.ones_like(b)
    x = _gamma_sample_shape_rate(a, ones_a)
    y = _gamma_sample_shape_rate(b, ones_b)
    return (x / (x + y).clamp_min(1.0e-12)).clamp(1.0e-6, 1.0 - 1.0e-6)


def _log_student_t(residual: torch.Tensor, variance: torch.Tensor, nu: float) -> torch.Tensor:
    nu_t = torch.as_tensor(nu, device=residual.device, dtype=residual.dtype)
    constant = (
        torch.lgamma((nu_t + 1.0) / 2.0)
        - torch.lgamma(nu_t / 2.0)
        - 0.5 * torch.log(nu_t * math.pi)
        - 0.5 * torch.log(variance)
    )
    return constant - 0.5 * (nu_t + 1.0) * torch.log1p(
        residual.square() / (nu_t * variance)
    )


@torch.no_grad()
def run_sampler(
    *,
    model: TravelTimeNet,
    xr: np.ndarray,
    tp: np.ndarray,
    ts: np.ndarray,
    event_ids: np.ndarray,
    n_events: int,
    xs_init: np.ndarray | None = None,
    t0_init: np.ndarray | None = None,
    mode: str,
    seed: int,
    n_samples: int = 800,
    burn: int = 400,
    thin: int = 2,
    nu: float = 4.0,
    alpha0: float = 3.0,
    beta0: float = 0.5,
    sigma_p_init: float = 0.5,
    sigma_s_init: float = 0.5,
    sigma_out_p: float = 15.0,
    sigma_out_s: float = 20.0,
    a_pi: float = 8.0,
    b_pi: float = 2.0,
    prop_scale: float = 2.0,
    adapt_steps: int | None = None,
    target_accept: float = 0.30,
    adapt_eta: float = 0.05,
    physical_bounds: tuple[tuple[float, float], ...] = PHYSICAL_BOUNDS_KM,
    verbose: bool = True,
) -> dict:
    """Run one vectorized catalog chain for Gaussian, Student-t, or Student-t+z."""
    if mode not in {"gaussian", "student_t", "student_t_z"}:
        raise ValueError(f"unknown mode: {mode}")
    if n_samples <= burn or burn < 0 or thin < 1:
        raise ValueError("require n_samples > burn >= 0 and thin >= 1")
    if any(not math.isfinite(v) or v <= 0 for v in
           (nu, alpha0, beta0, sigma_p_init, sigma_s_init, sigma_out_p,
            sigma_out_s, a_pi, b_pi, prop_scale)):
        raise ValueError("distribution scales, shapes, nu and proposal scale must be finite and positive")
    if n_events < 1 or len(event_ids) == 0:
        raise ValueError("at least one event and observation are required")
    if not np.array_equal(np.unique(event_ids), np.arange(n_events)):
        raise ValueError("event_ids must be dense integers in [0, n_events)")
    if np.asarray(xr).shape != (len(event_ids), 3):
        raise ValueError("xr must have shape (N,3)")
    if not all(np.isfinite(v).all() for v in (xr, tp, ts)):
        raise ValueError("observations must be finite; use MISSING for absent phases")
    if np.any((np.asarray(tp) == MISSING) & (np.asarray(ts) == MISSING)):
        raise ValueError("every observation row requires P or S")
    device = next(model.parameters()).device
    if device.type not in {"mps", "cuda", "cpu"}:
        raise RuntimeError(f"unsupported sampler device: {device}")
    torch.manual_seed(seed)
    use_student = mode != "gaussian"
    use_mixture = mode == "student_t_z"
    adapt_steps = burn if adapt_steps is None else min(adapt_steps, burn)
    bounds = np.asarray(physical_bounds, dtype=np.float32)
    if bounds.shape != (3, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("physical_bounds must contain three increasing (lower, upper) pairs")

    xr_t = torch.as_tensor(xr, dtype=torch.float32, device=device)
    tp_t = torch.as_tensor(tp, dtype=torch.float32, device=device)
    ts_t = torch.as_tensor(ts, dtype=torch.float32, device=device)
    ids = torch.as_tensor(event_ids, dtype=torch.long, device=device)
    mask_p = tp_t != MISSING
    mask_s = ts_t != MISSING
    any_mask = mask_p | mask_s
    if len(tp_t) != len(ts_t) or len(tp_t) != len(xr_t) or len(ids) != len(tp_t):
        raise ValueError("observation arrays have inconsistent lengths")
    if int(ids.max().item()) + 1 != n_events:
        raise ValueError("event_ids are not dense in [0, n_events)")

    def scatter_sum(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(n_events, dtype=torch.float32, device=device)
        output.scatter_add_(0, ids, torch.where(mask, value, torch.zeros_like(value)))
        return output

    def scatter_count(mask: torch.Tensor) -> torch.Tensor:
        return scatter_sum(torch.ones_like(tp_t), mask)

    if xs_init is not None:
        source = torch.as_tensor(xs_init, dtype=torch.float32, device=device).reshape(n_events, 3).clone()
    else:
        source = torch.empty((n_events, 3), dtype=torch.float32, device=device)
        global_mean = xr_t.mean(dim=0)
        for event_index in range(n_events):
            selected = (ids == event_index) & any_mask
            source[event_index] = xr_t[selected].mean(dim=0) if bool(selected.any()) else global_mean
        source[:, 2] += 5.0
    if t0_init is not None:
        t0 = torch.as_tensor(
            t0_init, dtype=torch.float32, device=device
        ).reshape(n_events).clone()
    else:
        t0 = torch.zeros(n_events, dtype=torch.float32, device=device)
    physical_lower = torch.as_tensor(bounds[:, 0], dtype=torch.float32, device=device)
    physical_upper = torch.as_tensor(bounds[:, 1], dtype=torch.float32, device=device)

    def in_physical_domain(source_value: torch.Tensor) -> torch.Tensor:
        return ((source_value >= physical_lower) & (source_value <= physical_upper)).all(dim=1)

    initial_in_domain = in_physical_domain(source)
    if not bool(initial_in_domain.all()):
        raise ValueError(
            f"{int((~initial_in_domain).sum().item())} initial sources lie outside physical_bounds"
        )
    sigma_p2 = torch.full((n_events,), sigma_p_init**2, dtype=torch.float32, device=device)
    sigma_s2 = torch.full((n_events,), sigma_s_init**2, dtype=torch.float32, device=device)
    lambda_p = torch.ones_like(tp_t)
    lambda_s = torch.ones_like(ts_t)
    z_p = mask_p.clone()
    z_s = mask_s.clone()
    pi_p = torch.tensor(0.9 if use_mixture else 1.0, dtype=torch.float32, device=device)
    pi_s = torch.tensor(0.9 if use_mixture else 1.0, dtype=torch.float32, device=device)
    out_p2 = torch.tensor(sigma_out_p**2, dtype=torch.float32, device=device)
    out_s2 = torch.tensor(sigma_out_s**2, dtype=torch.float32, device=device)
    log2pi = torch.tensor(math.log(2.0 * math.pi), dtype=torch.float32, device=device)
    log_step = torch.full((n_events,), math.log(prop_scale), dtype=torch.float32, device=device)
    accept = torch.zeros(n_events, dtype=torch.int32, device=device)

    def forward(source_value: torch.Tensor) -> torch.Tensor:
        return model(xr_t, source_value[ids])

    def log_normal_zero(residual: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        return -0.5 * (log2pi + torch.log(variance) + residual.square() / variance)

    def log_good_conditional(
        residual: torch.Tensor, variance: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        if use_student:
            return 0.5 * torch.log(scale) - 0.5 * (
                log2pi + torch.log(variance) + scale * residual.square() / variance
            )
        return log_normal_zero(residual, variance)

    def log_joint(prediction: torch.Tensor, source_value: torch.Tensor) -> torch.Tensor:
        residual_p = tp_t - t0[ids] - prediction[:, 0]
        residual_s = ts_t - t0[ids] - prediction[:, 1]
        var_p = sigma_p2[ids]
        var_s = sigma_s2[ids]
        log_values = torch.zeros_like(tp_t)
        good_p = mask_p & z_p if use_mixture else mask_p
        good_s = mask_s & z_s if use_mixture else mask_s
        bad_p = mask_p & (~z_p) if use_mixture else torch.zeros_like(mask_p)
        bad_s = mask_s & (~z_s) if use_mixture else torch.zeros_like(mask_s)
        if bool(good_p.any()):
            log_values[good_p] += log_good_conditional(
                residual_p[good_p], var_p[good_p], lambda_p[good_p]
            )
            if use_mixture:
                log_values[good_p] += torch.log(pi_p)
        if bool(good_s.any()):
            log_values[good_s] += log_good_conditional(
                residual_s[good_s], var_s[good_s], lambda_s[good_s]
            )
            if use_mixture:
                log_values[good_s] += torch.log(pi_s)
        if bool(bad_p.any()):
            log_values[bad_p] += log_normal_zero(residual_p[bad_p], out_p2) + torch.log(1.0 - pi_p)
        if bool(bad_s.any()):
            log_values[bad_s] += log_normal_zero(residual_s[bad_s], out_s2) + torch.log(1.0 - pi_s)
        likelihood = scatter_sum(log_values, any_mask)
        weak_prior = -0.5 * source_value.square().sum(dim=1) / (1000.0**2)
        return likelihood + weak_prior

    def update_lambdas(prediction: torch.Tensor) -> None:
        if not use_student:
            return
        residual_p = tp_t - t0[ids] - prediction[:, 0]
        residual_s = ts_t - t0[ids] - prediction[:, 1]
        shape = torch.tensor((nu + 1.0) / 2.0, dtype=torch.float32, device=device)
        good_p = mask_p & z_p if use_mixture else mask_p
        good_s = mask_s & z_s if use_mixture else mask_s
        if bool(good_p.any()):
            rate = 0.5 * (nu + residual_p[good_p].square() / sigma_p2[ids[good_p]])
            lambda_p[good_p] = _gamma_sample_shape_rate(shape, rate.clamp_min(1.0e-12))
        if bool(good_s.any()):
            rate = 0.5 * (nu + residual_s[good_s].square() / sigma_s2[ids[good_s]])
            lambda_s[good_s] = _gamma_sample_shape_rate(shape, rate.clamp_min(1.0e-12))

    def update_indicators(prediction: torch.Tensor) -> None:
        if not use_mixture:
            return
        residual_p = tp_t - t0[ids] - prediction[:, 0]
        residual_s = ts_t - t0[ids] - prediction[:, 1]
        if bool(mask_p.any()):
            good = torch.log(pi_p) + _log_student_t(
                residual_p[mask_p], sigma_p2[ids[mask_p]], nu
            )
            bad = torch.log(1.0 - pi_p) + log_normal_zero(residual_p[mask_p], out_p2)
            probability = torch.sigmoid(good - bad)
            z_p[mask_p] = torch.rand_like(probability) < probability
        if bool(mask_s.any()):
            good = torch.log(pi_s) + _log_student_t(
                residual_s[mask_s], sigma_s2[ids[mask_s]], nu
            )
            bad = torch.log(1.0 - pi_s) + log_normal_zero(residual_s[mask_s], out_s2)
            probability = torch.sigmoid(good - bad)
            z_s[mask_s] = torch.rand_like(probability) < probability

    kept = (n_samples - burn + thin - 1) // thin
    xs_samples = torch.empty((kept, n_events, 3), dtype=torch.float32, device=device)
    t0_samples = torch.empty((kept, n_events), dtype=torch.float32, device=device)
    sigma_p_samples = torch.empty((kept, n_events), dtype=torch.float32, device=device)
    sigma_s_samples = torch.empty((kept, n_events), dtype=torch.float32, device=device)
    pi_p_samples = torch.empty(kept, dtype=torch.float32, device=device)
    pi_s_samples = torch.empty(kept, dtype=torch.float32, device=device)
    z_p_sum = torch.zeros_like(tp_t)
    z_s_sum = torch.zeros_like(ts_t)
    sample_index = 0
    prediction = forward(source)
    started = time.perf_counter()

    for iteration in range(n_samples):
        update_indicators(prediction)
        # The z update integrates over the Student-t scale mixture.  Draw lambda
        # afterwards from its conditional for the newly selected inlier set.
        update_lambdas(prediction)
        if use_mixture:
            pi_p = _beta(
                torch.tensor(a_pi, device=device) + (mask_p & z_p).sum(),
                torch.tensor(b_pi, device=device) + (mask_p & ~z_p).sum(),
            )
            pi_s = _beta(
                torch.tensor(a_pi, device=device) + (mask_s & z_s).sum(),
                torch.tensor(b_pi, device=device) + (mask_s & ~z_s).sum(),
            )

        # Latent updates change the target; recompute the current state before MH.
        current_log = log_joint(prediction, source)
        proposed = source + torch.randn_like(source) * log_step.exp().unsqueeze(1)
        in_domain = in_physical_domain(proposed)
        # Never evaluate the neural surrogate outside its validated physical
        # domain.  An out-of-domain random-walk proposal is rejected, which is
        # the symmetric-proposal MH rule for the truncated spatial prior.
        safe_proposed = torch.where(in_domain[:, None], proposed, source)
        proposed_prediction = forward(safe_proposed)
        proposed_log = log_joint(proposed_prediction, safe_proposed)
        proposed_log = torch.where(
            in_domain, proposed_log, torch.full_like(proposed_log, -torch.inf)
        )
        accepted = in_domain & (
            torch.log(torch.rand(n_events, device=device)) < (proposed_log - current_log)
        )
        if bool(accepted.any()):
            source[accepted] = proposed[accepted]
            prediction[accepted[ids]] = proposed_prediction[accepted[ids]]
            accept[accepted] += 1
        if iteration < adapt_steps:
            log_step += adapt_eta * (accepted.float() - target_accept)

        residual_p_no_t0 = tp_t - prediction[:, 0]
        residual_s_no_t0 = ts_t - prediction[:, 1]
        good_p = mask_p & z_p if use_mixture else mask_p
        good_s = mask_s & z_s if use_mixture else mask_s
        bad_p = mask_p & (~z_p) if use_mixture else torch.zeros_like(mask_p)
        bad_s = mask_s & (~z_s) if use_mixture else torch.zeros_like(mask_s)
        weight_p = lambda_p if use_student else torch.ones_like(lambda_p)
        weight_s = lambda_s if use_student else torch.ones_like(lambda_s)
        precision_p = (
            scatter_sum(weight_p, good_p) / sigma_p2
            + scatter_count(bad_p) / out_p2
        )
        precision_s = (
            scatter_sum(weight_s, good_s) / sigma_s2
            + scatter_count(bad_s) / out_s2
        )
        numerator_p = (
            scatter_sum(weight_p * residual_p_no_t0, good_p) / sigma_p2
            + scatter_sum(residual_p_no_t0, bad_p) / out_p2
        )
        numerator_s = (
            scatter_sum(weight_s * residual_s_no_t0, good_s) / sigma_s2
            + scatter_sum(residual_s_no_t0, bad_s) / out_s2
        )
        precision = precision_p + precision_s
        valid = precision > 0
        if bool(valid.any()):
            variance_t0 = 1.0 / precision[valid]
            mean_t0 = variance_t0 * (numerator_p[valid] + numerator_s[valid])
            t0[valid] = mean_t0 + torch.randn_like(mean_t0) * variance_t0.sqrt()

        residual_p = tp_t - t0[ids] - prediction[:, 0]
        residual_s = ts_t - t0[ids] - prediction[:, 1]
        count_p = scatter_count(good_p)
        count_s = scatter_count(good_s)
        rss_p = scatter_sum(weight_p * residual_p.square(), good_p)
        rss_s = scatter_sum(weight_s * residual_s.square(), good_s)
        shape_p = torch.full((n_events,), alpha0, dtype=torch.float32, device=device) + 0.5 * count_p
        shape_s = torch.full((n_events,), alpha0, dtype=torch.float32, device=device) + 0.5 * count_s
        rate_p = torch.full((n_events,), beta0, dtype=torch.float32, device=device) + 0.5 * rss_p
        rate_s = torch.full((n_events,), beta0, dtype=torch.float32, device=device) + 0.5 * rss_s
        sigma_p2 = _invgamma(shape_p, rate_p)
        sigma_s2 = _invgamma(shape_s, rate_s)

        if iteration >= burn and (iteration - burn) % thin == 0:
            xs_samples[sample_index] = source
            t0_samples[sample_index] = t0
            sigma_p_samples[sample_index] = sigma_p2.sqrt()
            sigma_s_samples[sample_index] = sigma_s2.sqrt()
            pi_p_samples[sample_index] = pi_p
            pi_s_samples[sample_index] = pi_s
            z_p_sum += z_p.float()
            z_s_sum += z_s.float()
            sample_index += 1
        if verbose and (iteration + 1) % max(100, n_samples // 4) == 0:
            print(
                f"[{mode}] {iteration + 1}/{n_samples} "
                f"accept={accept.float().mean().item() / (iteration + 1):.3f} "
                f"piP={pi_p.item():.3f} piS={pi_s.item():.3f}",
                flush=True,
            )

    _synchronize(device)
    elapsed = time.perf_counter() - started
    output = {
        "xs_samples": xs_samples.cpu().numpy(),
        "t0_samples": t0_samples.cpu().numpy(),
        "sigma_p_samples": sigma_p_samples.cpu().numpy(),
        "sigma_s_samples": sigma_s_samples.cpu().numpy(),
        "pi_p_samples": pi_p_samples.cpu().numpy(),
        "pi_s_samples": pi_s_samples.cpu().numpy(),
        "inlier_probability_p": (z_p_sum / kept).cpu().numpy(),
        "inlier_probability_s": (z_s_sum / kept).cpu().numpy(),
        "acceptance_rate": (accept.float() / n_samples).cpu().numpy(),
        "final_proposal_scale": log_step.exp().cpu().numpy(),
        "elapsed_seconds": elapsed,
        "mode": mode,
        "seed": seed,
        "device": str(device),
        "n_samples": n_samples,
        "burn": burn,
        "thin": thin,
        "physical_bounds_km": bounds,
        "retained_source_samples_outside_domain": int(
            (~in_physical_domain(xs_samples.reshape(-1, 3))).sum().item()
        ),
    }
    return output


def posterior_event_summary(samples: dict, truth_xyz: np.ndarray) -> list[dict]:
    xs = samples["xs_samples"]
    t0 = samples["t0_samples"]
    mean = xs.mean(axis=0)
    q05 = np.quantile(xs, 0.05, axis=0)
    q95 = np.quantile(xs, 0.95, axis=0)
    error = mean - truth_xyz
    result: list[dict] = []
    for index in range(xs.shape[1]):
        result.append(
            {
                "event_index": index,
                "x_mean_km": float(mean[index, 0]),
                "y_mean_km": float(mean[index, 1]),
                "z_mean_km": float(mean[index, 2]),
                "horizontal_error_km": float(np.hypot(error[index, 0], error[index, 1])),
                "depth_error_km": float(abs(error[index, 2])),
                "time_error_s": float(abs(t0[:, index].mean())),
                "width_x90_km": float(q95[index, 0] - q05[index, 0]),
                "width_y90_km": float(q95[index, 1] - q05[index, 1]),
                "width_z90_km": float(q95[index, 2] - q05[index, 2]),
                "covered_x90": bool(q05[index, 0] <= truth_xyz[index, 0] <= q95[index, 0]),
                "covered_y90": bool(q05[index, 1] <= truth_xyz[index, 1] <= q95[index, 1]),
                "covered_z90": bool(q05[index, 2] <= truth_xyz[index, 2] <= q95[index, 2]),
                "acceptance_rate": float(samples["acceptance_rate"][index]),
            }
        )
    return result
