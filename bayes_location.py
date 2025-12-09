import math
import time
import datetime
import pickle
import random
import re
from typing import Optional, Dict

import numpy as np
import torch
from torch import Tensor

from models.mlp import PINNTravelTime

SEED = 2024
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.bfloat16
MISSING_VAL = -12345.0

model = PINNTravelTime().to(device).eval()
ckpt_path = "ckpt/travel_time.ps.v2.pth"
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.to(dtype)

forward_count = 0


@torch.no_grad()
def forward_tps(xs: Tensor, xr: Tensor) -> Tensor:
    global forward_count
    xs_f = xs.to(device=device, dtype=dtype)
    xr_f = xr.to(device=device, dtype=dtype)
    forward_count += 1
    return model(xr_f, xs_f)


def build_phase_masks(
    Tp_obs: Tensor, Ts_obs: Tensor, missing_value: float = MISSING_VAL
):
    maskP = Tp_obs != missing_value
    maskS = Ts_obs != missing_value
    Np = int(maskP.sum().item())
    Ns = int(maskS.sum().item())
    return maskP, maskS, Np, Ns


def sample_invgamma_batch(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    assert alpha.device == beta.device
    gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=beta)
    g = gamma_dist.sample()
    invg = 1.0 / g
    return invg


def gmm_logpdf(x: Tensor, gmm: Dict[str, Tensor]) -> Tensor:
    w, mu, L, log_det = gmm["w"], gmm["mu"], gmm["L"], gmm["log_det"]
    D = mu.shape[-1]
    x2d = x.view(-1, D)
    Ntot = x2d.shape[0]
    M = w.shape[0]
    logs = []
    for k in range(M):
        y = x2d - mu[k]
        z = torch.triangular_solve(y.T, L[k], upper=False)[0].T
        quad = torch.sum(z * z, dim=1)
        lg = (
            -0.5
            * (quad + log_det[k] + D * math.log(2.0 * math.pi))
            + torch.log(w[k] + 1e-12)
        )
        logs.append(lg)
    logs = torch.stack(logs, dim=1)
    m, _ = torch.max(logs, dim=1, keepdim=True)
    res = m.squeeze(1) + torch.log(
        torch.sum(torch.exp(logs - m), dim=1) + 1e-12
    )
    return res.view(x.shape[:-1])


def make_log_prior_xs(
    *,
    gmm_prior: Optional[Dict[str, Tensor]] = None,
    wide_sigma: float = 1000.0,
    prior_mix: Optional[float] = None,
    device: torch.device,
    dtype: torch.dtype,
):
    wide_sigma2 = torch.as_tensor(wide_sigma**2, device=device, dtype=dtype)
    log_norm_wide = -0.5 * (
        3 * math.log(2.0 * math.pi) + 3 * torch.log(wide_sigma2)
    )

    def log_wide(x: Tensor) -> Tensor:
        quad = torch.sum(x * x, dim=-1) / wide_sigma2
        return log_norm_wide - 0.5 * quad

    if gmm_prior is None and prior_mix is None:
        def _f(x: Tensor) -> Tensor:
            return log_wide(x)
        return _f

    if gmm_prior is not None and (prior_mix is None):
        def _f(x: Tensor) -> Tensor:
            return gmm_logpdf(x, gmm_prior)
        return _f

    if gmm_prior is None and (prior_mix is not None):
        def _f(x: Tensor) -> Tensor:
            return log_wide(x)
        return _f

    a = float(prior_mix)
    loga = math.log(max(a, 1e-12))
    log1a = math.log(max(1.0 - a, 1e-12))

    def _f(x: Tensor) -> Tensor:
        lg1 = gmm_logpdf(x, gmm_prior) + loga
        lg2 = log_wide(x) + log1a
        m = torch.maximum(lg1, lg2)
        return m + torch.log(torch.exp(lg1 - m) + torch.exp(lg2 - m))

    return _f


@torch.no_grad()
def forward_tps_multi(
    xs_batch: torch.Tensor, xr: torch.Tensor, event_ids: torch.Tensor
) -> torch.Tensor:
    xs_per_obs = xs_batch[event_ids]
    tps = forward_tps(xs_per_obs, xr)
    return tps


def _scatter_counts(mask: torch.Tensor, event_ids: torch.Tensor, NC: int) -> torch.Tensor:
    ones = mask.to(torch.float32)
    return torch.zeros(NC, device=mask.device, dtype=torch.float32).scatter_add_(
        0, event_ids, ones
    )


def _scatter_sum(
    vals: torch.Tensor, mask: torch.Tensor, event_ids: torch.Tensor, NC: int
) -> torch.Tensor:
    v = torch.where(
        mask,
        vals.to(torch.float32),
        torch.tensor(0.0, device=vals.device, dtype=torch.float32),
    )
    return torch.zeros(NC, device=vals.device, dtype=torch.float32).scatter_add_(
        0, event_ids, v
    )


def _scatter_sum_sq(
    vals: torch.Tensor, mask: torch.Tensor, event_ids: torch.Tensor, NC: int
) -> torch.Tensor:
    v = torch.where(
        mask,
        (vals.to(torch.float32) ** 2),
        torch.tensor(0.0, device=vals.device, dtype=torch.float32),
    )
    return torch.zeros(NC, device=vals.device, dtype=torch.float32).scatter_add_(
        0, event_ids, v
    )


def log_lik_per_event(
    T_pred: torch.Tensor,
    Tp_obs: torch.Tensor,
    Ts_obs: torch.Tensor,
    t0: torch.Tensor,
    sigmaP2: torch.Tensor,
    sigmaS2: torch.Tensor,
    log2pi: torch.Tensor,
    maskP: torch.Tensor,
    maskS: torch.Tensor,
    event_ids: torch.Tensor,
    NC: int,
    lambdaP: Optional[torch.Tensor] = None,
    lambdaS: Optional[torch.Tensor] = None,
):
    tp_model = T_pred[:, 0]
    ts_model = T_pred[:, 1]

    if lambdaP is None:
        lambdaP_eff = torch.ones_like(Tp_obs, dtype=torch.float32)
    else:
        lambdaP_eff = lambdaP.to(torch.float32)

    if lambdaS is None:
        lambdaS_eff = torch.ones_like(Ts_obs, dtype=torch.float32)
    else:
        lambdaS_eff = lambdaS.to(torch.float32)

    resP = Tp_obs - tp_model - t0[event_ids]
    resP_w2 = lambdaP_eff * (resP.to(torch.float32) ** 2)
    rssP_e = _scatter_sum(resP_w2, maskP, event_ids, NC)
    Np_e = _scatter_counts(maskP, event_ids, NC)

    resS = Ts_obs - ts_model - t0[event_ids]
    resS_w2 = lambdaS_eff * (resS.to(torch.float32) ** 2)
    rssS_e = _scatter_sum(resS_w2, maskS, event_ids, NC)
    Ns_e = _scatter_counts(maskS, event_ids, NC)

    sumP_e = _scatter_sum(resP, maskP, event_ids, NC)
    sumS_e = _scatter_sum(resS, maskS, event_ids, NC)

    llP_e = torch.zeros(NC, device=Tp_obs.device, dtype=torch.float32)
    llS_e = torch.zeros(NC, device=Tp_obs.device, dtype=torch.float32)

    validP = Np_e > 0
    if validP.any():
        llP_e[validP] = (
            -0.5 * (rssP_e[validP] / sigmaP2[validP])
            - 0.5 * Np_e[validP] * (log2pi + torch.log(sigmaP2[validP]))
        )

    validS = Ns_e > 0
    if validS.any():
        llS_e[validS] = (
            -0.5 * (rssS_e[validS] / sigmaS2[validS])
            - 0.5 * Ns_e[validS] * (log2pi + torch.log(sigmaS2[validS]))
        )

    ll_e = llP_e + llS_e
    return ll_e, Np_e, Ns_e, sumP_e, sumS_e, rssP_e, rssS_e


def update_lambda_student_t(
    Tp_obs: torch.Tensor,
    Ts_obs: torch.Tensor,
    T_pred_curr: torch.Tensor,
    t0: torch.Tensor,
    sigmaP2: torch.Tensor,
    sigmaS2: torch.Tensor,
    event_ids: torch.Tensor,
    maskP: torch.Tensor,
    maskS: torch.Tensor,
    lambdaP: torch.Tensor,
    lambdaS: torch.Tensor,
    nuP: float,
    nuS: float,
):
    device = Tp_obs.device
    dtype = Tp_obs.dtype

    tp_model = T_pred_curr[:, 0]
    ts_model = T_pred_curr[:, 1]

    sigmaP2_obs = sigmaP2[event_ids]
    sigmaS2_obs = sigmaS2[event_ids]

    idxP = maskP
    if idxP.any():
        resP = Tp_obs[idxP] - t0[event_ids[idxP]] - tp_model[idxP]
        r2P = resP * resP
        shapeP = 0.5 * (nuP + 1.0)
        rateP = 0.5 * (nuP + r2P / sigmaP2_obs[idxP])
        gammaP = torch.distributions.Gamma(shapeP, rateP)
        lambdaP_new = gammaP.sample()
        lambdaP[idxP] = lambdaP_new.to(dtype=dtype, device=device)

    idxS = maskS
    if idxS.any():
        resS = Ts_obs[idxS] - t0[event_ids[idxS]] - ts_model[idxS]
        r2S = resS * resS
        shapeS = 0.5 * (nuS + 1.0)
        rateS = 0.5 * (nuS + r2S / sigmaS2_obs[idxS])
        gammaS = torch.distributions.Gamma(shapeS, rateS)
        lambdaS_new = gammaS.sample()
        lambdaS[idxS] = lambdaS_new.to(dtype=dtype, device=device)


@torch.no_grad()
def gibbs_mh_location_multi(
    Tp_obs: torch.Tensor,
    Ts_obs: torch.Tensor,
    xr: torch.Tensor,
    event_ids: torch.Tensor,
    NC: int,
    n_samples: int = 4000,
    burn: int = 1000,
    thin: int = 1,
    xs_init: Optional[torch.Tensor] = None,
    t0_init: float = 0.0,
    alpha0: float = 1e-2,
    beta0: float = 1e-2,
    sigmaP_init: float = 0.1,
    sigmaS_init: float = 0.1,
    prop_scale: float = 0.5,
    *,
    device_: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
    generator: Optional[torch.Generator] = None,
    gmm_prior: Optional[Dict[str, torch.Tensor]] = None,
    prior_mix: Optional[float] = None,
    wide_sigma: float = 1000.0,
    adapt_steps: int = 1000,
    target_accept_rw: float = 0.3,
    adapt_eta: float = 0.05,
    use_student_t: bool = True,
):
    device_eff = device_ if device_ is not None else device
    if generator is None:
        generator = torch.Generator(device=device_eff)

    Tp_obs = torch.as_tensor(Tp_obs, dtype=dtype, device=device_eff).view(-1)
    Ts_obs = torch.as_tensor(Ts_obs, dtype=dtype, device=device_eff).view(-1)
    xr = torch.as_tensor(xr, dtype=dtype, device=device_eff)
    event_ids = torch.as_tensor(
        event_ids, dtype=torch.long, device=device_eff
    ).view(-1)

    N = Tp_obs.numel()
    assert Ts_obs.shape == (N,) and xr.shape == (N, 3)
    assert event_ids.shape == (N,) and int(event_ids.max().item()) < NC

    maskP, maskS, _, _ = build_phase_masks(Tp_obs, Ts_obs, MISSING_VAL)

    nuP = 4
    nuS = 4

    lambdaP = torch.ones_like(Tp_obs, dtype=dtype, device=device_eff)
    lambdaS = torch.ones_like(Ts_obs, dtype=dtype, device=device_eff)

    if xs_init is None:
        xs = xr.mean(dim=0, keepdim=True).expand(NC, 3).contiguous()
        xs = xs + torch.tensor(
            [0.1, 0.1, 10.0], device=device_eff, dtype=dtype
        ).view(1, 3)
    else:
        xs = torch.as_tensor(xs_init, dtype=dtype, device=device_eff).view(
            NC, 3
        )

    t0 = torch.full((NC,), float(t0_init), dtype=dtype, device=device_eff)
    sigmaP2 = torch.full(
        (NC,), float(sigmaP_init**2), dtype=dtype, device=device_eff
    )
    sigmaS2 = torch.full(
        (NC,), float(sigmaS_init**2), dtype=dtype, device=device_eff
    )

    M = (n_samples - burn) // max(1, thin)
    xs_samples = torch.empty((M, NC, 3), dtype=dtype, device=device_eff)
    t0_samples = torch.empty((M, NC), dtype=dtype, device=device_eff)
    sigmaP_samples = torch.empty((M, NC), dtype=dtype, device=device_eff)
    sigmaS_samples = torch.empty((M, NC), dtype=dtype, device=device_eff)

    log2pi = torch.tensor(
        math.log(2.0 * math.pi), dtype=dtype, device=device_eff
    )

    log_prior_xs = make_log_prior_xs(
        gmm_prior=gmm_prior,
        wide_sigma=wide_sigma,
        prior_mix=prior_mix,
        device=device_eff,
        dtype=dtype,
    )

    T_pred_curr = forward_tps_multi(xs, xr, event_ids)

    ll_e, Np_e, Ns_e, sumP_e, sumS_e, rssP_e, rssS_e = log_lik_per_event(
        T_pred_curr,
        Tp_obs,
        Ts_obs,
        t0,
        sigmaP2,
        sigmaS2,
        log2pi,
        maskP,
        maskS,
        event_ids,
        NC,
        lambdaP=lambdaP if use_student_t else None,
        lambdaS=lambdaS if use_student_t else None,
    )
    curr_logpost_e = log_prior_xs(xs) + ll_e

    log_prop_scale = torch.full(
        (NC,), math.log(max(prop_scale, 1e-6)), dtype=dtype, device=device_eff
    )

    accept_count_e = torch.zeros(NC, dtype=torch.int32, device=device_eff)
    total_prop = 0
    sample_idx = 0

    if verbose:
        print("Start sampling (RW + Student-t)...")

    for k in range(n_samples):
        if verbose and (k % 50 == 0):
            print(f"[iter {k}]")

        if use_student_t:
            update_lambda_student_t(
                Tp_obs,
                Ts_obs,
                T_pred_curr,
                t0,
                sigmaP2,
                sigmaS2,
                event_ids,
                maskP,
                maskS,
                lambdaP,
                lambdaS,
                nuP,
                nuS,
            )

        total_prop += 1

        steps = (
            torch.randn((NC, 3), dtype=dtype, device=device_eff, generator=generator)
            * log_prop_scale.exp().unsqueeze(1)
        )
        xs_prop = xs + steps

        T_pred_prop = forward_tps_multi(xs_prop, xr, event_ids)

        ll_prop_e, _, _, _, _, _, _ = log_lik_per_event(
            T_pred_prop,
            Tp_obs,
            Ts_obs,
            t0,
            sigmaP2,
            sigmaS2,
            log2pi,
            maskP,
            maskS,
            event_ids,
            NC,
            lambdaP=lambdaP if use_student_t else None,
            lambdaS=lambdaS if use_student_t else None,
        )
        prop_logpost_e = log_prior_xs(xs_prop) + ll_prop_e

        log_alpha_e = prop_logpost_e - curr_logpost_e
        u = torch.rand(
            (NC,), dtype=dtype, device=device_eff, generator=generator
        ).log()
        accept_e = u < log_alpha_e

        if accept_e.any():
            xs[accept_e] = xs_prop[accept_e]
            curr_logpost_e[accept_e] = prop_logpost_e[accept_e]
            accept_count_e[accept_e] += 1

            accept_obs = accept_e[event_ids]
            T_pred_curr[accept_obs] = T_pred_prop[accept_obs]

        if k < adapt_steps:
            acc_val = torch.clamp(log_alpha_e.exp(), max=1.0)
            log_prop_scale += adapt_eta * (acc_val - target_accept_rw)

        tp_model = T_pred_curr[:, 0]
        ts_model = T_pred_curr[:, 1]

        sumP_nominal = _scatter_sum(
            lambdaP * (Tp_obs - tp_model), maskP, event_ids, NC
        )
        sumS_nominal = _scatter_sum(
            lambdaS * (Ts_obs - ts_model), maskS, event_ids, NC
        )

        Np_eff = _scatter_sum(lambdaP, maskP, event_ids, NC)
        Ns_eff = _scatter_sum(lambdaS, maskS, event_ids, NC)

        denom = torch.zeros(NC, dtype=dtype, device=device_eff)
        num = torch.zeros(NC, dtype=dtype, device=device_eff)

        validP = Np_eff > 0
        if validP.any():
            denom[validP] += Np_eff[validP] / sigmaP2[validP]
            num[validP] += sumP_nominal[validP] / sigmaP2[validP]

        validS = Ns_eff > 0
        if validS.any():
            denom[validS] += Ns_eff[validS] / sigmaS2[validS]
            num[validS] += sumS_nominal[validS] / sigmaS2[validS]

        upd_mask = denom > 0
        if upd_mask.any():
            var_t0 = torch.zeros_like(denom)
            mean_t0 = torch.zeros_like(denom)
            var_t0[upd_mask] = 1.0 / denom[upd_mask]
            mean_t0[upd_mask] = var_t0[upd_mask] * num[upd_mask]
            noise = torch.zeros_like(mean_t0)
            noise[upd_mask] = torch.randn(
                (int(upd_mask.sum().item()),),
                dtype=dtype,
                device=device_eff,
                generator=generator,
            )
            t0[upd_mask] = (
                mean_t0[upd_mask] + noise[upd_mask] * var_t0[upd_mask].sqrt()
            )

        resP = Tp_obs - t0[event_ids] - tp_model
        resS = Ts_obs - t0[event_ids] - ts_model

        if use_student_t:
            resP_w2 = lambdaP * (resP ** 2)
            resS_w2 = lambdaS * (resS ** 2)
            rssP_e = _scatter_sum(resP_w2, maskP, event_ids, NC)
            rssS_e = _scatter_sum(resS_w2, maskS, event_ids, NC)
            Np_eff = _scatter_sum(lambdaP, maskP, event_ids, NC)
            Ns_eff = _scatter_sum(lambdaS, maskS, event_ids, NC)
        else:
            rssP_e = _scatter_sum_sq(resP, maskP, event_ids, NC)
            rssS_e = _scatter_sum_sq(resS, maskS, event_ids, NC)
            Np_eff = _scatter_sum(maskP.to(dtype), maskP, event_ids, NC)
            Ns_eff = _scatter_sum(maskS.to(dtype), maskS, event_ids, NC)

        alphaP = torch.full((NC,), alpha0, dtype=dtype, device=device_eff)
        betaP = torch.full((NC,), beta0, dtype=dtype, device=device_eff)
        alphaS = torch.full((NC,), alpha0, dtype=dtype, device=device_eff)
        betaS = torch.full((NC,), beta0, dtype=dtype, device=device_eff)

        validP = Np_eff > 0
        validS = Ns_eff > 0
        if validP.any():
            alphaP[validP] += 0.5 * Np_eff[validP].to(dtype)
            betaP[validP] += 0.5 * rssP_e[validP].to(dtype)
        if validS.any():
            alphaS[validS] += 0.5 * Ns_eff[validS].to(dtype)
            betaS[validS] += 0.5 * rssS_e[validS].to(dtype)

        sigmaP2 = sample_invgamma_batch(alphaP, betaP)
        sigmaS2 = sample_invgamma_batch(alphaS, betaS)

        ll_e, Np_e, Ns_e, sumP_e, sumS_e, rssP_e, rssS_e = log_lik_per_event(
            T_pred_curr,
            Tp_obs,
            Ts_obs,
            t0,
            sigmaP2,
            sigmaS2,
            log2pi,
            maskP,
            maskS,
            event_ids,
            NC,
            lambdaP=lambdaP if use_student_t else None,
            lambdaS=lambdaS if use_student_t else None,
        )
        curr_logpost_e = log_prior_xs(xs) + ll_e

        if k >= burn and ((k - burn) % thin == 0):
            xs_samples[sample_idx] = xs
            t0_samples[sample_idx] = t0
            sigmaP_samples[sample_idx] = sigmaP2.sqrt()
            sigmaS_samples[sample_idx] = sigmaS2.sqrt()
            sample_idx += 1

    accept_rate_e = accept_count_e.to(torch.float32) / max(1, total_prop)

    return {
        "xs_samples": xs_samples,
        "t0_samples": t0_samples,
        "sigmaP_samples": sigmaP_samples,
        "sigmaS_samples": sigmaS_samples,
        "accept_rate_per_event": accept_rate_e,
        "final_prop_scale_per_event": log_prop_scale.exp(),
    }


def starts_with_int(s: str) -> bool:
    pattern = r"^[+-]?\d+(?!\d|\.)"
    return re.match(pattern, s) is not None


def read_station_file(file_path="ayrdata/china.loc"):
    stations = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            network = parts[0]
            station_code = parts[1]
            lon = float(parts[3])
            lat = float(parts[4])
            ele = float(parts[5])
            stations[f"{network}{station_code}"] = np.array(
                [lon, lat, -ele / 1000]
            )
    return stations


def read_event_file(file_path):
    with open("data/vel.py.model", "rb") as f:
        grid_vp, grid_vs, grid_x, grid_y, grid_z, proj, *rest = pickle.load(f)

    events = []
    stloc = read_station_file()

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        if starts_with_int(lines[i]):
            header = lines[i][:].strip().split()[1:]
            try:
                origin_time = datetime.datetime.strptime(
                    " ".join(header[0:4]), "%Y %m %d %H %M %S.%f"
                )
            except Exception:
                origin_time = datetime.datetime.strptime(
                    " ".join(header[0:4]).replace("60", "59"),
                    "%Y %m %d %H:%M:%S.%f",
                )
            lat = float(header[6])
            lon = float(header[7])
            depth = float(header[8])
            mag1 = float(header[9])
            mag2 = float(header[10])
            base_delta = float(header[4])

            event_info = {
                "origin_time": origin_time,
                "latitude": lat,
                "longitude": lon,
                "depth": depth,
                "mag1": mag1,
                "mag2": mag2,
                "phases": [],
            }

            i += 1
            while i < len(lines) and not starts_with_int(lines[i]):
                parts = lines[i].split()
                station = "".join(parts[:2])
                travel_time = float(parts[4])
                weight = float(parts[6])
                phase = parts[2]
                event_info["phases"].append(
                    {
                        "station": station,
                        "travel_time": travel_time,
                        "weight": weight,
                        "phase": phase,
                    }
                )
                i += 1

            events.append(event_info)
        else:
            i += 1

    loc_events = []
    all_events = []
    for e in events:
        tstr = e["origin_time"].strftime("%Y%m%d")
        x, y = proj(e["longitude"], e["latitude"])
        z = e["depth"]
        all_events.append([x, y, z])

        stations = []
        for p in e["phases"]:
            stations.append(p["station"])

        set_stations = list(set(stations))
        st2id = {st: i for i, st in enumerate(set_stations)}
        rcv = np.zeros([len(set_stations), 3])
        T_p = np.ones([len(set_stations)]) * MISSING_VAL
        T_s = np.ones([len(set_stations)]) * MISSING_VAL

        for st in set_stations:
            if st not in stloc:
                continue
            loc = stloc[st]
            x, y = proj(loc[0], loc[1])
            z = loc[2]
            if x < -1000 or x > 1000:
                continue
            if y < -1000 or y > 1000:
                continue
            if z < -10 or z > 100:
                continue
            rcv[st2id[st], :] = np.array([x, y, z])

        for p in e["phases"]:
            if p["station"] not in stloc:
                continue
            loc = stloc[p["station"]]
            x, y = proj(loc[0], loc[1])
            z = loc[2]
            if x < -1000 or x > 1000:
                continue
            if y < -1000 or y > 1000:
                continue
            if z < -10 or z > 100:
                continue
            if p["phase"] == "P":
                T_p[st2id[p["station"]]] = p["travel_time"]
            elif p["phase"] == "S":
                T_s[st2id[p["station"]]] = p["travel_time"]

        locinfo = {
            "rcv": torch.tensor(rcv.astype(np.float32), device=device).float(),
            "T_p": torch.tensor(T_p.astype(np.float32), device=device).float(),
            "T_s": torch.tensor(T_s.astype(np.float32), device=device).float(),
        }
        loc_events.append(
            {
                "etime": e["origin_time"],
                "locinfo": locinfo,
                "mag1": e["mag1"],
                "mag2": e["mag2"],
            }
        )

    all_events = np.array(all_events).astype(np.float32)
    all_events = torch.tensor(all_events, device=device).float()
    return loc_events, all_events, proj


import tqdm


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    events, ctlg, proj = read_event_file("data/Real.txt")

    rcvs = []
    c = 0
    event_ids = []
    Tp_obs = []
    Ts_obs = []
    xr = []

    for eve, ctg in tqdm.tqdm(zip(events, ctlg)):
        etime = eve["etime"]
        locinfo = eve["locinfo"]
        rcv = locinfo["rcv"]
        T_p = locinfo["T_p"]
        T_s = locinfo["T_s"]
        event_id = torch.ones([len(rcv)], device=device)
        event_ids.append(event_id * c)
        c += 1
        rcvs.append({"xr": rcv, "Tp": T_p, "Ts": T_s})
        xr.append(rcv)
        Tp_obs.append(T_p)
        Ts_obs.append(T_s)

    event_ids = torch.cat(event_ids)
    Tp_obs = torch.cat(Tp_obs)
    Ts_obs = torch.cat(Ts_obs)
    xr = torch.cat(xr)

    t1 = time.perf_counter()
    alpha0 = 3.0
    beta0 = (0.5**2) * (alpha0 - 1)

    out = gibbs_mh_location_multi(
        Tp_obs,
        Ts_obs,
        xr,
        event_ids=event_ids,
        NC=c,
        n_samples=4000,
        burn=2000,
        thin=2,
        gmm_prior=None,
        prior_mix=None,
        prop_scale=2.0,
        device_=device,
        dtype=dtype,
        verbose=True,
        alpha0=alpha0,
        beta0=beta0,
        use_student_t=False,
    )
    t2 = time.perf_counter()
    print("Gibbs MH 耗时:", t2 - t1)
    print("Gibbs MH 耗时(per sample):", (t2 - t1) / len(rcvs))

    xs_samples = out["xs_samples"]
    t0_samples = out["t0_samples"]
    ofile = open("odata/reloc.raw.v3.2.txt", "w")
    print(len(events), len(ctlg), xs_samples.shape, t0_samples.shape)

    xs_samples = xs_samples.float()
    t0_samples = t0_samples.float()

    for idx, (eve, ctg) in enumerate(zip(events, ctlg)):
        xs_samps, t0_samps = xs_samples[:, idx, :], t0_samples[:, idx]
        xs_mean = xs_samps.mean(dim=0)
        xs_std = xs_samps.std(dim=0, unbiased=True)

        t0_mean = t0_samps.mean()
        etime = eve["etime"]

        t0_mean = t0_mean.flatten().detach().cpu().numpy()
        xs_std = xs_std.flatten().detach().cpu().numpy()

        q = torch.tensor(
            [0.05, 0.95], device=xs_samps.device, dtype=xs_samps.dtype
        )
        err = []
        for i, name in enumerate(["x", "y", "z"]):
            lo, hi = torch.quantile(xs_samps[:, i], q).tolist()
            err.append(hi - lo)

        tstr = (
            etime + datetime.timedelta(seconds=float(t0_mean[0]))
        ).strftime("%Y-%m-%d %H:%M:%S.%f")
        x, y, z = xs_mean.detach().cpu().numpy()
        lon, lat = proj(x, y, inverse=True)
        ofile.write(
            f"#EVENT,{tstr},{lon},{lat},{z},{x},{y},{z},"
            f"{xs_std[0]},{xs_std[1]},{xs_std[2]},{err[0]},{err[1]},{err[2]}\n"
        )
        ctg = ctg.cpu().numpy()
        x, y = proj(ctg[0], ctg[1], inverse=True)
        z = ctg[2]
        ofile.write(
            f"{ctg[0]},{ctg[1]},{ctg[2]},{x},{y},{z},"
            f"{eve['mag1']},{eve['mag2']}\n"
        )
        q = torch.tensor(
            [0.025, 0.975], device=xs_samps.device, dtype=xs_samps.dtype
        )
        for i, name in enumerate(["x", "y", "z"]):
            lo, hi = torch.quantile(xs_samps[:, i], q).tolist()
            ofile.write(f" {name}: [{lo:.3f}, {hi:.3f}], range: {(hi - lo):.3f}\n")
        ofile.flush()
