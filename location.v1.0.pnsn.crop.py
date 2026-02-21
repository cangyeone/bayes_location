# mcmc_speedup_gmm_mask_nuts.py
# 功能汇总：
# 1) GMM 先验 (由地震目录 ctlg 拟合)，可与宽高斯混合
# 2) RW-MH 自适应步长 (Robbins–Monro 简化)
# 3) 混合核：GMM 独立提议 + RW 随机游走
# 4) NUTS 更新 xs（t0/σ² 仍为条件更新）
# 5) P/S 缺测支持：Tp/Ts 用 -12345 表示缺失，自动掩码

import math
import time
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt


import numpy as np 
import random 
from pyproj import CRS, Transformer
# ================== 设备与随机种子 ==================

SEED   = 2024
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
# ================== 域与超参 ==================
# 建议数据侧把物理坐标无量纲化/归一到 [-1,1]^3

# ---------------------------
# 设备与全局设置
# ---------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.float32 

# 缺测标记
MISSING_VAL = -12345.0

# -------------------------
# Model (as provided)
# -------------------------

# -------------------------
# Model (as provided)
# -------------------------
class PINNTravelTime(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net_merge = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 4),      # <- 2 改 4
            nn.Softplus(),
        )

    def forward(self, x, xs):
        inp = torch.cat([x, xs.expand_as(x)], dim=-1)
        inp = inp / 1000.0
        out = self.net_merge(inp)
        out = out * 10.0
        return out



model = PINNTravelTime().to(device).eval()
ckpt_path = 'ckpt/time.real.pnsn.switch.v1.0.pt'
model.load_state_dict(torch.load(ckpt_path, map_location='cpu')["model_state"])
model.to(dtype)

#ckpt_path = 'ckpt/travel_time.ps.v2.pth'
#model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
#model.to(dtype)


# ===========================
# 批处理：把 rcv(list) 打包成定长 padding
# ===========================
def pack_rcv_to_batch(rcv, *, dtype=torch.float32, device=None):
    """
    rcv: list[dict], 每个包含:
        'xr': [Ni,3], 'Tp':[Ni], 'Ts':[Ni]
    返回:
        XR: [NC, Nmax, 3]
        Tp, Ts: [NC, Nmax]
        exist_mask: [NC, Nmax]  (True=真实台站)
        maskP, maskS: [NC, Nmax]  (相位可用)
        NsP, NsS: [NC]  每事件有效样本数
    """
    assert isinstance(rcv, list) and len(rcv) > 0
    NC = len(rcv)
    Ni_list = [ev['xr'].shape[0] for ev in rcv]
    Nmax = max(Ni_list)

    def pad_2d(mat, fill=0.0):
        Ni = mat.shape[0]
        out = torch.full((Nmax, mat.shape[1]), fill, dtype=dtype, device=device)
        out[:Ni] = torch.as_tensor(mat, dtype=dtype, device=device)
        return out

    def pad_1d(vec, fill):
        Ni = vec.shape[0]
        out = torch.full((Nmax,), fill, dtype=dtype, device=device)
        out[:Ni] = torch.as_tensor(vec, dtype=dtype, device=device)
        return out

    XR = torch.stack([pad_2d(ev['xr'], 0.0) for ev in rcv], dim=0)           # [NC, Nmax, 3]
    Tp = torch.stack([pad_1d(ev['Tp'], MISSING_VAL) for ev in rcv], dim=0)   # [NC, Nmax]
    Ts = torch.stack([pad_1d(ev['Ts'], MISSING_VAL) for ev in rcv], dim=0)   # [NC, Nmax]

    exist_mask = torch.zeros((NC, Nmax), dtype=torch.bool, device=device)
    for i, Ni in enumerate(Ni_list):
        exist_mask[i, :Ni] = True

    maskP = (Tp != MISSING_VAL) & exist_mask
    maskS = (Ts != MISSING_VAL) & exist_mask

    NsP = maskP.sum(dim=1)  # [NC]
    NsS = maskS.sum(dim=1)  # [NC]
    return XR, Tp, Ts, exist_mask, maskP, maskS, NsP, NsS



# ===========================
# 批量对数似然（掩码聚合，逐事件）
# ===========================
@torch.no_grad()
def batched_log_lik(
    T_pred: torch.Tensor, Tp: torch.Tensor, Ts: torch.Tensor,
    t0: torch.Tensor, sigmaP2: torch.Tensor, sigmaS2: torch.Tensor,
    log2pi: torch.Tensor, maskP: torch.Tensor, maskS: torch.Tensor
) -> torch.Tensor:
    """
    T_pred: [NC, Nmax, 2], Tp/Ts: [NC, Nmax], 其余: [NC]
    return: per-event log-lik: [NC]
    """
    tp_model = T_pred[..., 0]
    ts_model = T_pred[..., 1]
    NC = Tp.shape[0]
    ll = Tp.new_zeros(NC)

    # P
    if maskP.any():
        resP = Tp[maskP] - t0.view(-1, 1).expand_as(Tp)[maskP] - tp_model[maskP]
        evt_id_P = torch.nonzero(maskP, as_tuple=False)[:, 0]
        contribP = -0.5 * (resP * resP) / sigmaP2[evt_id_P]
        ll.index_add_(0, evt_id_P, contribP)
        NP_evt = maskP.sum(dim=1)
        ll += -0.5 * (NP_evt * (log2pi + torch.log(sigmaP2)))

    # S
    if maskS.any():
        resS = Ts[maskS] - t0.view(-1, 1).expand_as(Ts)[maskS] - ts_model[maskS]
        evt_id_S = torch.nonzero(maskS, as_tuple=False)[:, 0]
        contribS = -0.5 * (resS * resS) / sigmaS2[evt_id_S]
        ll.index_add_(0, evt_id_S, contribS)
        NS_evt = maskS.sum(dim=1)
        ll += -0.5 * (NS_evt * (log2pi + torch.log(sigmaS2)))

    return ll

def build_phase_masks(Tp_obs: Tensor, Ts_obs: Tensor, missing_value: float = MISSING_VAL):
    """
    返回两种相位是否可用的布尔掩码（形状均为 (N,)）以及各自的样本数。
    """
    maskP = (Tp_obs != missing_value)
    maskS = (Ts_obs != missing_value)
    Np = int(maskP.sum().item())
    Ns = int(maskS.sum().item())
    return maskP, maskS, Np, Ns

def log_lik_from_pred_masked(
    T_pred: Tensor,
    Tp_obs: Tensor, Ts_obs: Tensor,
    t0_local: Tensor,
    sP2: Tensor, sS2: Tensor,
    log2pi: Tensor,
    maskP: Tensor, maskS: Tensor,
    Np: int, Ns: int,
) -> Tensor:
    """
    只在有观测的相位上计算似然。
    - 若某一相位完全缺失（N=0），跳过该相位的项。
    """
    tp_model = T_pred[:, 0]
    ts_model = T_pred[:, 1]
    ll = torch.tensor(0.0, dtype=Tp_obs.dtype, device=Tp_obs.device)

    if Np > 0:
        resP = Tp_obs[maskP] - t0_local - tp_model[maskP]
        llP = -0.5 * torch.sum(resP * resP) / sP2 - 0.5 * Np * (log2pi + torch.log(sP2))
        ll = ll + llP

    if Ns > 0:
        resS = Ts_obs[maskS] - t0_local - ts_model[maskS]
        llS = -0.5 * torch.sum(resS * resS) / sS2 - 0.5 * Ns * (log2pi + torch.log(sS2))
        ll = ll + llS

    return ll

def sample_invgamma(alpha: torch.Tensor, beta: torch.Tensor,
                    dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    采样 InvGamma(alpha, beta)；MPS 上回退 CPU。
    """
    alpha_cpu = torch.as_tensor(alpha, dtype=dtype, device='cpu')
    beta_cpu  = torch.as_tensor(beta,  dtype=dtype, device='cpu')
    g = torch.distributions.Gamma(concentration=alpha_cpu, rate=beta_cpu).sample(())
    invg = 1.0 / g
    return invg.to(device=device, dtype=dtype)
def sample_invgamma_batch(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    批量采样 InvGamma(alpha_i, beta_i)，返回 shape=[NC]。
    - CUDA/CPU: 原地采样
    - MPS: 显式移到 CPU 做 Gamma 采样（MPS 缺少 gamma sampling kernel），再拷回

    约定：InvGamma(alpha, beta) 可由 G ~ Gamma(alpha, rate=beta)，取 1/G 得到。
    """
    assert alpha.device == beta.device, "alpha 和 beta 必须在同一 device 上"
    device = alpha.device
    dtype  = alpha.dtype

    # 数值/合法性保护
    alpha_ = alpha.to(dtype=dtype).clamp_min(eps)
    beta_  = beta.to(dtype=dtype).clamp_min(eps)

    if device.type == "mps":
        # --- MPS workaround: sample on CPU ---
        a_cpu = alpha_.detach().to("cpu")
        b_cpu = beta_.detach().to("cpu")
        g_cpu = torch.distributions.Gamma(concentration=a_cpu, rate=b_cpu).sample()
        g_cpu = g_cpu.clamp_min(eps)
        invg_cpu = (1.0 / g_cpu).to(dtype=dtype)
        return invg_cpu.to(device=device)
    else:
        g = torch.distributions.Gamma(concentration=alpha_, rate=beta_).sample()
        g = g.clamp_min(eps)
        return 1.0 / g

# ===========================
# GMM 拟合 + logpdf + 采样
# ===========================
@torch.no_grad()
def fit_gmm_em_torch(
    ctlg: Tensor,
    n_components: int = 5,
    n_iters: int = 50,
    reg_eps: float = 1e-3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
):
    """
    纯 Torch EM（3D 全协方差），返回 {w, mu, L, log_det}
    L 为 Cholesky 下三角，Σ = L L^T
    """
    assert ctlg.ndim == 2 and ctlg.shape[1] == 3
    device = device or ctlg.device
    dtype  = dtype
    X = ctlg.to(device=device, dtype=dtype)
    N, D = X.shape
    M = min(n_components, max(1, N))

    if generator is None:
        generator = torch.Generator(device=device)

    # KMeans++ 粗初始化
    idx0 = torch.randint(0, N, (1,), device=device)
    mu = [X[idx0]]
    for _ in range(1, M):
        dist2 = torch.stack([torch.sum((X - m)**2, dim=1) for m in mu], dim=1).min(dim=1).values
        probs = (dist2 + 1e-12) / (dist2.sum() + 1e-12)
        idx = torch.multinomial(probs, 1)
        mu.append(X[idx])
    mu = torch.cat(mu, dim=0)  # (M,3)

    Xc = X - X.mean(dim=0, keepdim=True)
    Sigma0 = (Xc.T @ Xc) / max(N - 1, 1)
    Sigma0 = Sigma0 + reg_eps * torch.eye(D, device=device, dtype=dtype)
    L0 = torch.linalg.cholesky(Sigma0)

    L = L0.expand(M, D, D).clone()
    w = torch.full((M,), 1.0 / M, device=device, dtype=dtype)

    def log_gauss_L(x, mu_k, L_k):
        y = x - mu_k  # (N,3)
        z = torch.triangular_solve(y.T, L_k, upper=False)[0].T
        quad = torch.sum(z * z, dim=1)
        log_det = 2.0 * torch.sum(torch.log(torch.diag(L_k)))
        return -0.5 * (quad + log_det + D * math.log(2.0 * math.pi))

    for _ in range(n_iters):
        # E-step
        log_resp = []
        for k in range(M):
            lg = log_gauss_L(X, mu[k], L[k]) + torch.log(w[k] + 1e-12)
            log_resp.append(lg)
        log_resp = torch.stack(log_resp, dim=1)  # (N,M)
        maxv, _ = torch.max(log_resp, dim=1, keepdim=True)
        resp = torch.exp(log_resp - maxv)
        resp = resp / (resp.sum(dim=1, keepdim=True) + 1e-12)

        Nk = resp.sum(dim=0) + 1e-12
        w = Nk / Nk.sum()

        # M-step
        mu = (resp.T @ X) / Nk.unsqueeze(1)
        for k in range(M):
            Xmk = X - mu[k]
            Sk = (resp[:, k].unsqueeze(1) * Xmk).T @ Xmk / Nk[k]
            Sk = Sk + reg_eps * torch.eye(D, device=device, dtype=dtype)
            L[k] = torch.linalg.cholesky(Sk)

    log_det = torch.zeros((M,), device=device, dtype=dtype)
    for k in range(M):
        log_det[k] = 2.0 * torch.sum(torch.log(torch.diag(L[k])))

    return {"w": w, "mu": mu, "L": L, "log_det": log_det}

def gmm_logpdf(x: Tensor, gmm: Dict[str, Tensor]) -> Tensor:
    """
    log sum_k w_k N(x|mu_k, Σ_k)；x:(...,3) -> (...,)
    """
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
        lg = -0.5 * (quad + log_det[k] + D * math.log(2.0 * math.pi)) + torch.log(w[k] + 1e-12)
        logs.append(lg)
    logs = torch.stack(logs, dim=1)  # (Ntot,M)
    m, _ = torch.max(logs, dim=1, keepdim=True)
    res = m.squeeze(1) + torch.log(torch.sum(torch.exp(logs - m), dim=1) + 1e-12)
    return res.view(x.shape[:-1])

def sample_gmm(gmm: Dict[str, Tensor], generator: Optional[torch.Generator] = None) -> Tensor:
    """
    从 GMM 采一个样本；返回 (3,)
    """
    w, mu, L = gmm["w"], gmm["mu"], gmm["L"]
    device = w.device
    dtype_ = w.dtype
    if generator is None:
        generator = torch.Generator(device=device)
    M = w.shape[0]
    comp = torch.multinomial(w, 1).item()
    z = torch.randn((3,1), device=device, dtype=dtype_)  # (3,1)
    x = (mu[comp].unsqueeze(1) + L[comp] @ z).squeeze(1)  # (3,)
    return x

def sample_gmm_batch(
    gmm: Dict[str, Tensor],
    NC: int,
    generator: Optional[torch.Generator] = None
) -> Tensor:
    """
    从 GMM 批量采样 NC 个样本；返回形状 (NC, 3)
    """
    w, mu, L = gmm["w"], gmm["mu"], gmm["L"]
    device = w.device
    dtype_ = w.dtype

    if generator is None:
        generator = torch.Generator(device=device)

    M = w.shape[0]

    # 1) 根据权重 w 为每个样本选择一个组件：shape = (NC,)
    comp = torch.multinomial(w, NC, replacement=True)

    # 2) 从标准正态生成 z：shape = (NC, 3, 1)
    z = torch.randn((NC, 3, 1), device=device, dtype=dtype_)

    # 3) 根据组件索引取 mu 和 L：
    # mu_selected: (NC, 3, 1)
    mu_selected = mu[comp].unsqueeze(2)

    # L_selected: (NC, 3, 3)
    L_selected = L[comp]

    # 4) 采样 x = mu + L @ z → shape (NC, 3)
    x = (mu_selected + L_selected @ z).squeeze(-1)

    return x


def make_log_prior_xs(
    *,
    gmm_prior: Optional[Dict[str, Tensor]] = None,
    wide_sigma: float = 1000.0,
    prior_mix: Optional[float] = None,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    返回 log_prior_xs(x) 闭包；支持：
      - 仅 GMM ；仅宽高斯；或二者混合 log(a p1 + (1-a) p2)
    """
    wide_sigma2 = torch.as_tensor(wide_sigma**2, device=device, dtype=dtype)
    log_norm_wide = -0.5 * (3 * math.log(2.0 * math.pi) + 3 * torch.log(wide_sigma2))

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
        lg2 = log_wide(x)              + log1a
        m = torch.maximum(lg1, lg2)
        return m + torch.log(torch.exp(lg1 - m) + torch.exp(lg2 - m))
    return _f

# ===========================
# NUTS（只用于 xs；带掩码）
# ===========================
def logpost_xs_with_grad(
    xs: Tensor,
    *,
    xr: Tensor,
    Tp_obs: Tensor, Ts_obs: Tensor,
    t0: Tensor,
    sigmaP2: Tensor, sigmaS2: Tensor,
    log2pi: Tensor,
    log_prior_xs,  # 闭包
    maskP: Tensor, maskS: Tensor,
    Np: int, Ns: int,
) -> Tuple[Tensor, Tensor]:
    """
    计算 log posterior 与对 xs 的梯度（仅在 NUTS 内部用），带相位掩码。
    """
    xs = xs.requires_grad_(True)
    T_pred = model(xr, xs)  # (N,2) 带梯度
    tp_model = T_pred[:,0]
    ts_model = T_pred[:,1]

    ll = torch.tensor(0.0, dtype=Tp_obs.dtype, device=Tp_obs.device)
    if Np > 0:
        resP = Tp_obs[maskP] - t0 - tp_model[maskP]
        llP = -0.5 * torch.sum(resP * resP) / sigmaP2 - 0.5 * Np * (log2pi + torch.log(sigmaP2))
        ll = ll + llP
    if Ns > 0:
        resS = Ts_obs[maskS] - t0 - ts_model[maskS]
        llS = -0.5 * torch.sum(resS * resS) / sigmaS2 - 0.5 * Ns * (log2pi + torch.log(sigmaS2))
        ll = ll + llS

    logp = log_prior_xs(xs) + ll
    grad = torch.autograd.grad(logp, xs)[0]
    return logp.detach(), grad.detach()

def leapfrog(xs, r, eps, grad_fn):
    """
    单步 leapfrog；grad_fn 返回 (logp, grad)
    """
    logp, grad = grad_fn(xs)
    r_half = r + 0.5 * eps * grad
    xs_new = xs + eps * r_half
    logp_new, grad_new = grad_fn(xs_new)
    r_new = r_half + 0.5 * eps * grad_new
    H = -logp + 0.5 * torch.dot(r, r)
    H_new = -logp_new + 0.5 * torch.dot(r_new, r_new)
    return xs_new, r_new, logp_new, grad_new, H_new, H

def stop_criterion(xs_minus, xs_plus, r_minus, r_plus):
    """
    NUTS 停止条件：不再 U-turn
    """
    delta1 = torch.dot(xs_plus - xs_minus, r_minus)
    delta2 = torch.dot(xs_plus - xs_minus, r_plus)
    return (delta1 >= 0.0) and (delta2 >= 0.0)

def nuts_step(
    xs: Tensor,
    *,
    step_size: float,
    max_tree_depth: int,
    grad_fn,  # 返回 (logp, grad)
    generator: Optional[torch.Generator] = None,
    target_accept: float = 0.8,
):
    """
    单次 NUTS 构建树并返回新 xs、是否接受、以及新的 logp。
    简化实现，质量矩阵=I。
    """
    if generator is None:
        generator = torch.Generator(device=xs.device)

    r0 = torch.randn_like(xs)
    logp0, grad0 = grad_fn(xs)
    H0 = -logp0 + 0.5 * torch.dot(r0, r0)

    xs_minus = xs.clone()
    xs_plus  = xs.clone()
    r_minus  = r0.clone()
    r_plus   = r0.clone()
    xs_prop  = xs.clone()
    logp_prop = logp0.clone()
    s = True
    accept_stat = 0.0

    for depth in range(max_tree_depth):
        direction = 1 if torch.rand(()) < 0.5 else -1
        if direction == 1:
            xs_plus, r_plus, logp_new, grad_new, H_new, _ = leapfrog(xs_plus, r_plus, step_size, grad_fn)
        else:
            xs_minus, r_minus, logp_new, grad_new, H_new, _ = leapfrog(xs_minus, r_minus, -step_size, grad_fn)

        dH = H_new - H0
        if torch.isfinite(dH):
            prob = min(1.0, float(torch.exp(-dH).clamp(max=1e6)))
            if torch.rand(()) < prob:
                xs_prop = (xs_plus if direction == 1 else xs_minus).clone()
                logp_prop = logp_new.clone()
            accept_stat += min(1.0, float(torch.exp(-dH).clamp(max=1e6)))

        s = s and stop_criterion(xs_minus, xs_plus, r_minus, r_plus)
        if not s:
            break

    accepted = (xs_prop != xs).any()
    # 经验接受率（每层一次统计）
    accept_rate = float(accept_stat / max(1, depth + 1))
    return xs_prop, accepted, logp_prop, accept_rate
@torch.no_grad()
def forward_tt4_multi(xs_batch: torch.Tensor,
                      xr: torch.Tensor,
                      event_ids: torch.Tensor) -> torch.Tensor:
    """
    xs_batch: [NC,3]   每个事件的震源
    xr:       [N,3]    台站坐标（所有观测拼接）
    event_ids: [N]     每条观测对应的事件索引 in [0, NC-1]

    返回:
      T_pred: [N,2]  列0=Tp, 列1=Ts
    """
    # xs_per_obs 通过索引映射到每条观测，不复制 xr
    xs_per_obs = xs_batch[event_ids]           # [N,3]
    tps = model(xr, xs_per_obs)
    
    return tps 




def _scatter_counts(mask: torch.Tensor, event_ids: torch.Tensor, NC: int) -> torch.Tensor:
    # 统计每事件的观测条数（仅在 mask=True 的样本）
    ones = mask.to(torch.float32)
    return torch.zeros(NC, device=mask.device, dtype=torch.float32).scatter_add_(0, event_ids, ones)

def _scatter_sum(vals: torch.Tensor, mask: torch.Tensor, event_ids: torch.Tensor, NC: int) -> torch.Tensor:
    # 统计每事件的加和（仅 mask=True）
    v = torch.where(mask, vals.to(torch.float32), torch.tensor(0.0, device=vals.device, dtype=torch.float32))
    return torch.zeros(NC, device=vals.device, dtype=torch.float32).scatter_add_(0, event_ids, v)

def _scatter_sum_sq(vals: torch.Tensor, mask: torch.Tensor, event_ids: torch.Tensor, NC: int) -> torch.Tensor:
    v = torch.where(mask, (vals.to(torch.float32) ** 2), torch.tensor(0.0, device=vals.device, dtype=torch.float32))
    return torch.zeros(NC, device=vals.device, dtype=torch.float32).scatter_add_(0, event_ids, v)


def log_lik_per_event(
    T_pred: torch.Tensor,             # [N,2]
    Tp_obs: torch.Tensor, Ts_obs: torch.Tensor,  # [N]
    t0: torch.Tensor,                 # [NC]
    sigmaP2: torch.Tensor, sigmaS2: torch.Tensor,# [NC]
    log2pi: torch.Tensor,
    maskP: torch.Tensor, maskS: torch.Tensor,    # [N] bool
    event_ids: torch.Tensor,          # [N] long
    NC: int,
    lambdaP: Optional[torch.Tensor] = None,      # [N] or None
    lambdaS: Optional[torch.Tensor] = None,      # [N] or None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    若 lambdaP/lambdaS 不为 None，则按 Student-t scale mixture 做加权 RSS：
        rss = sum_i lambda_i * r_i^2
        N   = sum_i 1（仍然是样本数）
    """
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

    # P 残差与统计
    resP = Tp_obs - tp_model - t0[event_ids]
    # 加权残差平方和
    resP_w2 = lambdaP_eff * (resP.to(torch.float32) ** 2)
    rssP_e = _scatter_sum(resP_w2, maskP, event_ids, NC)     # 注意：用 _scatter_sum 而不是 _scatter_sum_sq
    Np_e   = _scatter_counts(maskP, event_ids, NC)           # 样本数还是未加权的

    # S 残差
    resS = Ts_obs - ts_model - t0[event_ids]
    resS_w2 = lambdaS_eff * (resS.to(torch.float32) ** 2)
    rssS_e = _scatter_sum(resS_w2, maskS, event_ids, NC)
    Ns_e   = _scatter_counts(maskS, event_ids, NC)

    sumP_e = _scatter_sum(resP, maskP, event_ids, NC)  
    sumS_e = _scatter_sum(resS, maskS, event_ids, NC)


    llP_e = torch.zeros(NC, device=Tp_obs.device, dtype=torch.float32)
    llS_e = torch.zeros(NC, device=Tp_obs.device, dtype=torch.float32)

    # P log-lik：近似仍用 Np_e 作为有效样本数（严格 Student-t 常数项会变，但差别在常数）
    validP = Np_e > 0
    if validP.any():
        llP_e[validP] = -0.5 * (rssP_e[validP] / sigmaP2[validP]) \
                        -0.5 * Np_e[validP] * (log2pi + torch.log(sigmaP2[validP]))

    validS = Ns_e > 0
    if validS.any():
        llS_e[validS] = -0.5 * (rssS_e[validS] / sigmaS2[validS]) \
                        -0.5 * Ns_e[validS] * (log2pi + torch.log(sigmaS2[validS]))

    ll_e = llP_e + llS_e
    return ll_e, Np_e, Ns_e, sumP_e, sumS_e, rssP_e, rssS_e



def _gamma_sample_shape_rate(shape: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """
    Sample from Gamma(shape, rate) with shape broadcasting to rate.
    - CUDA/CPU: sample on the same device.
    - MPS: move to CPU to sample (since gamma sampling ops are not implemented on MPS),
           then move back.
    """
    # Ensure tensors
    if shape.ndim == 0:
        shape_b = shape.expand_as(rate)
    else:
        shape_b = shape

    if rate.is_mps:
        # --- MPS workaround: sample on CPU ---
        shape_cpu = shape_b.detach().to("cpu")
        rate_cpu  = rate.detach().to("cpu")
        # torch.distributions.Gamma uses (concentration=shape, rate=rate)
        g = torch.distributions.Gamma(shape_cpu, rate_cpu)
        samp_cpu = g.sample()  # [N]
        return samp_cpu.to(device=rate.device, dtype=rate.dtype)
    else:
        g = torch.distributions.Gamma(shape_b, rate)
        return g.sample()
def _beta_sample(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """
    Sample Beta(a,b) using Gamma trick:
        X ~ Gamma(a, 1), Y ~ Gamma(b, 1),  Beta = X/(X+Y)

    Works on CPU/CUDA; on MPS uses your _gamma_sample_shape_rate CPU fallback.
    a,b can be scalar tensors or broadcastable tensors.
    """
    one = torch.ones_like(a if a.ndim > 0 else b)
    # Use rate=1
    x = _gamma_sample_shape_rate(a, one)
    y = _gamma_sample_shape_rate(b, one)
    return (x / (x + y + eps)).clamp(eps, 1.0 - eps)

def log_student_t0(r: torch.Tensor, sig2: torch.Tensor, nu: float) -> torch.Tensor:
    """
    log pdf of Student-t with df=nu, mean 0, scale sigma (variance = sig2*nu/(nu-2) if nu>2).
    Parameterization: r = sigma * t_nu
    pdf = Gamma((nu+1)/2)/[Gamma(nu/2)*sqrt(nu*pi)*sigma] * (1 + (r^2)/(nu*sigma^2))^(-(nu+1)/2)
    """
    nu_t = torch.tensor(float(nu), device=r.device, dtype=r.dtype)
    # logC = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(nu*pi) - log(sigma)
    logC = (torch.lgamma((nu_t + 1.0) / 2.0) -
            torch.lgamma(nu_t / 2.0) -
            0.5 * (torch.log(nu_t) + torch.log(torch.tensor(math.pi, device=r.device, dtype=r.dtype))) -
            0.5 * torch.log(sig2))
    quad = 1.0 + (r * r) / (nu_t * sig2)
    return logC - 0.5 * (nu_t + 1.0) * torch.log(quad)

from typing import List

@torch.no_grad()
def update_lambda_student_t_4ph(
    Tobs: torch.Tensor,            # [N,4]  columns: Pg,Sg,Pn,Sn
    T_pred_curr: torch.Tensor,     # [N,4]
    t0: torch.Tensor,              # [NC]
    sigma2: torch.Tensor,          # [4,NC]
    event_ids: torch.Tensor,       # [N]
    masks: List[torch.Tensor],     # len=4, each [N] bool
    lambdas: torch.Tensor,         # [N,4]
    nu: torch.Tensor,              # [4] (float or int)
    eps_rate: float = 1e-12,
) -> torch.Tensor:
    """
    Student-t scale mixture (per-phase):
      r | λ,σ² ~ N(0, σ²/λ)
      λ ~ Gamma(ν/2, ν/2)  (shape, rate)

    Gibbs:
      λ | r,σ² ~ Gamma((ν+1)/2, (ν + r²/σ²)/2)

    Notes:
      - sigma2 is per-event and per-phase: sigma2[j, e]
      - masks[j] indicates observed picks for that phase
      - lambdas[:, j] updated only where masks[j] is True
      - Uses your _gamma_sample_shape_rate(shape, rate) which should handle MPS fallback.
    """
    device = Tobs.device
    dtype  = Tobs.dtype

    # Coerce key tensors to same device/dtype
    event_ids = event_ids.to(device=device)
    t0 = t0.to(device=device, dtype=dtype)
    sigma2 = sigma2.to(device=device, dtype=dtype)
    T_pred_curr = T_pred_curr.to(device=device, dtype=dtype)
    lambdas = lambdas.to(device=device, dtype=dtype)

    # nu -> tensor [4] float dtype
    if not torch.is_tensor(nu):
        nu = torch.tensor(nu, device=device, dtype=dtype)
    else:
        nu = nu.to(device=device, dtype=dtype).view(-1)
    assert nu.numel() == 4, f"nu must be [4], got {tuple(nu.shape)}"

    N = Tobs.shape[0]
    assert Tobs.shape == (N, 4)
    assert T_pred_curr.shape == (N, 4)
    assert lambdas.shape == (N, 4)
    assert sigma2.shape[0] == 4 and sigma2.shape[1] == t0.numel()

    # residuals per phase: r = Tobs - t0[event] - Tpred
    # shape: [N,4]
    res = Tobs - t0[event_ids].unsqueeze(1) - T_pred_curr
    r2  = res * res

    # update each phase independently
    for j in range(4):
        m = masks[j].to(device=device)
        if not bool(m.any()):
            continue

        # sigma2 for those observations: [Nm]
        sig2_obs = sigma2[j, event_ids[m]]
        # rate = 0.5 * (nu + r2/sigma2)
        rate = 0.5 * (nu[j] + r2[m, j] / sig2_obs)
        rate = rate.clamp_min(eps_rate)

        # shape is scalar tensor: 0.5*(nu+1)
        shape = 0.5 * (nu[j] + 1.0)

        lam = _gamma_sample_shape_rate(shape, rate)  # [Nm]
        lambdas[m, j] = lam

    return lambdas



import math
from typing import Dict, Optional, List
import torch

# 你工程里已有的常量/函数我沿用名字：
# - MISSING_VAL
# - make_log_prior_xs(...)
# - log_student_t0(...)
# - update_lambda_student_t(...)  # 你需要扩展成 4 相版本（见下方）
# - sample_invgamma_batch(...)
# - _beta_sample(...)
# - build_phase_masks(...)        # 这里我自己写了 4 相版本
#
# 以及：你需要提供一个能输出 4 相走时的 forward：
# forward_tt4_multi(xs, xr, event_ids) -> Tensor[N,4]  columns: Pg,Sg,Pn,Sn

PHASES = ("Pg", "Sg", "Pn", "Sn")
KPH = 4

def build_phase_masks4(
    Tpg: torch.Tensor, Tsg: torch.Tensor, Tpn: torch.Tensor, Tsn: torch.Tensor,
    missing_value: float,
):
    mPg = torch.isfinite(Tpg) & (Tpg != missing_value)
    mSg = torch.isfinite(Tsg) & (Tsg != missing_value)
    mPn = torch.isfinite(Tpn) & (Tpn != missing_value)
    mSn = torch.isfinite(Tsn) & (Tsn != missing_value)
    return mPg, mSg, mPn, mSn


@torch.no_grad()
def gibbs_mh_location_multi_4ph(
    Tpg_obs: torch.Tensor, Tsg_obs: torch.Tensor, Tpn_obs: torch.Tensor, Tsn_obs: torch.Tensor,  # [N]
    xr: torch.Tensor,                              # [N,3]
    event_ids: torch.Tensor,                       # [N] long in [0..NC-1]
    NC: int,
    n_samples: int = 4000, burn: int = 1000, thin: int = 1,
    xs_init: Optional[torch.Tensor] = None,        # [NC,3] or None
    t0_init: float = 0.0,
    alpha0: float = 1e-2, beta0: float = 1e-2,
    # 每个 phase 一个 init（也可以都给同一个数）
    sigma_init: Optional[Dict[str, float]] = None,  # {"Pg":0.1,"Sg":0.1,"Pn":0.1,"Sn":0.1}
    prop_scale: float = 0.5,
    *,
    device_: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
    generator: Optional[torch.Generator] = None,
    # 先验（仅用于 log_prior_xs）
    gmm_prior: Optional[Dict[str, torch.Tensor]] = None,
    prior_mix: Optional[float] = None,
    wide_sigma: float = 1000.0,
    # RW 步长自适应
    adapt_steps: int = 1000,
    target_accept_rw: float = 0.3,
    adapt_eta: float = 0.05,
    # Student-t
    use_student_t: bool = True,

    # ===== 两分量污染模型参数（每个 phase 一套）=====
    a_pi: float = 8.0,
    b_pi: float = 2.0,
    pi_init: Optional[Dict[str, float]] = None,      # {"Pg":0.9,"Sg":0.9,"Pn":0.9,"Sn":0.9}
    sigma_out: Optional[Dict[str, float]] = None,    # {"Pg":15,"Sg":20,"Pn":15,"Sn":20}
) -> Dict[str, torch.Tensor]:

    # ----------------------------
    # device / generator
    # ----------------------------
    if device_ is None:
        # 尽量从输入推断
        device_eff = Tpg_obs.device if torch.is_tensor(Tpg_obs) else torch.device("cpu")
    else:
        device_eff = device_

    if generator is None:
        generator = torch.Generator(device=device_eff)

    # 默认参数
    if sigma_init is None:
        sigma_init = {"Pg": 0.1, "Sg": 0.1, "Pn": 0.1, "Sn": 0.1}
    if pi_init is None:
        pi_init = {"Pg": 0.9, "Sg": 0.9, "Pn": 0.9, "Sn": 0.9}
    if sigma_out is None:
        sigma_out = {"Pg": 15.0, "Sg": 20.0, "Pn": 15.0, "Sn": 20.0}

    # ----------------------------
    # 0) dtype / shape
    # ----------------------------
    Tpg_obs = torch.as_tensor(Tpg_obs, dtype=dtype, device=device_eff).view(-1)
    Tsg_obs = torch.as_tensor(Tsg_obs, dtype=dtype, device=device_eff).view(-1)
    Tpn_obs = torch.as_tensor(Tpn_obs, dtype=dtype, device=device_eff).view(-1)
    Tsn_obs = torch.as_tensor(Tsn_obs, dtype=dtype, device=device_eff).view(-1)

    xr = torch.as_tensor(xr, dtype=dtype, device=device_eff)
    event_ids = torch.as_tensor(event_ids, dtype=torch.long, device=device_eff).view(-1)

    N = Tpg_obs.numel()
    assert Tsg_obs.shape == (N,) and Tpn_obs.shape == (N,) and Tsn_obs.shape == (N,)
    assert xr.shape == (N, 3)
    assert event_ids.shape == (N,) and int(event_ids.max().item()) < NC

    # masks (observed)
    maskPg, maskSg, maskPn, maskSn = build_phase_masks4(
        Tpg_obs, Tsg_obs, Tpn_obs, Tsn_obs, missing_value=MISSING_VAL
    )
    masks: List[torch.Tensor] = [maskPg, maskSg, maskPn, maskSn]

    # Student-t df (每相可不同，这里先统一)
    nu = torch.tensor([4, 4, 4, 4], dtype=torch.int64, device=device_eff)

    # 将观测堆成 [N,4]
    Tobs = torch.stack([Tpg_obs, Tsg_obs, Tpn_obs, Tsn_obs], dim=1)  # [N,4]

    # latent lambdas (Student-t mixture), z, pi
    lambdas = torch.ones((N, KPH), dtype=dtype, device=device_eff)
    z = torch.ones((N, KPH), dtype=torch.bool, device=device_eff)  # good/outlier
    pi = torch.tensor(
        [float(pi_init[p]) for p in PHASES], dtype=dtype, device=device_eff
    )  # [4]

    sigOut2 = torch.tensor(
        [float(sigma_out[p] ** 2) for p in PHASES], dtype=dtype, device=device_eff
    )  # [4]

    # ----------------------------
    # 1) init xs
    # ----------------------------
    if xs_init is None:
        xs = torch.empty((NC, 3), device=device_eff, dtype=dtype)
        xr_global_mean = xr.mean(dim=0)
        valid_obs = maskPg | maskSg | maskPn | maskSn
        for e in range(NC):
            idx_e = (event_ids == e) & valid_obs
            if idx_e.any():
                xs[e] = xr[idx_e].mean(dim=0)
            else:
                idx_e2 = (event_ids == e)
                xs[e] = xr[idx_e2].mean(dim=0) if idx_e2.any() else xr_global_mean
        xs = xs + torch.tensor([0.0, 0.0, 5.0], device=device_eff, dtype=dtype).view(1, 3)
    else:
        xs = torch.as_tensor(xs_init, dtype=dtype, device=device_eff).view(NC, 3)

    # init t0, sigma^2 (per-event, per-phase)
    t0 = torch.full((NC,), float(t0_init), dtype=dtype, device=device_eff)
    sigma2 = torch.empty((KPH, NC), dtype=dtype, device=device_eff)  # [4,NC]
    for j, ph in enumerate(PHASES):
        sigma2[j] = float(sigma_init[ph] ** 2)

    # sample buffers
    M = (n_samples - burn) // max(1, thin)
    xs_samples = torch.empty((M, NC, 3), dtype=dtype, device=device_eff)
    t0_samples = torch.empty((M, NC), dtype=dtype, device=device_eff)
    sigma_samples = torch.empty((M, KPH, NC), dtype=dtype, device=device_eff)  # std
    pi_samples = torch.empty((M, KPH), dtype=dtype, device=device_eff)

    # constants
    log2pi = torch.tensor(math.log(2.0 * math.pi), dtype=dtype, device=device_eff)

    # prior closure
    log_prior_xs = make_log_prior_xs(
        gmm_prior=gmm_prior,
        wide_sigma=wide_sigma,
        prior_mix=prior_mix,
        device=device_eff,
        dtype=dtype,
    )

    # RW step scale (per-event)
    log_prop_scale = torch.full((NC,), math.log(max(prop_scale, 1e-6)), dtype=dtype, device=device_eff)
    accept_count_e = torch.zeros(NC, dtype=torch.int32, device=device_eff)
    total_prop = 0
    sample_idx = 0

    # forward once: [N,4] (Pg,Sg,Pn,Sn)
    T_pred_curr = forward_tt4_multi(xs, xr, event_ids)  # [N,4]

    # ----------------------------
    # helper: scatter sum for values with boolean mask
    # ----------------------------
    def scatter_sum_masked(val: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((NC,), dtype=dtype, device=device_eff)
        if m.any():
            out.scatter_add_(0, event_ids[m], val[m])
        return out

    def scatter_count_masked(m: torch.Tensor) -> torch.Tensor:
        return scatter_sum_masked(torch.ones((N,), dtype=dtype, device=device_eff), m)

    # ----------------------------
    # helper: log N(r | 0, var)  (elementwise)
    # ----------------------------
    def log_norm0(r: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return -0.5 * (log2pi + torch.log(var) + (r * r) / var)

    # ----------------------------
    # helper: log N(r | 0, sigma^2/lambda) for "good" (conditional Student-t)
    # ----------------------------
    def log_good_cond(r: torch.Tensor, sig2: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.log(lam) - 0.5 * (log2pi + torch.log(sig2) + (lam * r * r) / sig2)

    # ----------------------------
    # joint log-likelihood per-event (sum over obs & phases) + log p(z|pi) + prior(xs)
    # ----------------------------
    def log_joint_per_event(
        T_pred: torch.Tensor,   # [N,4]
        xs_: torch.Tensor,
        t0_: torch.Tensor,
        sigma2_: torch.Tensor,  # [4,NC]
        lambdas_: torch.Tensor, # [N,4]
        z_: torch.Tensor,       # [N,4]
        pi_: torch.Tensor,      # [4]
    ) -> torch.Tensor:
        # residuals: [N,4]
        res = Tobs - t0_[event_ids].unsqueeze(1) - T_pred
        ll = torch.zeros((N, KPH), dtype=dtype, device=device_eff)

        for j in range(KPH):
            m = masks[j]
            if not m.any():
                continue

            good = m & z_[:, j]
            bad = m & (~z_[:, j])

            sig2_i = sigma2_[j, event_ids]  # [N]

            if good.any():
                if use_student_t:
                    ll[good, j] = log_good_cond(res[good, j], sig2_i[good], lambdas_[good, j])
                else:
                    ll[good, j] = log_norm0(res[good, j], sig2_i[good])
                ll[good, j] += torch.log(pi_[j].clamp(min=1e-6))

            if bad.any():
                ll[bad, j] = log_norm0(res[bad, j], sigOut2[j])
                ll[bad, j] += torch.log((1.0 - pi_[j]).clamp(min=1e-6))

        ll_obs = ll.sum(dim=1)  # [N]
        any_obs = maskPg | maskSg | maskPn | maskSn
        ll_e = scatter_sum_masked(ll_obs, any_obs)
        return ll_e + log_prior_xs(xs_)

    # ----------------------------
    # update z (good/outlier) per phase
    # ----------------------------
    def update_z_two_component(
        T_pred: torch.Tensor,   # [N,4]
        t0_: torch.Tensor,
        sigma2_: torch.Tensor,  # [4,NC]
        lambdas_: torch.Tensor, # [N,4]
        pi_: torch.Tensor,      # [4]
        z_: torch.Tensor,       # [N,4]
    ):
        res = Tobs - t0_[event_ids].unsqueeze(1) - T_pred  # [N,4]

        for j in range(KPH):
            m = masks[j]
            if not m.any():
                continue

            sig2_i = sigma2_[j, event_ids]  # [N]

            if use_student_t:
                # 注意：你原来用 log_student_t0(res, sig2, nu)（边缘 t 分布）来更新 z
                log_good = log_student_t0(res[m, j], sig2_i[m], int(nu[j].item()))
            else:
                log_good = log_norm0(res[m, j], sig2_i[m])

            log_out = log_norm0(res[m, j], sigOut2[j])

            logp1 = torch.log(pi_[j].clamp(min=1e-6)) + log_good
            logp0 = torch.log((1.0 - pi_[j]).clamp(min=1e-6)) + log_out

            mmax = torch.maximum(logp1, logp0)
            denom = mmax + torch.log(torch.exp(logp1 - mmax) + torch.exp(logp0 - mmax))
            p1 = torch.exp(logp1 - denom).clamp(1e-6, 1 - 1e-6)

            u = torch.rand_like(p1)
            z_new = u < p1
            z_[m, j] = z_new

    # ----------------------------
    # init current logpost per-event
    # ----------------------------
    curr_logpost_e = log_joint_per_event(
        T_pred_curr, xs, t0, sigma2, lambdas, z, pi
    )

    if verbose:
        print("Start sampling (RW + Student-t + Good/Outlier mixture) with 4 phases: Pg,Sg,Pn,Sn...")

    for k in range(n_samples):
        if verbose and (k % 50 == 0):
            bad_frac = []
            for j in range(KPH):
                m = masks[j]
                frac = (m & (~z[:, j])).float().mean().item() if m.any() else 0.0
                bad_frac.append(frac)
            print(f"[iter {k}] "
                  f"pi={pi.detach().cpu().numpy()} "
                  f"bad={bad_frac}")

        # ===== A0. update lambdas (Student-t latent) =====
        if use_student_t:
            # 你需要一个 4 相版本：对每相、对每条观测更新 lambda
            # 形式建议：
            # update_lambda_student_t_4ph(Tobs, T_pred_curr, t0, sigma2, event_ids, masks, lambdas, nu)
            update_lambda_student_t_4ph(
                Tobs, T_pred_curr,
                t0, sigma2,
                event_ids, masks,
                lambdas, nu,
            )

        # ===== A1. update z (good/outlier indicators) =====
        update_z_two_component(T_pred_curr, t0, sigma2, lambdas, pi, z)

        # ===== A2. update pi (Beta posterior) =====
        for j in range(KPH):
            m = masks[j]
            if not m.any():
                continue
            n_good = (m & z[:, j]).sum().to(dtype)
            n_bad  = (m & (~z[:, j])).sum().to(dtype)
            a = torch.tensor(a_pi, device=device_eff, dtype=dtype) + n_good
            b = torch.tensor(b_pi, device=device_eff, dtype=dtype) + n_bad
            pi[j] = _beta_sample(a, b, eps=1e-6)

        # ===== B. RW-MH update xs (per-event, vectorized) =====
        total_prop += 1

        steps = torch.randn((NC, 3), dtype=dtype, device=device_eff, generator=generator) \
                * log_prop_scale.exp().unsqueeze(1)
        xs_prop = xs + steps

        T_pred_prop = forward_tt4_multi(xs_prop, xr, event_ids)  # [N,4]

        prop_logpost_e = log_joint_per_event(
            T_pred_prop, xs_prop, t0, sigma2, lambdas, z, pi
        )

        log_alpha_e = prop_logpost_e - curr_logpost_e
        u = torch.rand((NC,), dtype=dtype, device=device_eff, generator=generator).log()
        accept_e = (u < log_alpha_e)

        if accept_e.any():
            xs[accept_e] = xs_prop[accept_e]
            curr_logpost_e[accept_e] = prop_logpost_e[accept_e]
            accept_count_e[accept_e] += 1

            accept_obs = accept_e[event_ids]
            T_pred_curr[accept_obs] = T_pred_prop[accept_obs]

        # RW adapt
        if k < adapt_steps:
            log_prop_scale += adapt_eta * (accept_e.to(dtype) - target_accept_rw)

        # ===== C. update t0 (condition on GOOD only, all phases) =====
        denom = torch.zeros((NC,), dtype=dtype, device=device_eff)
        num = torch.zeros((NC,), dtype=dtype, device=device_eff)

        for j in range(KPH):
            m_good = masks[j] & z[:, j]
            if not m_good.any():
                continue

            # (T_obs - T_model)
            d = (Tobs[:, j] - T_pred_curr[:, j])  # [N]
            if use_student_t:
                sum_nom = scatter_sum_masked(lambdas[:, j] * d, m_good)
                n_eff = scatter_sum_masked(lambdas[:, j], m_good)
            else:
                sum_nom = scatter_sum_masked(d, m_good)
                n_eff = scatter_count_masked(m_good)

            sig2_e = sigma2[j]  # [NC]
            valid = n_eff > 0
            denom[valid] += n_eff[valid] / sig2_e[valid]
            num[valid]   += sum_nom[valid] / sig2_e[valid]

        upd = denom > 0
        if upd.any():
            var = torch.zeros_like(denom)
            mean = torch.zeros_like(denom)
            var[upd] = 1.0 / denom[upd]
            mean[upd] = var[upd] * num[upd]
            noise = torch.zeros_like(mean)
            noise[upd] = torch.randn((int(upd.sum().item()),), dtype=dtype, device=device_eff, generator=generator)
            t0[upd] = mean[upd] + noise[upd] * var[upd].sqrt()

        # ===== D. update sigma2 per phase (InvGamma; GOOD only) =====
        res = Tobs - t0[event_ids].unsqueeze(1) - T_pred_curr  # [N,4]

        for j in range(KPH):
            m_good = masks[j] & z[:, j]
            alpha = torch.full((NC,), alpha0, dtype=dtype, device=device_eff)
            beta  = torch.full((NC,), beta0,  dtype=dtype, device=device_eff)

            if not m_good.any():
                sigma2[j] = sample_invgamma_batch(alpha, beta)
                continue

            if use_student_t:
                rss = scatter_sum_masked(lambdas[:, j] * (res[:, j] ** 2), m_good)
                n_eff = scatter_sum_masked(lambdas[:, j], m_good)
            else:
                rss = scatter_sum_masked((res[:, j] ** 2), m_good)
                n_eff = scatter_count_masked(m_good)

            valid = n_eff > 0
            if valid.any():
                alpha[valid] += 0.5 * n_eff[valid]
                beta[valid]  += 0.5 * rss[valid]

            sigma2[j] = sample_invgamma_batch(alpha, beta)

        # ===== E. recompute current joint logpost per-event =====
        curr_logpost_e = log_joint_per_event(
            T_pred_curr, xs, t0, sigma2, lambdas, z, pi
        )

        # ===== F. record =====
        if k >= burn and ((k - burn) % thin == 0):
            xs_samples[sample_idx] = xs
            t0_samples[sample_idx] = t0
            sigma_samples[sample_idx] = sigma2.sqrt()  # [4,NC]
            pi_samples[sample_idx] = pi
            sample_idx += 1

    accept_rate_e = accept_count_e.to(torch.float32) / max(1, total_prop)

    return {
        "xs_samples": xs_samples,                 # [M, NC, 3]
        "t0_samples": t0_samples,                 # [M, NC]
        "sigma_samples": sigma_samples,           # [M, 4, NC]   std
        "pi_samples": pi_samples,                 # [M, 4]
        "accept_rate_per_event": accept_rate_e,   # [NC]
        "final_prop_scale_per_event": log_prop_scale.exp(),  # [NC]
        "z_final": z,                             # [N,4]
        "pi_final": pi,                           # [4]
    }




import datetime 
import numpy as np 
import pickle 

import re

import re

_INT_PREFIX_RE = re.compile(r'^\s*[+-]?\d+(?![\d.])')

def starts_with_int(s: str) -> bool:
    return _INT_PREFIX_RE.match(s) is not None

def read_station_file(file_path='data/station.txt'):
    stations = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            network = parts[0]
            station_code = parts[1]
            network_id = parts[2]
            lon = float(parts[2])
            lat = float(parts[3])
            ele = float(parts[4]) 
            stations[f"{network}{station_code}"] = np.array([lon, lat, -ele/1000])
    return stations 

def build_inverse_aeqd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(crs_aeqd, CRS.from_epsg(4326), always_xy=True)

def build_forward_aeqd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(CRS.from_epsg(4326), crs_aeqd, always_xy=True)  # lon,lat -> x,y (m)


def read_real_file(file_path, station_path):
    #with open("data/vel.py.model", 'rb') as f:
    #    grid_vp, grid_vs, grid_x, grid_y, grid_z, proj, *rest = pickle.load(f)
    proj_inv = build_inverse_aeqd(102.5, 27.5)
    proj_fwd = build_forward_aeqd(102.5, 27.5)
    events = []
    stloc = read_station_file(station_path) 
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    #10950   2022 09 15 15:57:18.825    57438.825  0.2910 29.8570 101.8684   6.00    0.0   0.0  11  11  22   8  95.26
    #SC      JLO     P   57455.8100   16.9846 0.00e+00 -0.1447   0.6710  19.8026
    #SC     TGXJ     S   57464.0100   25.1846 0.00e+00 -0.6463   0.2680  23.6516
    #SC    KDXDQ     P   57445.9200    7.0946 0.00e+00 -0.3215   0.9550 118.9069
    #SC     YLXJ     S   57450.2100   11.3846 0.00e+00  0.1281   0.3880 186.7945
    #SC     YLXJ     P   57445.4800    6.6546 0.00e+00  0.1457   0.8010 186.7945
    i = 0
    while i < len(lines):
        # 读取事件头信息
        #print(starts_with_int(lines[i]), lines[i])
        if starts_with_int(lines[i]):
            #print(lines[i])
            evid = int(lines[i].strip().split()[0])
            header = lines[i][:].strip().split()[1:]
            # 时间
            try:
                origin_time = datetime.datetime.strptime(" ".join(header[0:4]), "%Y %m %d %H %M %S.%f")
            except:
                origin_time = datetime.datetime.strptime(" ".join(header[0:4]).replace("60", "59"), "%Y %m %d %H:%M:%S.%f")
            lat = float(header[6])
            lon = float(header[7])
            depth = float(header[8])
            mag1 = float(header[9])
            mag2 = float(header[10])
            #print(lat, lon, depth)
            base_delta = float(header[4])  # 这个是儒略日
            # 后面的几个是未知或保留字段
            event_info = {
                "evid":evid, 
                "origin_time": origin_time,
                "latitude": lat,
                "longitude": lon,
                "depth": depth,
                "mag1": mag1,
                "mag2": mag2,
                "phases": []
            }

            # 读取震相行
            i += 1
            while i < len(lines) and not starts_with_int(lines[i]):
                parts = lines[i].split()
                station = "".join(parts[:2])
                travel_time = float(parts[4])# - base_delta 
                weight = 0.0
                phase = parts[2]
                event_info["phases"].append({
                    "station": station,
                    "travel_time": travel_time,
                    "weight": weight,
                    "phase": phase
                })
                i += 1

            events.append(event_info)
        else:
            i += 1
    loc_events = []
    all_events = []
    for e in events: 
        #print("事件时间:", e["origin_time"])
        #print("位置: lat=%.4f lon=%.4f depth=%.2f" % (e["latitude"], e["longitude"], e["depth"]))
        #print("震相数据:")
        #locinfo = []
        tstr = e["origin_time"].strftime("%Y%m%d")
        #if "0905" not in tstr:
        #    continue
        ex, ey = proj_fwd.transform(e["longitude"], e["latitude"]) 
        ex, ey = ex / 1000., ey / 1000.
        ez = e["depth"]
        all_events.append([ex, ey, ez])
        #events_loc.append(e)
        stations = []
        for p in e["phases"]:
            stations.append(p["station"])

        set_stations = list(set(stations))
        #print(set_stations)
        st2id = {}
        for i, station in enumerate(set_stations):
            st2id[station] = i
        rcv = np.zeros([len(set_stations), 3])
        T_p = np.ones([len(set_stations)]) * MISSING_VAL
        T_s = np.ones([len(set_stations)]) * MISSING_VAL
        #print(stloc)
        for st in set_stations:
            #print(st)
            if st not in stloc:
                continue
            loc = stloc[st]
            #print(loc)
            x, y = proj_fwd.transform(loc[0], loc[1])
            x, y = x / 1000., y / 1000. 
            z = loc[2] 
            if x < -1000 or x > 1000:
                continue
            if y < -1000 or y > 1000:
                continue
            if z < -10 or z > 100:
                continue
            rcv[st2id[st], :] = np.array([x, y, z]) 
    
        for p in e["phases"]: 
            if p['station'] not in stloc:
                continue
            loc = stloc[p['station']]
            x, y = proj_fwd.transform(loc[0], loc[1])
            x, y = x / 1000., y / 1000.
            z = loc[2] 
            if x < -1000 or x > 1000:
                continue
            if y < -1000 or y > 1000:
                continue
            if z < -10 or z > 100:
                continue
            if p["phase"] == "P":
                T_p[st2id[p["station"]]] = p['travel_time'] 
            elif p["phase"] == "S":
                T_s[st2id[p["station"]]] = p['travel_time']
        locinfo = {
            "rcv":torch.tensor(rcv.astype(np.float32), device=device).float(), 
            "T_p":torch.tensor(T_p.astype(np.float32), device=device).float(),
            "T_s":torch.tensor(T_s.astype(np.float32), device=device).float(),
        }
        loc_events.append({
            "evid":e["evid"], 
            "etime":e["origin_time"], 
            "locinfo":locinfo, 
            "mag1":e["mag1"],
            "mag2":e["mag2"],
            "eloc": [e["longitude"], e["latitude"], e["depth"]], 
            "eloc_km": [ex, ey, ez], 
        })
    all_events = np.array(all_events).astype(np.float32)
    all_events = torch.tensor(all_events, device=device).float()
    return loc_events, all_events, proj_fwd, proj_inv


import datetime
import re
import numpy as np
import torch

def read_event_file(file_path, station_path):
    """
    Read event file in CSV-like format:

    #LD.20220905033033730000,2022-09-05 03:30:33.730000,102.025,29.581,6,-123456,3,3,3,3,
    Pg,Pg,1.0,1.8497,2022-09-05 03:30:35.579700,102.066,29.690,12.7619,1.000,SC.LDXX
    Pn,Pn,1.0,2.0123,2022-09-05 03:30:35.742300,102.066,29.690,12.7619,1.000,SC.LDXX
    Sg,Sg,1.0,3.4567,2022-09-05 03:30:37.186700,102.066,29.690,12.7619,1.000,SC.LDXX
    Sn,Sn,1.0,3.9876,2022-09-05 03:30:37.717600,102.066,29.690,12.7619,1.000,SC.LDXX
    ...

    Output matches your old function structure but with 4-phase travel times:
        loc_events, all_events, proj_fwd, proj_inv

    loc_events[*]["locinfo"] contains:
        - "rcv": [Ns,3]
        - "T_pg","T_sg","T_pn","T_sn": [Ns]  (missing -> MISSING_VAL)
    """
    proj_inv = build_inverse_aeqd(102.5, 27.5)
    proj_fwd = build_forward_aeqd(102.5, 27.5)

    stloc = read_station_file(station_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    def is_header(line: str) -> bool:
        return line.startswith("#")

    def parse_time(s: str) -> datetime.datetime:
        s = (s or "").strip()
        if not s:
            raise ValueError("empty time string")

        s = s.replace("T", " ")
        if s.endswith("Z"):
            s = s[:-1].strip()
        s = re.sub(r"([+-]\d{2}:\d{2})$", "", s).strip()

        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.datetime.strptime(s, fmt)
            except ValueError:
                pass

        m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(\.\d+)$", s)
        if m:
            base, frac = m.group(1), m.group(2)[1:]
            frac = (frac + "000000")[:6]
            return datetime.datetime.strptime(f"{base}.{frac}", "%Y-%m-%d %H:%M:%S.%f")

        raise ValueError(f"time data {s!r} not recognized")

    def parse_station_key(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        return s.replace(".", "").replace(" ", "")

    # 只保留这四种（其它相位忽略）
    PHASE2IDX = {"Pg": 0, "Sg": 1, "Pn": 2, "Sn": 3}

    events = []
    i = 0
    while i < len(lines):
        if not is_header(lines[i]):
            i += 1
            continue

        # -------------------------
        # 1) parse event header
        # -------------------------
        header_line = lines[i].lstrip("#").strip()
        cols = [c.strip() for c in header_line.split(",") if c.strip() != ""]

        evid_str = cols[0] if len(cols) > 0 else f"EV{i}"
        m = re.findall(r"\d+", evid_str)
        evid = int(max(m, key=len)) if m else i

        origin_time = parse_time(cols[1])
        lon = float(cols[2])
        lat = float(cols[3])
        depth = float(cols[4])

        mag1 = 0.0
        mag2 = 0.0
        if len(cols) >= 7:
            try:
                mag1 = float(cols[6])
            except Exception:
                mag1 = 0.0
        if len(cols) >= 8:
            try:
                mag2 = float(cols[7])
            except Exception:
                mag2 = mag1

        event_info = {
            "evid": evid,
            "origin_time": origin_time,
            "latitude": lat,
            "longitude": lon,
            "depth": depth,
            "mag1": mag1,
            "mag2": mag2,
            "phases": [],  # raw phase lines
        }

        # -------------------------
        # 2) parse phase lines
        # -------------------------
        i += 1
        while i < len(lines) and (not is_header(lines[i])):
            parts = [c.strip() for c in lines[i].split(",")]
            # 0=ph_main,1=ph2,2=...,3=travel_time,4=pick_time,5=stlon,6=stlat,7=dist_km,8=weight,9=station
            if len(parts) < 10:
                i += 1
                continue

            ph_main = parts[0]
            if ph_main not in PHASE2IDX:
                i += 1
                continue

            try:
                travel_time = float(parts[3])
                if travel_time > 100:  # 你原来的规则
                    i += 1
                    continue
            except Exception:
                i += 1
                continue

            try:
                weight = float(parts[8])
            except Exception:
                weight = 0.0

            station = parse_station_key(parts[9])

            event_info["phases"].append({
                "station": station,
                "travel_time": travel_time,
                "weight": weight,
                "phase": ph_main,          # Pg/Sg/Pn/Sn
                "phase_idx": PHASE2IDX[ph_main],
            })
            i += 1

        events.append(event_info)

    # =========================================================
    # 3) build outputs
    # =========================================================
    loc_events = []
    all_events = []

    for e in events:
        ex, ey = proj_fwd.transform(e["longitude"], e["latitude"])
        ex, ey = ex / 1000.0, ey / 1000.0
        ez = e["depth"]
        all_events.append([ex, ey, ez])

        # --- stations in this event
        stations = [p["station"] for p in e["phases"] if p["station"]]
        set_stations = list(set(stations))
        st2id = {st: idx for idx, st in enumerate(set_stations)}

        Ns = len(set_stations)
        rcv = np.zeros([Ns, 3], dtype=np.float32)

        T_pg = np.ones([Ns], dtype=np.float32) * MISSING_VAL
        T_sg = np.ones([Ns], dtype=np.float32) * MISSING_VAL
        T_pn = np.ones([Ns], dtype=np.float32) * MISSING_VAL
        T_sn = np.ones([Ns], dtype=np.float32) * MISSING_VAL
        T_all = [T_pg, T_sg, T_pn, T_sn]

        # --- receiver positions + station range filter (keep same logic)
        valid_station = np.zeros([Ns], dtype=np.bool_)
        for st in set_stations:
            if st not in stloc:
                continue
            loc = stloc[st]
            x, y = proj_fwd.transform(loc[0], loc[1])
            x, y = x / 1000.0, y / 1000.0
            z = loc[2]

            if x < -1000 or x > 1000:
                continue
            if y < -1000 or y > 1000:
                continue
            if z < -10 or z > 100:
                continue

            idx = st2id[st]
            rcv[idx, :] = np.array([x, y, z], dtype=np.float32)
            valid_station[idx] = True

        # --- per (station, phase) pick: keep the minimum travel_time (or you can weight-based select)
        # best[(st, phase_idx)] = travel_time
        best = {}
        for p in e["phases"]:
            st = p["station"]
            if (not st) or (st not in stloc):
                continue

            idx = st2id.get(st, None)
            if idx is None or (not valid_station[idx]):
                continue

            j = p["phase_idx"]
            tt = p["travel_time"]
            key = (st, j)
            if key not in best or tt < best[key]:
                best[key] = tt

        # write into arrays
        for (st, j), tt in best.items():
            idx = st2id[st]
            T_all[j][idx] = tt

        locinfo = {
            "rcv": torch.tensor(rcv, device=device).float(),
            "T_pg": torch.tensor(T_pg, device=device).float(),
            "T_sg": torch.tensor(T_sg, device=device).float(),
            "T_pn": torch.tensor(T_pn, device=device).float(),
            "T_sn": torch.tensor(T_sn, device=device).float(),
        }

        loc_events.append({
            "evid": e["evid"],
            "etime": e["origin_time"],
            "locinfo": locinfo,
            "mag1": e["mag1"],
            "mag2": e["mag2"],
            "eloc": [e["longitude"], e["latitude"], e["depth"]],
            "eloc_km": [ex, ey, ez],
        })

    all_events = np.array(all_events, dtype=np.float32)
    all_events = torch.tensor(all_events, device=device).float()

    return loc_events, all_events, proj_fwd, proj_inv

# ---------------------------
# 示例：生成数据并运行（含缺测示例）
# ---------------------------
import random 
import tqdm 
if __name__ == "__main__":
    #data_tool = GenerateDataFromTrueData()
    # 固定 Python 内置随机数
    random.seed(42)

    # 固定 NumPy 随机数
    np.random.seed(42)

    # 固定 PyTorch CPU 随机数
    torch.manual_seed(42)
    # ======目录 ctlg: (K,3) —— 请替换为你的真实目录 ======
    events, ctlg, proj_fwd, proj_inv = read_event_file("data/real.crop.txt", "data/station.txt")
    # 拟合 GMM 作为先验
    #gmm = fit_gmm_em_torch(
    #    ctlg, n_components=30, n_iters=100, reg_eps=5e-2,
    #    device=device, dtype=dtype
    #)

    # ====== 生成一批数据并演示缺测 ======
    
    rcvs = []
    c = 0 
    event_ids = []
    Tpg_obs = []
    Tsg_obs = []
    Tpn_obs = []
    Tsn_obs = []    
    xr = []
    xs_init = []
    for eve, ctg in tqdm.tqdm(zip(events, ctlg)):
        #print(eve)
        etime = eve["etime"]
        locinfo = eve["locinfo"]
        rcv = locinfo["rcv"] 
        Tpg = locinfo["T_pg"]
        Tsg = locinfo["T_sg"]
        Tpn = locinfo["T_pn"]
        Tsn = locinfo["T_sn"]
        
        #print(rcv.shape, T_p.shape, T_s.shape)
        #print(T_p, T_s)
        xs_init.append(eve["eloc_km"])
        event_id = torch.ones([len(rcv)], device=device)
        event_ids.append(event_id * c)
        c += 1
        #print("src/rcv/T shapes:", rcv.shape, T_p.shape, T_s.shape)
        t1 = time.perf_counter()
        rcvs.append({"xr":rcv, "Tpg":Tpg, "Tsg":Tsg, "Tpn":Tpn, "Tsn":Tsn})
        xr.append(rcv)
        Tpg_obs.append(Tpg)
        Tsg_obs.append(Tsg)
        Tpn_obs.append(Tpn)
        Tsn_obs.append(Tsn)
    event_ids = torch.cat(event_ids) 
    Tpg_obs = torch.cat(Tpg_obs) 
    Tsg_obs = torch.cat(Tsg_obs) 
    Tpn_obs = torch.cat(Tpn_obs)
    Tsn_obs = torch.cat(Tsn_obs)
    xr = torch.cat(xr) 
    t1 = time.perf_counter()
    alpha0 = 3.0
    beta0  = (0.5 ** 2) * (alpha0 - 1)  # 假设先验期望 ~ 0.5^2
    use_student_t = True 
    out = gibbs_mh_location_multi_4ph(
        Tpg_obs, Tsg_obs, Tpn_obs, Tsn_obs, xr,
        xs_init=None,
        event_ids=event_ids, NC=c,
        n_samples=4000, burn=2000, thin=2,
        gmm_prior=None, prior_mix=None,  # 或仅用宽高斯
        prop_scale=2,
        device_=device, dtype=dtype, verbose=True, 
        alpha0=alpha0, beta0=beta0,
        use_student_t=use_student_t,
    )
    t2 = time.perf_counter() 
    print("Gibbs MH 耗时:", t2-t1)
    print("Gibbs MH 耗时(per sample):", (t2-t1)/len(rcvs))
    xs_samples = out["xs_samples"]#.cpu().numpy()
    t0_samples = out["t0_samples"]#.cpu().numpy()
    ofile = open(f"run_fm3d/data/ours/reloc.crop.{use_student_t}.pnsn.v1.0.txt", "w")
    #print(len(events), len(ctlg), xs_samples.shape, t0_samples.shape)
    #41.45272073103115
    #
    xs_samples = xs_samples.float()
    t0_samples = t0_samples.float()
    for idx, (eve, ctg) in enumerate(zip(events, ctlg)):
        xs_samps, t0_samps = xs_samples[:, idx, :], t0_samples[:, idx]
        xs_mean = xs_samps.mean(dim=0)
        xs_std  = xs_samps.std(dim=0, unbiased=True)

        t0_mean = t0_samps.mean()
        etime = eve["etime"]
        locinfo = eve["locinfo"]
        evid = eve["evid"]
        event_orig_loc = eve["eloc"]
        #print("True xs:", src.detach().cpu().numpy())
        t0_mean = t0_mean.flatten().detach().cpu().numpy() 
        xs_std = xs_std.flatten().detach().cpu().numpy()

        q = torch.tensor([0.05, 0.95], device=xs_samps.device, dtype=xs_samps.dtype)
        err = [ ]
        for i, name in enumerate(['x', 'y', 'z']):
            lo, hi = torch.quantile(xs_samps[:, i], q).tolist()
            #ofile.write(f" {name}: [{lo:.3f}, {hi:.3f}], range: {(hi - lo):.3f}\n")
            err.append(hi - lo)
        #if err[0] > 10.0 or err[1] > 10.0 or err[2] > 20.0:
        #    continue 

        #print("Posterior mean xs:", xs_mean.detach().cpu().numpy())
        #print("Posterior std  xs:", xs_std.detach().cpu().numpy())
        tstr = (etime+datetime.timedelta(seconds=float(t0_mean[0]))).strftime("%Y-%m-%d %H:%M:%S.%f")
        x, y, z = xs_mean.detach().cpu().numpy()
        lon, lat = proj_inv.transform(x * 1000.0, y * 1000.0)

        ofile.write(f"#{evid},{tstr},{lon},{lat},{z},{x},{y},{z},{xs_std[0]},{xs_std[1]},{xs_std[2]},{err[0]},{err[1]},{err[2]},{event_orig_loc[0]},{event_orig_loc[1]},{event_orig_loc[2]}\n")
        ctg = ctg.cpu().numpy()
        x, y = proj_inv.transform(ctg[0] * 1000.0, ctg[1] * 1000.0)
        z = ctg[2]
        ofile.write(f"{ctg[0]},{ctg[1]},{ctg[2]},{x},{y},{z},{eve['mag1']},{eve['mag2']}\n")
        # 置信区间
        q = torch.tensor([0.025, 0.975], device=xs_samps.device, dtype=xs_samps.dtype)
        for i, name in enumerate(['x', 'y', 'z']):
            lo, hi = torch.quantile(xs_samps[:, i], q).tolist()
            ofile.write(f" {name}: [{lo:.3f}, {hi:.3f}], range: {(hi - lo):.3f}\n")
        ofile.flush()
        continue 
        print("accept_rate_event ", out["accept_rate_per_event"][idx].item())
        print("first 10 samples x,y,z:")
        print(xs_samps[:10])
        print("std:", xs_samps.std(dim=0, unbiased=True))
        sigP_e = out["sigmaP_samples"][:, idx]
        print("sigmaP_e median / mean:", sigP_e.median().item(), sigP_e.mean().item())
        xs_samps = out["xs_samples"][:, idx, :]
        print("xs std:", xs_samps.std(dim=0, unbiased=True))
        print("sigmaP mean / median:", out["sigmaP_samples"].mean().item(),
            out["sigmaP_samples"].median().item())
        print("sigmaS mean / median:", out["sigmaS_samples"].mean().item(),
            out["sigmaS_samples"].median().item())
        
        time.sleep(3.0)
        #xs_samples=xs_samples,       # [NC, M, 3]
        #t0_samples=t0_samples,       # [NC, M]
        #sigmaP_samples=sigmaP_samples,
        #sigmaS_samples=sigmaS_samples,
        #accept_rate=accept_rate,     # [NC]
        #final_prop_scale=log_prop_scale.exp(),  # [NC]