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

from models.mlp import PINNTravelTime, TauPSNet
import numpy as np 
import random 
# ================== 设备与随机种子 ==================

SEED   = 2024
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
# ================== 域与超参 ==================
# 建议数据侧把物理坐标无量纲化/归一到 [-1,1]^3
DOMAIN_MIN = -1000
DOMAIN_MAX =  1000 

# Godunov / 连续残差
USE_GODUNOV = True
H_REL       = 0.01    # 差分步长比例 h = H_REL * (DOMAIN_MAX - DOMAIN_MIN)

# 源点球 + 采样
EPS        = 0.02
B_MAX      = 16       # 每步用于 PDE/BC 的最大“唯一源点”数
M_PDE      = 1024     # 每源 PDE 采样点
K_BC       = 512      # 每源球面 BC 点
RAR_EVERY  = 500      # RAR 周期（步），0 关闭
RAR_TOPK   = 1024     # 每源 RAR 加密点数上限

# 课程式权重 & 损失
LPDE_START = 0.1
LPDE_END   = 3.0
LBC        = 10.0
LDAT_P     = 1.0
LDAT_S     = 1.0

# 优化
LR         = 1e-3
GRAD_CLIP  = 1.0
STEPS1     = 4000     # Adam
STEPS2     = 1000     # 第二阶段（Adam 或 LBFGS）
USE_LBFGS  = False

# 网络结构
HIDDEN     = 512 
LAYERS     = 8 
USE_FF     = True
FF_NUM     = 16
FF_B       = 8.0

# =============== 速度模型（默认常速；请替换为你的 3D 查询） ===============
VP0 = 6.0  # km/s
VS0 = 3.5

# ---------------------------
# 设备与全局设置
# ---------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.bfloat16

# 缺测标记
MISSING_VAL = -12345.0

# ---------------------------
# 模型
# ---------------------------
model = TauPSNet().to(device).eval()
ckpt_path = 'ckpt/pinn_eikonal_ps_one_net.v2.1.pt'
model.load_state_dict(torch.load(ckpt_path, map_location='cpu')["net"])
model.to(dtype)

# 前向计数器（用于诊断是否真正减少了前向次数）
forward_count = 0

@torch.no_grad()
def forward_tps(xs: Tensor, xr: Tensor) -> Tensor:
    """
    一次前向返回 (Tp, Ts)，float32。
    xs: (3,), xr: (N,3)
    返回: (N, 2) float32, device=model.device
    """
    global forward_count
    xs_f = xs.to(device=device, dtype=torch.float32)
    xr_f = xr.to(device=device, dtype=torch.float32)
    forward_count += 1
    return model(xr_f, xs_f)  # (N, 2)

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
def build_feats(x: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """条件特征：逐样本一一对应 (x_i, xs_i) -> feats_i"""
    r = x - xs
    d = torch.linalg.norm(r, dim=-1, keepdim=True) + 1e-6 
    u = r / d
    return torch.cat([x, xs, r, u, d], dim=-1)  # [N,13]
# ===========================
# 批量前向（仅计算真实台站）
# ===========================
@torch.no_grad()
def forward_tps_batched(xs_batch: torch.Tensor, XR: torch.Tensor, exist_mask: torch.Tensor) -> torch.Tensor:
    """
    xs_batch:  [NC, 3]
    XR:        [NC, Nmax, 3]
    exist_mask:[NC, Nmax]
    return:    T_pred [NC, Nmax, 2]，无效位置置 0
    """
    NC, Nmax, _ = XR.shape
    flat_mask = exist_mask.view(-1)                    # [NC*Nmax]
    x_all = XR.view(-1, 3)[flat_mask]                  # [Ntot,3]

    evt_ids = torch.arange(NC, device=XR.device).view(NC, 1).expand(NC, Nmax).reshape(-1)[flat_mask]
    xs_all = xs_batch[evt_ids]                         # [Ntot,3]

    feats = build_feats(x_all, xs_all)
    tau_p, tau_s = model(feats)                        # [Ntot,1] x2
    d = torch.linalg.norm(x_all - xs_all, dim=-1, keepdim=True) + 1e-6
    T_all = torch.cat([d * tau_p, d * tau_s], dim=-1)  # [Ntot,2]

    T_pred = torch.zeros((NC * Nmax, 2), dtype=XR.dtype, device=XR.device)
    T_pred[flat_mask] = T_all
    return T_pred.view(NC, Nmax, 2)

@torch.no_grad()
def forward_tps_batched(xs_batch: torch.Tensor, XR: torch.Tensor, exist_mask: torch.Tensor) -> torch.Tensor:
    """
    xs_batch:  [NC, 3]
    XR:        [NC, Nmax, 3]
    exist_mask:[NC, Nmax]
    return:    T_pred [NC, Nmax, 2]，无效位置置 0
    """
    NC, Nmax, _ = XR.shape
    flat_mask = exist_mask.view(-1)                    # [NC*Nmax]
    x_all = XR.view(-1, 3)[flat_mask]                  # [Ntot,3]

    evt_ids = torch.arange(NC, device=XR.device).view(NC, 1).expand(NC, Nmax).reshape(-1)[flat_mask]
    xs_all = xs_batch[evt_ids]                         # [Ntot,3]


    T_all = model(x_all, xs_all)                        # [Ntot,1] x2
    
    T_pred = torch.zeros((NC * Nmax, 2), dtype=XR.dtype, device=XR.device)
    T_pred[flat_mask] = T_all
    return T_pred.view(NC, Nmax, 2)

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
def sample_invgamma_batch(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    在 CUDA 或 CPU 上进行批量采样，采样 InvGamma(alpha_i, beta_i)。
    alpha, beta: shape=[NC] (batch)
    返回 shape=[NC] 的逆 Gamma 样本。
    """
    # 确保 alpha, beta 在同一个 device
    assert alpha.device == beta.device, "alpha 和 beta 必须在同一 device 上"

    # 创建 Gamma 分布（batch）
    gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=beta)

    # 采样：sample() 默认就是元素级别的 batch 采样
    g = gamma_dist.sample()

    # 逆 Gamma = 1/Gamma
    invg = 1.0 / g

    return invg

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
    idx0 = torch.randint(0, N, (1,), device=device, generator=generator)
    mu = [X[idx0]]
    for _ in range(1, M):
        dist2 = torch.stack([torch.sum((X - m)**2, dim=1) for m in mu], dim=1).min(dim=1).values
        probs = (dist2 + 1e-12) / (dist2.sum() + 1e-12)
        idx = torch.multinomial(probs, 1, generator=generator)
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
    comp = torch.multinomial(w, 1, generator=generator).item()
    z = torch.randn((3,1), device=device, dtype=dtype_, generator=generator)  # (3,1)
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
    comp = torch.multinomial(w, NC, replacement=True, generator=generator)

    # 2) 从标准正态生成 z：shape = (NC, 3, 1)
    z = torch.randn((NC, 3, 1), device=device, dtype=dtype_, generator=generator)

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

    r0 = torch.randn_like(xs, generator=generator)
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
def forward_tps_multi(xs_batch: torch.Tensor,
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
    feats = build_feats(xr, xs_per_obs)        # [N,13]
    tau_p, tau_s = model(feats)                # [N,1], [N,1]
    d = torch.linalg.norm(xr - xs_per_obs, dim=-1, keepdim=True) + 1e-6
    T_p = d * tau_p
    T_s = d * tau_s
    return torch.cat([T_p, T_s], dim=-1)       # [N,2]
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
      ll_e: [NC] 每事件log似然
      Np_e, Ns_e: [NC] P/S数
      sumP_e, sumS_e: [NC] 残差求和（用于 t0 条件更新）
      rssP_e, rssS_e: [NC] 残差平方和（用于 sigma 条件更新）
    """
    tp_model = T_pred[:, 0]
    ts_model = T_pred[:, 1]

    # P 残差与统计
    resP = Tp_obs - tp_model - t0[event_ids]
    Np_e   = _scatter_counts(maskP, event_ids, NC)                                  # [NC]
    sumP_e = _scatter_sum(resP, maskP, event_ids, NC)                               # [NC]
    rssP_e = _scatter_sum_sq(resP, maskP, event_ids, NC)                            # [NC]

    # S 残差与统计
    resS = Ts_obs - ts_model - t0[event_ids]
    Ns_e   = _scatter_counts(maskS, event_ids, NC)
    sumS_e = _scatter_sum(resS, maskS, event_ids, NC)
    rssS_e = _scatter_sum_sq(resS, maskS, event_ids, NC)

    # 逐事件 log-lik
    # 注意：当 Np/Ns=0 时，对应项为 0
    llP_e = torch.zeros(NC, device=Tp_obs.device, dtype=torch.float32)
    llS_e = torch.zeros(NC, device=Tp_obs.device, dtype=torch.float32)

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

# ===========================
# 批处理版采样器（RW / indep_mix；NUTS 仍逐事件）
# ===========================
@torch.no_grad()
def gibbs_mh_location_multi(
    Tp_obs: torch.Tensor, Ts_obs: torch.Tensor,   # [N]
    xr: torch.Tensor,                             # [N,3]
    event_ids: torch.Tensor,                      # [N] long in [0..NC-1]
    NC: int,
    n_samples: int = 4000, burn: int = 1000, thin: int = 1,
    xs_init: Optional[torch.Tensor] = None,       # [NC,3] 或 None
    t0_init: float = 0.0,
    alpha0: float = 1e-2, beta0: float = 1e-2,
    sigmaP_init: float = 0.1, sigmaS_init: float = 0.1,
    prop_scale: float = 0.5,
    *,
    device_: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
    generator: Optional[torch.Generator] = None,
    # 先验
    gmm_prior: Optional[Dict[str, torch.Tensor]] = None,
    prior_mix: Optional[float] = None,
    wide_sigma: float = 1000.0,
    # 提议/自适应
    xs_update: str = "rw",     # "rw" | "indep_mix"
    p_indep: float = 0.2,
    adapt_steps: int = 1000,
    target_accept_rw: float = 0.3,
    adapt_eta: float = 0.05,
) -> Dict[str, torch.Tensor]:
    assert xs_update in ("rw", "indep_mix")
    device_eff = device_ if device_ is not None else device
    if generator is None:
        generator = torch.Generator(device=device_eff)
    print(f"{xs_update=}")
    # 统一形状/类型
    Tp_obs = torch.as_tensor(Tp_obs, dtype=dtype, device=device_eff).view(-1) # 多个地震的P 
    Ts_obs = torch.as_tensor(Ts_obs, dtype=dtype, device=device_eff).view(-1) # 多个地震的S 
    xr     = torch.as_tensor(xr,     dtype=dtype, device=device_eff)          # 第一个的台站分布，跟TpTs长度相同
    event_ids = torch.as_tensor(event_ids, dtype=torch.long, device=device_eff).view(-1)
    
    N = Tp_obs.numel()# 总共多少个观测
    assert Ts_obs.shape == (N,) and xr.shape == (N,3), "输入维度不匹配"
    assert event_ids.shape == (N,) and int(event_ids.max().item()) < NC

    # 缺测掩码（按观测）
    maskP, maskS, _, _ = build_phase_masks(Tp_obs, Ts_obs, missing_value=MISSING_VAL)

    # 初始化参数（按事件）
    if xs_init is None:
        xs = xr.mean(dim=0, keepdim=True).expand(NC, 3).contiguous()
        xs = xs + torch.tensor([0.1, 0.1, 10.0], device=device_eff, dtype=dtype).view(1,3)
    else:
        xs = torch.as_tensor(xs_init, dtype=dtype, device=device_eff).view(NC,3)

    t0      = torch.full((NC,), float(t0_init), dtype=dtype, device=device_eff)
    sigmaP2 = torch.full((NC,), float(sigmaP_init**2), dtype=dtype, device=device_eff)
    sigmaS2 = torch.full((NC,), float(sigmaS_init**2), dtype=dtype, device=device_eff)

    # 样本缓存
    M = (n_samples - burn) // max(1, thin)
    xs_samples     = torch.empty((M, NC, 3), dtype=dtype, device=device_eff)
    t0_samples     = torch.empty((M, NC),     dtype=dtype, device=device_eff)
    sigmaP_samples = torch.empty((M, NC),     dtype=dtype, device=device_eff)
    sigmaS_samples = torch.empty((M, NC),     dtype=dtype, device=device_eff)

    log2pi = torch.tensor(math.log(2.0 * math.pi), dtype=dtype, device=device_eff)
    print(f"getting prior...")
    # 先验闭包
    log_prior_xs = make_log_prior_xs(
        gmm_prior=gmm_prior, wide_sigma=wide_sigma, prior_mix=prior_mix,
        device=device_eff, dtype=dtype
    )

    # 当前前向（一次）
    T_pred_curr = forward_tps_multi(xs, xr, event_ids)  # [N,2]
    # 当前逐事件 logpost
    ll_e, Np_e, Ns_e, sumP_e, sumS_e, rssP_e, rssS_e = log_lik_per_event(
        T_pred_curr, Tp_obs, Ts_obs, t0, sigmaP2, sigmaS2,
        log2pi, maskP, maskS, event_ids, NC
    )
    curr_logpost_e = log_prior_xs(xs) + ll_e            # [NC]
    curr_logpost_sum = curr_logpost_e.sum()

    # 自适应步长（逐事件）
    log_prop_scale = torch.full((NC,), math.log(max(prop_scale, 1e-6)),
                                dtype=dtype, device=device_eff)

    accept_count_e = torch.zeros(NC, dtype=torch.int32, device=device_eff)
    total_prop = 0
    sample_idx = 0
    print('Start sampling...')
    for k in range(n_samples):
        if verbose and (k % 50 == 0):
            print(f"[iter {k}]")

        # ------- 1) 逐事件并行提议 xs' -------
        total_prop += 1
        if xs_update == "indep_mix" and (gmm_prior is not None):
            use_indep = (torch.rand((NC,), device=device_eff) < p_indep)  # [NC] bool
        else:
            use_indep = torch.zeros((NC,), dtype=torch.bool, device=device_eff)

        # RW 提议
        steps = torch.randn((NC,3), dtype=dtype, device=device_eff, generator=generator) \
                * log_prop_scale.exp().unsqueeze(1)
        xs_prop = xs + steps

        # 独立提议（逐事件选择）
        if use_indep.any():
            cnt = use_indep.sum().item()
            xs_prob_ind = sample_gmm_batch(gmm_prior, cnt, generator=generator)
            if cnt > 0:
                xs_prop[use_indep] = xs_prob_ind

        # 单次前向（所有观测，提议的 xs_prop）
        T_pred_prop = forward_tps_multi(xs_prop, xr, event_ids)

        # 逐事件 logpost（提议）
        ll_prop_e, Np_e_prop, Ns_e_prop, sumP_e_prop, sumS_e_prop, rssP_e_prop, rssS_e_prop = log_lik_per_event(
            T_pred_prop, Tp_obs, Ts_obs, t0, sigmaP2, sigmaS2,
            log2pi, maskP, maskS, event_ids, NC
        )
        prop_logpost_e = log_prior_xs(xs_prop) + ll_prop_e  # [NC]

        # 提议分布修正（仅对独立提议的事件）
        log_q_ratio_e = torch.zeros(NC, dtype=dtype, device=device_eff)
        if gmm_prior is not None and use_indep.any():
            # q(x'|x)=p_gmm(x'), q(x|x')=p_gmm(x)
            log_q_ratio_e[use_indep] = gmm_logpdf(xs[use_indep], gmm_prior) - gmm_logpdf(xs_prop[use_indep], gmm_prior)

        log_alpha_e = (prop_logpost_e - curr_logpost_e) + log_q_ratio_e  # [NC]
        u = torch.rand((NC,), dtype=dtype, device=device_eff, generator=generator).log()
        accept_e = u < log_alpha_e    # [NC] bool

        # 接受后，替换对应事件的 xs 与 T_pred_curr（行替换）
        if accept_e.any():
            # 替换 xs / logpost
            xs[accept_e] = xs_prop[accept_e]
            curr_logpost_e[accept_e] = prop_logpost_e[accept_e]
            accept_count_e[accept_e] += 1

            # 将被接受事件对应观测行整体替换为 T_pred_prop
            accept_obs = accept_e[event_ids]                    # [N] bool
            T_pred_curr[accept_obs] = T_pred_prop[accept_obs]

        #print(xs.shape, xs_prop.shape)
        # 自适应 RW 步长（仅对 RW 的事件做）
        if k < adapt_steps:
            # 经验接受率（逐事件，等价于一次伯努利）
            acc_val = torch.where(log_alpha_e > 0, torch.ones_like(log_alpha_e), log_alpha_e.exp().clamp(max=1.0))
            # 对于使用独立提议的事件，不更新 RW 步长
            rw_mask = ~use_indep
            if rw_mask.any():
                log_prop_scale[rw_mask] += adapt_eta * (acc_val[rw_mask] - target_accept_rw)

        # sum(T_obs - T_model) = (sumP_e + sumS_e) + Np_e*t0 + Ns_e*t0 之前是减了 t0 的定义
        # 我们复用 log_lik_per_event 里构造的 sumP_e, sumS_e (基于当前 t0)，因此重新算一次以便一致
        ll_e, Np_e, Ns_e, sumP_e, sumS_e, _, _ = log_lik_per_event(
            T_pred_curr, Tp_obs, Ts_obs, t0, sigmaP2, sigmaS2,
            log2pi, maskP, maskS, event_ids, NC
        )

        # ------- 2) 条件更新 t0（向量化） -------
        # 需要新的逐事件统计（用当前 T_pred_curr 与 t0 之前的值，但 t0 更新用“未减 t0”的和）
        tp_model = T_pred_curr[:,0]
        ts_model = T_pred_curr[:,1]

        denom = torch.zeros(NC, dtype=dtype, device=device_eff)
        num   = torch.zeros(NC, dtype=dtype, device=device_eff)

        validP = Np_e > 0
        if validP.any():
            denom[validP] += Np_e[validP].to(dtype) / sigmaP2[validP]
            # sumP_e = sum(Tp_obs - tp_model - t0) → sum(Tp_obs - tp_model) = sumP_e + Np_e * t0
            num[validP]   += sumP_e[validP].to(dtype) / sigmaP2[validP]

        validS = Ns_e > 0
        if validS.any():
            denom[validS] += Ns_e[validS].to(dtype) / sigmaS2[validS]
            num[validS]   += sumS_e[validS].to(dtype) / sigmaS2[validS]

        # 若某事件 P/S 全缺失，保持原 t0（或可加先验）；这里简单跳过
        upd_mask = denom > 0
        if upd_mask.any():
            var_t0  = torch.zeros_like(denom)
            mean_t0 = torch.zeros_like(denom)
            var_t0[upd_mask]  = 1.0 / denom[upd_mask]
            mean_t0[upd_mask] = var_t0[upd_mask] * num[upd_mask]
            noise = torch.zeros_like(mean_t0)
            noise[upd_mask] = torch.randn((int(upd_mask.sum().item()),), dtype=dtype, device=device_eff, generator=generator)
            t0[upd_mask] = mean_t0[upd_mask] + noise[upd_mask] * var_t0[upd_mask].sqrt()

        # ------- 3) 条件更新 sigmaP2 / sigmaS2（向量化 + InvGamma 逐事件采样） -------
        # 重新计算基于新 t0 的残差平方和
        resP = Tp_obs - t0[event_ids] - tp_model
        resS = Ts_obs - t0[event_ids] - ts_model
        rssP_e = _scatter_sum_sq(resP, maskP, event_ids, NC)
        rssS_e = _scatter_sum_sq(resS, maskS, event_ids, NC)

        # 逐事件后验参数
        alphaP = torch.full((NC,), alpha0, dtype=dtype, device=device_eff)
        betaP  = torch.full((NC,), beta0,  dtype=dtype, device=device_eff)
        alphaS = torch.full((NC,), alpha0, dtype=dtype, device=device_eff)
        betaS  = torch.full((NC,), beta0,  dtype=dtype, device=device_eff)

        validP = Np_e > 0
        validS = Ns_e > 0
        if validP.any():
            alphaP[validP] += 0.5 * Np_e[validP].to(dtype)
            betaP[validP]  += 0.5 * rssP_e[validP].to(dtype)
        if validS.any():
            alphaS[validS] += 0.5 * Ns_e[validS].to(dtype)
            betaS[validS]  += 0.5 * rssS_e[validS].to(dtype)

        # InvGamma 采样（用 CPU gamma 回退），逐事件 loop（标量调用）
        #for e in range(NC):
        #    sigmaP2[e] = sample_invgamma(alphaP[e], betaP[e], dtype=dtype, device=device_eff)
        #    sigmaS2[e] = sample_invgamma(alphaS[e], betaS[e], dtype=dtype, device=device_eff)
        sigmaP2 = sample_invgamma_batch(alphaP, betaP)
        sigmaS2 = sample_invgamma_batch(alphaS, betaS)
        # ------- 4) 记录样本 -------
        
        if k >= burn and ((k - burn) % thin == 0):
            xs_samples[sample_idx]     = xs
            t0_samples[sample_idx]     = t0
            sigmaP_samples[sample_idx] = sigmaP2.sqrt()
            sigmaS_samples[sample_idx] = sigmaS2.sqrt()
            
            #xss = xs_samples.cpu().numpy()[sample_idx-1]
            #plt.scatter(xss[:,0], xss[:,1], c="r", s=5)
            #lt.savefig(f"tfig/steps/{k}.png")
            #plt.cla()
            #plt.clf()
            sample_idx += 1
        # （可选）更新 curr_logpost_sum —— 仅用于监控
        curr_logpost_sum = curr_logpost_e.sum()

    accept_rate_e = (accept_count_e.to(torch.float32) / max(1, total_prop))
    return {
        'xs_samples': xs_samples,         # [M, NC, 3]
        't0_samples': t0_samples,         # [M, NC]
        'sigmaP_samples': sigmaP_samples, # [M, NC]
        'sigmaS_samples': sigmaS_samples, # [M, NC]
        'accept_rate_per_event': accept_rate_e,  # [NC]
        'final_prop_scale_per_event': log_prop_scale.exp(),  # [NC]
    }


import datetime 
import numpy as np 
import pickle 
def read_station_file(file_path='ayrdata/china.loc'):
    stations = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            network = parts[0]
            station_code = parts[1]
            network_id = parts[2]
            lon = float(parts[3])
            lat = float(parts[4])
            ele = float(parts[5]) 
            stations[f"{network}{station_code}"] = np.array([lon, lat, -ele/1000])
    return stations 
def read_event_file(file_path):
    with open("data/vel.py.model", 'rb') as f:
        grid_vp, grid_vs, grid_x, grid_y, grid_z, proj, *rest = pickle.load(f)
    events = []
    stloc = read_station_file() 
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        # 读取事件头信息
        if lines[i].startswith("#"):
            header = lines[i][1:].strip().split()
            # 时间
            try:
                origin_time = datetime.datetime.strptime(" ".join(header[0:6]), "%Y %m %d %H %M %S.%f")
            except:
                origin_time = datetime.datetime.strptime(" ".join(header[0:6]).replace("60", "59"), "%Y %m %d %H %M %S.%f")
            lat = float(header[6])
            lon = float(header[7])
            depth = float(header[8])
            mag1 = float(header[9])
            mag2 = float(header[10])
            # 后面的几个是未知或保留字段
            event_info = {
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
            while i < len(lines) and not lines[i].startswith("#"):
                parts = lines[i].split()
                station = parts[0]
                travel_time = float(parts[1])
                weight = float(parts[2])
                phase = parts[3]
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
        x, y = proj(e["longitude"], e["latitude"]) 
        z = e["depth"]
        all_events.append([x, y, z])
        #events_loc.append(e)
        stations = []
        for p in e["phases"]:
            stations.append(p["station"])

        set_stations = list(set(stations))
        st2id = {}
        for i, station in enumerate(set_stations):
            st2id[station] = i
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
            if p['station'] not in stloc:
                continue
            loc = stloc[p['station']]
            x, y = proj(loc[0], loc[1])
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
            "etime":e["origin_time"], 
            "locinfo":locinfo, 
            "mag1":e["mag1"],
            "mag2":e["mag2"],
        })
    all_events = np.array(all_events).astype(np.float32)
    all_events = torch.tensor(all_events, device=device).float()
    return loc_events, all_events, proj 
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
    events, ctlg, proj = read_event_file("data/Catalog_1.Pha")
    # 拟合 GMM 作为先验
    #gmm = fit_gmm_em_torch(
    #    ctlg, n_components=30, n_iters=100, reg_eps=5e-2,
    #    device=device, dtype=dtype
    #)

    # ====== 生成一批数据并演示缺测 ======
    
    rcvs = []
    c = 0 
    event_ids = []
    Tp_obs = []
    Ts_obs = []
    xr = []
    for eve, ctg in tqdm.tqdm(zip(events, ctlg)):
        #print(eve)
        etime = eve["etime"]
        locinfo = eve["locinfo"]
        rcv = locinfo["rcv"] 
        T_p = locinfo["T_p"]
        T_s = locinfo["T_s"] 
        event_id = torch.ones([len(rcv)], device=device)
        event_ids.append(event_id * c)
        c += 1
        #print("src/rcv/T shapes:", rcv.shape, T_p.shape, T_s.shape)
        t1 = time.perf_counter()
        rcvs.append({"xr":rcv, "Tp":T_p, "Ts":T_s})
        xr.append(rcv)
        Tp_obs.append(T_p)
        Ts_obs.append(T_s)
    event_ids = torch.cat(event_ids) 
    Tp_obs = torch.cat(Tp_obs) 
    Ts_obs = torch.cat(Ts_obs) 
    xr = torch.cat(xr) 
    t1 = time.perf_counter()
    out = gibbs_mh_location_multi(
        Tp_obs, Ts_obs, xr,
        event_ids=event_ids, NC=c,
        n_samples=4000, burn=2000, thin=2,
        xs_update="rw", p_indep=0.2,
        gmm_prior=None, prior_mix=0.5,  # 或仅用宽高斯
        prop_scale=1.0,
        device_=device, dtype=dtype, verbose=True, 
    )
    t2 = time.perf_counter() 
    print("Gibbs MH 耗时:", t2-t1)
    print("Gibbs MH 耗时(per sample):", (t2-t1)/len(rcvs))
    xs_samples = out["xs_samples"]#.cpu().numpy()
    t0_samples = out["t0_samples"]#.cpu().numpy()
    ofile = open("odata/reloc.raw.v2.1.txt", "w")
    print(len(events), len(ctlg), xs_samples.shape, t0_samples.shape)
    xs_samples = xs_samples.float()
    t0_samples = t0_samples.float()
    for idx, (eve, ctg) in enumerate(zip(events, ctlg)):
        xs_samps, t0_samps = xs_samples[:, idx, :], t0_samples[:, idx]
        xs_mean = xs_samps.mean(dim=0)
        xs_std  = xs_samps.std(dim=0, unbiased=True)
        t0_mean = t0_samps.mean()
        etime = eve["etime"]
        locinfo = eve["locinfo"]
        #print("True xs:", src.detach().cpu().numpy())
        t0_mean = t0_mean.flatten().detach().cpu().numpy() 
        xs_std = xs_std.flatten().detach().cpu().numpy()
        #print("Posterior mean xs:", xs_mean.detach().cpu().numpy())
        #print("Posterior std  xs:", xs_std.detach().cpu().numpy())
        tstr = (etime+datetime.timedelta(seconds=float(t0_mean[0]))).strftime("%Y-%m-%d %H:%M:%S")
        x, y, z = xs_mean.detach().cpu().numpy()
        lon, lat = proj(x, y, inverse=True)
        ofile.write(f"#EVENT,{tstr},{lon},{lat},{z},{x},{y},{z},{xs_std[0]},{xs_std[1]},{xs_std[2]}\n")
        ctg = ctg.cpu().numpy()
        x, y = proj(ctg[0], ctg[1], inverse=True)
        z = ctg[2]
        ofile.write(f"{ctg[0]},{ctg[1]},{ctg[2]},{x},{y},{z},{eve['mag1']},{eve['mag2']}\n")
        # 置信区间
        q = torch.tensor([0.025, 0.975], device=xs_samps.device, dtype=xs_samps.dtype)
        for i, name in enumerate(['x', 'y', 'z']):
            lo, hi = torch.quantile(xs_samps[:, i], q).tolist()
            ofile.write(f" {name}: [{lo:.3f}, {hi:.3f}], range: {(hi - lo):.3f}\n")
        ofile.flush()
        #xs_samples=xs_samples,       # [NC, M, 3]
        #t0_samples=t0_samples,       # [NC, M]
        #sigmaP_samples=sigmaP_samples,
        #sigmaS_samples=sigmaS_samples,
        #accept_rate=accept_rate,     # [NC]
        #final_prop_scale=log_prop_scale.exp(),  # [NC]
