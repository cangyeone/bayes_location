#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_pinn_pred_scatter.py

随机生成震源-台站几何 (0~200 km 震中距)，加载 PINNTravelTime checkpoint，
绘制“震中距-走时”散点图（P/S 同一张图），用于查看模型预测是否合理。

说明：
- 这里的输入坐标单位按你训练脚本保持一致：Xr/Xs 是 km（forward 内部 /1000 后变成 Mm 量级）
- checkpoint 结构按你 train_pinn_from_real.py 的 save_checkpoint/load_checkpoint 设计：
  ckpt["model_state"] 里是 state_dict
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -------------------------
# Model (与你训练脚本保持一致)
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
            nn.Linear(hidden_dim, 4),
            nn.Softplus(), 
        )

    def forward(self, x, xs):
        # x : (B,3) receiver [xr,yr,zr]
        # xs: (B,3) source   [xe,ye,ze]
        inp = torch.cat([x, xs.expand_as(x)], dim=-1)  # (B,6)
        inp = inp / 1000.0
        out = self.net_merge(inp)
        out = out * 10.0
        return out


def load_checkpoint(ckpt_path: str, model: nn.Module, map_location: str = "cpu", strict: bool = True):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if "model_state" not in ckpt:
        raise RuntimeError(f"Checkpoint missing key 'model_state': {ckpt_path}")
    model.load_state_dict(ckpt["model_state"], strict=strict)
    return ckpt


# -------------------------
# 随机几何生成：固定震中距 r（0~rmax），随机方位角；深度随机
# -------------------------
def gen_random_geometry(
    n: int,
    rmax_km: float = 200.0,
    z_ev_min_km: float = 0.0,
    z_ev_max_km: float = 20.0,
    station_z_km: float = 0.0,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Xr: (N,3) station xyz (km)
      Xs: (N,3) event   xyz (km)
      r : (N,) epicentral distance in xy-plane (km)
    """
    rng = np.random.default_rng(seed)

    # station 固定在 (0,0,zs)
    Xr = np.zeros((n, 3), dtype=np.float32)
    Xr[:, 2] = station_z_km

    # event 在 xy 平面上与台站距离 r，方位角随机；深度随机
    r = rng.uniform(0.0, rmax_km, size=n).astype(np.float32)
    az = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)

    xe = r * np.cos(az)
    ye = r * np.sin(az)
    ze = rng.uniform(z_ev_min_km, z_ev_max_km, size=n).astype(np.float32)

    Xs = np.stack([xe, ye, ze], axis=1).astype(np.float32)
    return Xr, Xs, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/time.real.pnsn.switch.v1.0.pt", help="checkpoint path, e.g. run_fm3d/run_demo/data/pinn_tt.ckpt.pt")
    ap.add_argument("--hidden_dim", type=int, default=256, help="model hidden dim (must match ckpt)")
    ap.add_argument("--device", default="mps", help="cpu/cuda/mps")

    ap.add_argument("--n", type=int, default=50000, help="number of random samples")
    ap.add_argument("--rmax_km", type=float, default=200.0)
    ap.add_argument("--zmin_km", type=float, default=0.0)
    ap.add_argument("--zmax_km", type=float, default=20.0)
    ap.add_argument("--station_z_km", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--out", default="", help="optional figure output path (png/pdf)")
    args = ap.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("[WARN] MPS not available, fallback to CPU")
        device = "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU")
        device = "cpu"

    # 1) build model + load ckpt
    model = PINNTravelTime(hidden_dim=args.hidden_dim)
    ckpt = load_checkpoint(args.ckpt, model, map_location=device, strict=True)
    model.to(device).eval()

    # 如果 ckpt 里保存了 hidden_dim，可做一次提示校验
    ckpt_hd = ckpt.get("model_hidden_dim", None)
    if ckpt_hd is not None and int(ckpt_hd) != int(args.hidden_dim):
        print(f"[WARN] ckpt model_hidden_dim={ckpt_hd}, but you set --hidden_dim={args.hidden_dim}. "
              f"If mismatch, load may fail or predictions may be wrong.")

    # 2) random geometry
    Xr, Xs, r = gen_random_geometry(
        n=args.n,
        rmax_km=args.rmax_km,
        z_ev_min_km=args.zmin_km,
        z_ev_max_km=args.zmax_km,
        station_z_km=args.station_z_km,
        seed=args.seed,
    )
    Xs = np.random.uniform(-500, 500, [10000, 3])
    Xs[:, 2] = np.random.uniform(0, 20, [10000])

    Xr = np.random.uniform(-500, 500, [10000, 3])
    Xr[:, 2] = 0 

    r = np.linalg.norm(Xs[:, :2] - Xr[:, :2], axis=1)

    xr_t = torch.from_numpy(Xr).float()
    xs_t = torch.from_numpy(Xs).float()


    # 3) predict in batches
    with torch.no_grad():
        pr = model(xs_t.to(device), xr_t.to(device)).detach().cpu().numpy()
        preds = pr

    tp_pred = preds[:, 0]
    ts_pred = preds[:, 1]
    tpn_pred = preds[:, 2] 
    tsn_pred = preds[:, 3]

    # 4) plot scatter (P/S in one figure)
    plt.figure(figsize=(8, 5))
    plt.scatter(r, tp_pred, s=3, alpha=0.35, color="#ff0000", label="P (pred)")
    plt.scatter(r, ts_pred, s=3, alpha=0.35, color="#0000ff", label="S (pred)")
    plt.scatter(r, tpn_pred, s=3, alpha=0.35, color="#55c120", label="P (pred)")
    plt.scatter(r, tsn_pred, s=3, alpha=0.35, color="#847282", label="S (pred)")
    plt.xlabel("Epicentral distance (km)")
    plt.ylabel("Travel time (s)")
    plt.title(f"PINN Predicted Distance–TravelTime Scatter (0–{args.rmax_km:g} km)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=200)
        print(f"[OK] saved figure: {args.out}")
    else:
        plt.show()

    # 5) quick stats
    print("[INFO] pred tp range:", float(np.min(tp_pred)), float(np.max(tp_pred)))
    print("[INFO] pred ts range:", float(np.min(ts_pred)), float(np.max(ts_pred)))


if __name__ == "__main__":
    main()
