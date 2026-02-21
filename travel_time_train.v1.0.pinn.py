#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pinn_from_real.py  (Eikonal-PINN version; inputs unchanged: (xr, xs) only)

- Parse REAL-like file (real.txt)
- Build (receiver, source) -> (tp, ts) dataset with -1 for missing phases
- Train travel-time net T(xr, xs) with:
    loss = masked_data_mse + lambda_eik * eikonal_residual
  where Eikonal residual (constant velocities):
    ||∇_{xr} Tp|| = 1/vp ,  ||∇_{xr} Ts|| = 1/vs

- Save / Load checkpoints (model + optimizer + epoch + meta)

Notes:
- This enforces an isotropic constant-velocity Eikonal constraint.
- Inputs remain exactly as before: receiver xyz + source xyz (6 dims). No vv input.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyproj import CRS, Transformer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Model: same inputs (xr,xs) -> (tp,ts)
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
            nn.Linear(hidden_dim, 2),
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


# -------------------------
# Regex helpers
# -------------------------
_INT_PREFIX_RE = re.compile(r"^\s*[+-]?\d+(?![\d.])")

def starts_with_int(line: str) -> bool:
    return _INT_PREFIX_RE.match(line) is not None


# -------------------------
# Projection
# -------------------------
def build_aeqd_fwd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(CRS.from_epsg(4326), crs_aeqd, always_xy=True)  # lon,lat -> x,y (m)


# -------------------------
# Parsed structures
# -------------------------
@dataclass
class EventBlock:
    evid: int
    ev_lat: float
    ev_lon: float
    ev_dep_km: float
    picks: Dict[str, Dict[str, float]]  # station -> dict with lon,lat,P,S


def _parse_event_header(line: str) -> Optional[Tuple[int, float, float, float]]:
    """
    Event header example (your REAL-like line):
      783 2022 09 05 00:10:03.000 8430.000 0.0000 21.9211 99.7457 25.00 ...
    We parse:
      evid = parts[0]
      lat  = parts[7]
      lon  = parts[8]
      dep  = parts[9]
    """
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    try:
        evid = int(parts[0])
        lat = float(parts[7])
        lon = float(parts[8])
        dep = float(parts[9])
        return evid, lat, lon, dep
    except Exception:
        return None


def _parse_pick_line(line: str) -> Optional[Tuple[str, str, float, float, float]]:
    """
    Pick line example:
      SC L09971B02373  P  8461.1334 31.1334 ... 99.7068 23.7262
                             ^abs     ^t_obs
    We return:
      (sta, phase, t_obs, sta_lon, sta_lat)

    IMPORTANT:
    - We use parts[4] as the relative travel time (sec), consistent with your current code.
    - Station lon/lat are the last 2 tokens.
    """
    s = line.strip()
    if not s.startswith("SC "):
        return None
    parts = s.split()
    if len(parts) < 6:
        return None
    sta = parts[1]
    phase = parts[2].upper()
    if phase not in ("P", "S"):
        return None
    try:
        t_obs = float(parts[4])
        lon_s = float(parts[-2])
        lat_s = float(parts[-1])
    except Exception:
        return None
    return sta, phase, t_obs, lon_s, lat_s


def parse_real_txt(real_txt: str) -> List[EventBlock]:
    blocks: List[EventBlock] = []
    cur: Optional[EventBlock] = None

    with open(real_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            if starts_with_int(line):
                if cur is not None:
                    blocks.append(cur)

                hdr = _parse_event_header(line)
                if hdr is None:
                    cur = None
                    continue
                evid, lat, lon, dep = hdr
                cur = EventBlock(evid=evid, ev_lat=lat, ev_lon=lon, ev_dep_km=dep, picks={})
                continue

            pk = _parse_pick_line(line)
            if pk is None or cur is None:
                continue

            sta, phase, t_obs, lon_s, lat_s = pk
            d = cur.picks.get(sta, {"lon": lon_s, "lat": lat_s})
            d["lon"] = lon_s
            d["lat"] = lat_s
            d[phase] = t_obs
            cur.picks[sta] = d

    if cur is not None:
        blocks.append(cur)

    if not blocks:
        raise RuntimeError(f"No events parsed from {real_txt}. Check file format.")
    return blocks


# -------------------------
# Build samples
# -------------------------
def blocks_to_samples_old(
    blocks: List[EventBlock],
    meta_json: str,
    station_z_km: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Returns:
      Xr: (N,3) station xyz in km
      Xs: (N,3) event   xyz in km
      Y : (N,2) [tp, ts] in sec, -1 for missing
      proj_meta: dict with lon0/lat0 used for reproducibility
    """
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    lon0 = float(meta["lon0"])
    lat0 = float(meta["lat0"])
    fwd = build_aeqd_fwd(lon0, lat0)

    Xr_list, Xs_list, Y_list = [], [], []

    for ev in blocks:
        ex_m, ey_m = fwd.transform(ev.ev_lon, ev.ev_lat)
        ex_km, ey_km = ex_m / 1000.0, ey_m / 1000.0
        ez_km = float(ev.ev_dep_km)

        for _, pk in ev.picks.items():
            sx_m, sy_m = fwd.transform(pk["lon"], pk["lat"])
            sx_km, sy_km = sx_m / 1000.0, sy_m / 1000.0
            sz_km = float(station_z_km)

            tp = float(pk["P"]) if "P" in pk else -1.0
            ts = float(pk["S"]) if "S" in pk else -1.0

            Xr_list.append([sx_km, sy_km, sz_km])
            Xs_list.append([ex_km, ey_km, ez_km])
            Y_list.append([tp, ts])

    Xr = np.asarray(Xr_list, dtype=np.float32)
    Xs = np.asarray(Xs_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)

    if Xr.shape[0] == 0:
        raise RuntimeError("No samples built. Check parsing of pick lines.")

    proj_meta = dict(
        projection=str(meta.get("projection", "AEQD")),
        lon0=lon0,
        lat0=lat0,
        station_z_km=float(station_z_km),
    )
    return Xr, Xs, Y, proj_meta

def blocks_to_samples(
    blocks: List[EventBlock],
    meta_json: str,
    station_z_km: float = 0.0,
    augment_swap: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Returns:
      Xr: (N,3) station xyz in km
      Xs: (N,3) event   xyz in km
      Y : (N,2) [tp, ts] in sec, -1 for missing
      proj_meta: dict with lon0/lat0 used for reproducibility

    If augment_swap=True:
      append reciprocal samples (swap receiver/source): (Xr,Xs,Y) -> (Xs,Xr,Y)
    """
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    lon0 = float(meta["lon0"])
    lat0 = float(meta["lat0"])
    fwd = build_aeqd_fwd(lon0, lat0)

    Xr_list, Xs_list, Y_list = [], [], []

    for ev in blocks:
        ex_m, ey_m = fwd.transform(ev.ev_lon, ev.ev_lat)
        ex_km, ey_km = ex_m / 1000.0, ey_m / 1000.0
        ez_km = float(ev.ev_dep_km)

        for _, pk in ev.picks.items():
            sx_m, sy_m = fwd.transform(pk["lon"], pk["lat"])
            sx_km, sy_km = sx_m / 1000.0, sy_m / 1000.0
            sz_km = float(station_z_km)

            tp = float(pk["P"]) if "P" in pk else -1.0
            ts = float(pk["S"]) if "S" in pk else -1.0

            xr = [sx_km, sy_km, sz_km]
            xs = [ex_km, ey_km, ez_km]
            y  = [tp, ts]

            Xr_list.append(xr)
            Xs_list.append(xs)
            Y_list.append(y)

            # reciprocity augmentation: swap (xr,xs) -> (xs,xr), same labels
            if augment_swap:
                Xr_list.append(xs)
                Xs_list.append(xr)
                Y_list.append(y)

    Xr = np.asarray(Xr_list, dtype=np.float32)
    Xs = np.asarray(Xs_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)

    if Xr.shape[0] == 0:
        raise RuntimeError("No samples built. Check parsing of pick lines.")

    proj_meta = dict(
        projection=str(meta.get("projection", "AEQD")),
        lon0=lon0,
        lat0=lat0,
        station_z_km=float(station_z_km),
        augment_swap=bool(augment_swap),
    )
    return Xr, Xs, Y, proj_meta

# -------------------------
# Dataset + masked loss
# -------------------------
class TravelTimeDataset(Dataset):
    def __init__(self, Xr: np.ndarray, Xs: np.ndarray, Y: np.ndarray):
        assert Xr.shape[0] == Xs.shape[0] == Y.shape[0]
        self.Xr = torch.from_numpy(Xr)  # (N,3)
        self.Xs = torch.from_numpy(Xs)  # (N,3)
        self.Y = torch.from_numpy(Y)    # (N,2), -1 for missing

    def __len__(self):
        return self.Xr.shape[0]

    def __getitem__(self, idx: int):
        return self.Xr[idx], self.Xs[idx], self.Y[idx]


def masked_mse(pred: torch.Tensor, target: torch.Tensor, tmax: float = 2000.0) -> torch.Tensor:
    """
    target uses -1 for missing.
    """
    mask = (target >= 0.0) & (target <= tmax) & torch.isfinite(target) & torch.isfinite(pred)
    if mask.sum().item() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    diff2 = (pred - target) ** 2
    return diff2[mask].mean()


def eikonal_loss_wrt_receiver(
    t_pred: torch.Tensor,
    xr: torch.Tensor,
    vp: float,
    vs: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Constant-velocity Eikonal constraint using gradients w.r.t receiver location xr:
      ||∇_{xr} Tp|| = 1/vp
      ||∇_{xr} Ts|| = 1/vs

    t_pred: (B,2) (sec)
    xr: (B,3) (km), requires_grad=True
    """
    sp = 1.0 / float(vp)  # s/km
    ss = 1.0 / float(vs)  # s/km

    # Sum to get scalar outputs for autograd
    Tp_sum = t_pred[:, 0].sum()
    Ts_sum = t_pred[:, 1].sum()

    grad_Tp = torch.autograd.grad(Tp_sum, xr, create_graph=True, retain_graph=True)[0]  # (B,3)
    grad_Ts = torch.autograd.grad(Ts_sum, xr, create_graph=True, retain_graph=True)[0]  # (B,3)

    gp = torch.sqrt((grad_Tp ** 2).sum(dim=-1) + eps)
    gs = torch.sqrt((grad_Ts ** 2).sum(dim=-1) + eps)

    lp = (gp - sp) ** 2
    ls = (gs - ss) ** 2
    return lp.mean() + ls.mean()


# -------------------------
# Checkpoint IO
# -------------------------
def save_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    args_dict: Dict,
    extra_meta: Dict,
):
    ckpt = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "model_class": model.__class__.__name__,
        "model_hidden_dim": getattr(model, "hidden_dim", None),
        "args": args_dict,
        "meta": extra_meta,
    }
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Tuple[int, Dict]:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = int(ckpt.get("epoch", 0))
    meta = {"args": ckpt.get("args", {}), "meta": ckpt.get("meta", {})}
    return start_epoch, meta


# -------------------------
# Training loop
# -------------------------
def train(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    save_ckpt: str = "",
    save_every: int = 1,
    resume_ckpt: str = "",
    run_args: Dict = None,
    meta: Dict = None,
    vp: float = 6.0,
    vs: float = 3.5,
    lambda_eik: float = 0.05,
    eik_frac: float = 0.5,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    start_epoch = 0
    if resume_ckpt:
        start_epoch, resume_meta = load_checkpoint(resume_ckpt, model, opt, map_location=device, strict=True)
        print(f"[OK] resumed from {resume_ckpt} (epoch={start_epoch})")
        if meta is not None:
            meta = {**resume_meta.get("meta", {}), **meta}

    run_args = run_args or {}
    meta = meta or {}
    steps = 0

    for ep in range(start_epoch + 1, epochs + 1):
        model.train()
        losses = []
        losses_data = []
        losses_eik = []

        for xr, xs, y in loader:
            xr = xr.to(device)
            xs = xs.to(device)
            y = y.to(device)

            # 1) data prediction (no grad wrt xr needed here)
            pred = model(xr, xs)

            if not torch.isfinite(pred).all():
                bad = (~torch.isfinite(pred)).nonzero(as_tuple=False)
                print("[FATAL] pred has NaN/inf, first bad index:", bad[0].tolist())
                i, j = bad[0].tolist()
                print("pred[i] =", pred[i].detach().cpu().numpy())
                print("target[i] =", y[i].detach().cpu().numpy())
                print("xr[i] =", xr[i].detach().cpu().numpy())
                print("xs[i] =", xs[i].detach().cpu().numpy())
                raise RuntimeError("pred non-finite")

            loss_data = masked_mse(pred, y)

            # 2) Eikonal loss (needs xr requires_grad)
            if lambda_eik > 0.0 and eik_frac > 0.0:
                B = xr.shape[0]
                if eik_frac < 1.0:
                    m = max(1, int(B * float(eik_frac)))
                    idx = torch.randperm(B, device=device)[:m]
                    xr_e = xr[idx].detach().clone().requires_grad_(True)
                    xs_e = xs[idx].detach()
                else:
                    xr_e = xr.detach().clone().requires_grad_(True)
                    xs_e = xs.detach()

                pred_e = model(xr_e, xs_e)
                loss_eik = eikonal_loss_wrt_receiver(pred_e, xr_e, vp=vp, vs=vs)
            else:
                loss_eik = torch.zeros((), device=device, dtype=loss_data.dtype)

            loss = loss_data + float(lambda_eik) * loss_eik

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))
            losses_data.append(float(loss_data.detach().cpu()))
            losses_eik.append(float(loss_eik.detach().cpu()))
            steps += 1

            if steps % 1000 == 0:
                print(
                    f"[EPOCH {ep:03d}] step={steps} "
                    f"loss={np.mean(losses):.6f} data={np.mean(losses_data):.6f} eik={np.mean(losses_eik):.6f}"
                )

        print(
            f"[EPOCH {ep:03d}] "
            f"loss={np.mean(losses):.6f} data={np.mean(losses_data):.6f} eik={np.mean(losses_eik):.6f} "
            f"batches={len(losses)}"
        )

        if save_ckpt and (ep % max(1, save_every) == 0):
            save_checkpoint(
                save_ckpt,
                model=model,
                optimizer=opt,
                epoch=ep,
                args_dict=run_args,
                extra_meta=meta,
            )
            print(f"[OK] saved checkpoint: {save_ckpt}")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_txt", default="run_fm3d/data/train.synth_arrivals_skfmm_noise.new.txt", help="real.txt path")
    ap.add_argument("--meta_json", default="run_fm3d/data/xyz_vp_vs_meta.json", help="meta.json path (contains lon0/lat0)")
    ap.add_argument("--station_z_km", type=float, default=0.0)

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--t_scale", type=float, default=200.0, help="scale for positive travel-time head (sec)")

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)

    # Eikonal PINN args (constant velocity)
    ap.add_argument("--vp", type=float, default=6.0, help="constant vp (km/s) for Eikonal constraint")
    ap.add_argument("--vs", type=float, default=3.5, help="constant vs (km/s) for Eikonal constraint")
    ap.add_argument("--lambda_eik", type=float, default=100, help="weight for Eikonal residual")
    ap.add_argument("--eik_frac", type=float, default=0.5, help="fraction of each batch used for Eikonal (0~1)")

    ap.add_argument("--save_ckpt", default="ckpt/time.v1.0.eikonal.pt", help="checkpoint path to save")
    ap.add_argument("--save_every", type=int, default=1, help="save every N epochs")
    ap.add_argument("--resume", default="", help="checkpoint path to resume/load")
    ap.add_argument("--eval_only", action="store_true", help="load checkpoint and run sanity inference only")
    args = ap.parse_args()

    # Build dataset
    blocks = parse_real_txt(args.real_txt)
    print(f"[OK] parsed events: {len(blocks)}")

    Xr, Xs, Y, proj_meta = blocks_to_samples(blocks, args.meta_json, station_z_km=args.station_z_km)
    print(f"[OK] samples: {len(Y)}")

    miss_p = float(np.mean(Y[:, 0] < 0.0))
    miss_s = float(np.mean(Y[:, 1] < 0.0))
    print(f"[INFO] missing ratio: P={miss_p:.3f}, S={miss_s:.3f}")

    ds = TravelTimeDataset(Xr, Xs, Y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=False)

    device = "mps"  # keep your original setting; change to "cuda" or "cpu" as needed
    model = PINNTravelTime(hidden_dim=args.hidden_dim)

    # If eval_only: must have resume
    if args.eval_only:
        if not args.resume:
            raise RuntimeError("--eval_only requires --resume <ckpt_path>")
        _ = load_checkpoint(args.resume, model, optimizer=None, map_location=device, strict=True)
        model.to(device)
        model.eval()
        with torch.no_grad():
            xr, xs, y = next(iter(dl))
            pred = model(xr.to(device), xs.to(device)).cpu().numpy()
            print("[EVAL] pred[0:10] (tp, ts):\n", pred[:10])
            print("[EVAL] true[0:10] (tp, ts) (-1 means missing):\n", y.numpy()[:10])
        return

    # Train (supports resume)
    run_args = vars(args).copy()
    meta = {"projection_meta": proj_meta}

    train(
        model=model,
        loader=dl,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_ckpt=args.save_ckpt,
        save_every=args.save_every,
        resume_ckpt=args.resume,
        run_args=run_args,
        meta=meta,
        vp=args.vp,
        vs=args.vs,
        lambda_eik=args.lambda_eik,
        eik_frac=args.eik_frac,
    )

    # Final sanity
    model.eval()
    with torch.no_grad():
        xr, xs, y = next(iter(dl))
        pred = model(xr.to(device), xs.to(device)).cpu().numpy()
        print("[SANITY] pred[0:5] (tp, ts):\n", pred[:5])
        print("[SANITY] true[0:5] (tp, ts):\n", y.numpy()[:5])


if __name__ == "__main__":
    main()
