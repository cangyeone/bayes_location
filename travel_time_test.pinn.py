#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pyproj import CRS, Transformer


# -------------------------
# Model must match training
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
        )

    def forward(self, x, xs):
        # x : (B,3) receiver [xr,yr,zr]
        # xs: (B,3) source   [xe,ye,ze]
        inp = torch.cat([x, xs.expand_as(x)], dim=-1)  # (B,6)
        inp = inp / 1000.0
        out = self.net_merge(inp)
        out = out.sigmoid() * 1000.0
        return out


# -------------------------
# Checkpoint IO
# -------------------------
def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    meta = {"args": ckpt.get("args", {}), "meta": ckpt.get("meta", {}), "epoch": ckpt.get("epoch", 0)}
    return meta


# -------------------------
# Parsing helpers (same as your training)
# -------------------------
_INT_PREFIX_RE = re.compile(r"^\s*[+-]?\d+(?![\d.])")

def starts_with_int(line: str) -> bool:
    return _INT_PREFIX_RE.match(line) is not None

def build_aeqd_fwd(lon0: float, lat0: float) -> Transformer:
    crs_aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(CRS.from_epsg(4326), crs_aeqd, always_xy=True)

@dataclass
class EventBlock:
    evid: int
    ev_lat: float
    ev_lon: float
    ev_dep_km: float
    picks: Dict[str, Dict[str, float]]

def _parse_event_header(line: str) -> Optional[Tuple[int, float, float, float]]:
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

def blocks_to_pairs(
    blocks: List[EventBlock],
    meta_json: str,
    station_z_km: float = 0.0,
    augment_swap: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    lon0 = float(meta["lon0"])
    lat0 = float(meta["lat0"])
    fwd = build_aeqd_fwd(lon0, lat0)

    Xr_list, Xs_list = [], []

    for ev in blocks:
        ex_m, ey_m = fwd.transform(ev.ev_lon, ev.ev_lat)
        xs = np.array([ex_m / 1000.0, ey_m / 1000.0, float(ev.ev_dep_km)], dtype=np.float32)

        for _, pk in ev.picks.items():
            sx_m, sy_m = fwd.transform(pk["lon"], pk["lat"])
            xr = np.array([sx_m / 1000.0, sy_m / 1000.0, float(station_z_km)], dtype=np.float32)

            Xr_list.append(xr)
            Xs_list.append(xs)
            if augment_swap:
                Xr_list.append(xs)
                Xs_list.append(xr)

    Xr = np.stack(Xr_list, axis=0).astype(np.float32)
    Xs = np.stack(Xs_list, axis=0).astype(np.float32)
    return Xr, Xs


# -------------------------
# Eikonal residual evaluation
# -------------------------
@torch.no_grad()
def predict_times(model: nn.Module, xr: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    return model(xr, xs)

def eikonal_residual_batch(
    model: nn.Module,
    xr: torch.Tensor,   # (B,3) requires_grad=True
    xs: torch.Tensor,   # (B,3) no grad needed
    vp: float,
    vs: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      gp, gs: (B,) norm of grad wrt xr for P,S (s/km)
      rp, rs: (B,) residuals gp-1/vp, gs-1/vs (s/km)
    """
    # IMPORTANT: need grad graph
    t = model(xr, xs)  # (B,2)

    Tp_sum = t[:, 0].sum()
    Ts_sum = t[:, 1].sum()

    grad_Tp = torch.autograd.grad(Tp_sum, xr, create_graph=False, retain_graph=True)[0]  # (B,3)
    grad_Ts = torch.autograd.grad(Ts_sum, xr, create_graph=False, retain_graph=False)[0]

    gp = torch.sqrt((grad_Tp ** 2).sum(dim=-1) + eps)
    gs = torch.sqrt((grad_Ts ** 2).sum(dim=-1) + eps)

    sp = 1.0 / float(vp)
    ss = 1.0 / float(vs)

    rp = gp - sp
    rs = gs - ss
    return gp, gs, rp, rs

def summarize_residual(name: str, r: np.ndarray, tol: float):
    r_abs = np.abs(r)
    stats = {
        "mean": float(r.mean()),
        "median": float(np.median(r)),
        "rmse": float(np.sqrt(np.mean(r**2))),
        "p90_abs": float(np.percentile(r_abs, 90)),
        "p95_abs": float(np.percentile(r_abs, 95)),
        "p99_abs": float(np.percentile(r_abs, 99)),
        "pass_rate(|r|<=tol)": float(np.mean(r_abs <= tol)),
    }
    print(f"\n[{name}] residual unit: s/km, tol={tol:g} s/km")
    for k, v in stats.items():
        print(f"  {k:>18s}: {v:.6g}")

def binned_by_distance(dist_km: np.ndarray, r: np.ndarray, nbins: int = 20):
    lo, hi = float(dist_km.min()), float(dist_km.max())
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = []
    for i in range(nbins):
        m = (dist_km >= edges[i]) & (dist_km < edges[i+1])
        if m.sum() < 10:
            out.append((centers[i], np.nan, np.nan, int(m.sum())))
            continue
        ri = r[m]
        out.append((centers[i], float(np.mean(np.abs(ri))), float(np.sqrt(np.mean(ri**2))), int(m.sum())))
    return np.array(out, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_txt", default="run_fm3d/data/test.synth_arrivals_skfmm_noise.new.txt", help="real.txt path")
    ap.add_argument("--meta_json", default="run_fm3d/data/xyz_vp_vs_meta.json", help="meta.json path (contains lon0/lat0)")
    ap.add_argument("--ckpt", default="ckpt/time.v3.1.pt")

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--t_scale", type=float, default=200.0)

    ap.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--station_z_km", type=float, default=0.0)
    ap.add_argument("--augment_swap", action="store_true")

    ap.add_argument("--vp", type=float, default=6.0)
    ap.add_argument("--vs", type=float, default=3.5)

    ap.add_argument("--nsample", type=int, default=20000, help="how many (xr,xs) pairs to sample for evaluation")
    ap.add_argument("--batch_size", type=int, default=512)

    ap.add_argument("--tol", type=float, default=0.02, help="pass threshold for |residual| in s/km")
    ap.add_argument("--check_source_grad", action="store_true",
                    help="also compute residuals wrt source gradient ∇_{xs}T (optional diagnostic)")

    ap.add_argument("--bin_distance", action="store_true", help="print residual stats binned by distance")
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--save_npz", default="", help="optional path to save arrays (.npz)")
    args = ap.parse_args()

    # Load pairs
    blocks = parse_real_txt(args.real_txt)
    Xr, Xs = blocks_to_pairs(
        blocks, args.meta_json,
        station_z_km=args.station_z_km,
        augment_swap=args.augment_swap
    )
    N = Xr.shape[0]
    print(f"[OK] parsed pairs: {N} (augment_swap={args.augment_swap})")

    # Subsample
    ns = min(int(args.nsample), N)
    rng = np.random.default_rng(12345)
    idx = rng.choice(N, size=ns, replace=False) if ns < N else np.arange(N)
    Xr = Xr[idx]
    Xs = Xs[idx]

    # Distance
    dist_km = np.linalg.norm(Xr - Xs, axis=1).astype(np.float64)

    # Load model
    model = PINNTravelTime(hidden_dim=args.hidden_dim)
    meta = load_checkpoint(args.ckpt, model, map_location=args.device, strict=True)
    model.to(args.device)
    model.eval()
    print(f"[OK] loaded ckpt epoch={meta.get('epoch', 0)}")
    # (optional) show training args saved in ckpt
    # print(meta.get("args", {}))

    vp, vs = float(args.vp), float(args.vs)
    tol = float(args.tol)

    # Evaluate residuals in batches
    rp_all, rs_all = [], []
    gp_all, gs_all = [], []

    rpS_all, rsS_all = [], []  # source grad residuals, optional

    bs = int(args.batch_size)
    for i0 in range(0, ns, bs):
        i1 = min(ns, i0 + bs)
        xr = torch.from_numpy(Xr[i0:i1]).to(args.device).detach().clone().requires_grad_(True)
        xs = torch.from_numpy(Xs[i0:i1]).to(args.device).detach()

        gp, gs, rp, rs = eikonal_residual_batch(model, xr, xs, vp=vp, vs=vs)

        gp_all.append(gp.detach().cpu().numpy())
        gs_all.append(gs.detach().cpu().numpy())
        rp_all.append(rp.detach().cpu().numpy())
        rs_all.append(rs.detach().cpu().numpy())

        if args.check_source_grad:
            # compute wrt source by swapping roles: treat xs as "variable", xr fixed
            xs_var = xs.detach().clone().requires_grad_(True)
            xr_fix = xr.detach()
            gpp, gss, rpp, rss = eikonal_residual_batch(model, xs_var, xr_fix, vp=vp, vs=vs)
            rpS_all.append(rpp.detach().cpu().numpy())
            rsS_all.append(rss.detach().cpu().numpy())

    gp_all = np.concatenate(gp_all)
    gs_all = np.concatenate(gs_all)
    rp_all = np.concatenate(rp_all)
    rs_all = np.concatenate(rs_all)

    print("\n=== Eikonal check (wrt receiver gradient ∇_{xr}T) ===")
    print(f"Target slowness: 1/vp={1.0/vp:.6g} s/km, 1/vs={1.0/vs:.6g} s/km")
    summarize_residual("P", rp_all, tol=tol)
    summarize_residual("S", rs_all, tol=tol)

    # Additional diagnostics: norm stats
    print("\n[Norm stats] gp=||∇Tp||, gs=||∇Ts|| (s/km)")
    print(f"  gp mean={gp_all.mean():.6g}, median={np.median(gp_all):.6g}")
    print(f"  gs mean={gs_all.mean():.6g}, median={np.median(gs_all):.6g}")

    if args.check_source_grad:
        rpS_all = np.concatenate(rpS_all) if rpS_all else np.array([], dtype=np.float64)
        rsS_all = np.concatenate(rsS_all) if rsS_all else np.array([], dtype=np.float64)
        print("\n=== Optional: check wrt source gradient ∇_{xs}T (diagnostic) ===")
        summarize_residual("P_source", rpS_all, tol=tol)
        summarize_residual("S_source", rsS_all, tol=tol)

    if args.bin_distance:
        print("\n=== Residual binned by distance (center_km, mean_abs, rmse, count) ===")
        bp = binned_by_distance(dist_km, rp_all, nbins=args.nbins)
        bs_ = binned_by_distance(dist_km, rs_all, nbins=args.nbins)
        print("\n[P residual bins]")
        for c, mean_abs, rmse, cnt in bp:
            print(f"  {c:8.2f} km | mean|r|={mean_abs:10.6g} rmse={rmse:10.6g} n={int(cnt)}")
        print("\n[S residual bins]")
        for c, mean_abs, rmse, cnt in bs_:
            print(f"  {c:8.2f} km | mean|r|={mean_abs:10.6g} rmse={rmse:10.6g} n={int(cnt)}")

    if args.save_npz:
        out = {
            "dist_km": dist_km,
            "gp": gp_all,
            "gs": gs_all,
            "rp": rp_all,
            "rs": rs_all,
        }
        if args.check_source_grad and rpS_all.size > 0:
            out["rp_source"] = rpS_all
            out["rs_source"] = rsS_all
        np.savez(args.save_npz, **out)
        print(f"\n[OK] saved: {args.save_npz}")


if __name__ == "__main__":
    main()
