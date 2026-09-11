from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from nllgrid import NLLGrid


@dataclass
class NLLGridFromMetaConfig:
    # Inputs: produced by your projection/resample step
    meta_json: str = "run_fm3d/data/xyz_vp_vs_meta.json"

    # Output basenames (without extension). NLLGrid will append .P.mod / .S.mod
    out_basename_p: str = "nlloc_script/test_file_v2/slow_P_cubic"
    out_basename_s: str = "nlloc_script/test_file_v2/slow_S_cubic"

    # Numerical safety
    eps_vel: float = 1e-6

    # If True, also write a quick npz dump of slow grids for debugging
    dump_npz: bool = False


def load_meta(meta_json: str) -> dict:
    p = Path(meta_json)
    meta = json.loads(p.read_text(encoding="utf-8"))
    return meta


def load_axes_and_values(meta: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x_km (Nx,), y_km (Ny,), z_km (Nz,), VP (Nz,Ny,Nx), VS (Nz,Ny,Nx)
    """
    axes_path = Path(meta["axes_file"])
    npy_path = Path(meta["npy_file"])

    if not axes_path.exists():
        raise FileNotFoundError(f"axes_file not found: {axes_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"npy_file not found: {npy_path}")

    axes = np.load(axes_path)
    x_km = axes["x_km"].astype(np.float64)
    y_km = axes["y_km"].astype(np.float64)
    z_km = axes["z_km"].astype(np.float64)

    arr = np.load(npy_path)  # (N,5) [x,y,z,vp,vs]
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"Bad npy array shape: {arr.shape}, expect (N,5)")

    Nz, Ny, Nx = map(int, meta["shape"])
    if (Nz, Ny, Nx) != (len(z_km), len(y_km), len(x_km)):
        raise ValueError(
            f"shape mismatch: meta {meta['shape']} vs axes {(len(z_km),len(y_km),len(x_km))}"
        )
    if arr.shape[0] != Nz * Ny * Nx:
        raise ValueError(f"n_points mismatch: arr has {arr.shape[0]} rows, expect {Nz*Ny*Nx}")

    # IMPORTANT:
    # Your writer used order: for z in z_km: for y in y_km: for x in x_km: write row
    # => vp/vs can be reshaped directly to (Nz,Ny,Nx)
    VP = arr[:, 3].reshape(Nz, Ny, Nx).astype(np.float32)
    VS = arr[:, 4].reshape(Nz, Ny, Nx).astype(np.float32)

    return x_km, y_km, z_km, VP, VS


def velocity_to_slow_len(vel_km_s: np.ndarray, grid_step_km: float, eps_vel: float) -> np.ndarray:
    vel_safe = np.maximum(vel_km_s, eps_vel)
    slow_len = (1.0 / vel_safe) * grid_step_km
    return slow_len.astype(np.float32)


def write_nll_grid_slow_len(
    slow_len: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    z_km: np.ndarray,
    basename: str,
    wave_type: str,
) -> None:
    """
    slow_len shape: (Nz,Ny,Nx)  [depth, y, x]
    NLLGrid expects array indexing consistent with its internal convention.
    In practice, for NLLoc grids, we want (nx,ny,nz) with x fastest, then y, then z.
    So we transpose to (Nx,Ny,Nz) before assigning to grd.array.
    """
    Nz, Ny, Nx = slow_len.shape

    dx = float(x_km[1] - x_km[0]) if len(x_km) > 1 else 0.0
    dy = float(y_km[1] - y_km[0]) if len(y_km) > 1 else 0.0
    dz = float(z_km[1] - z_km[0]) if len(z_km) > 1 else 0.0

    # Enforce cubic (you can relax this if needed)
    if not (abs(dx - dy) < 1e-6 and abs(dx - dz) < 1e-6):
        raise ValueError(f"Grid spacing not cubic: dx={dx}, dy={dy}, dz={dz}")

    x_orig = float(x_km[0])
    y_orig = float(y_km[0])
    z_orig = float(z_km[0])

    grd = NLLGrid(
        nx=Nx, ny=Ny, nz=Nz,
        x_orig=x_orig, y_orig=y_orig, z_orig=z_orig,
        dx=dx, dy=dy, dz=dz,
    )

    # Transpose (Nz,Ny,Nx) -> (Nx,Ny,Nz)
    grd.array = np.transpose(slow_len, (2, 1, 0)).astype(np.float32)

    grd.type = "SLOW_LEN"
    grd.float_type = "FLOAT"
    grd.basename = basename + f".{wave_type.upper()}.mod"

    Path(grd.basename).parent.mkdir(parents=True, exist_ok=True)
    grd.write_hdr_file()
    grd.write_buf_file()


def main(cfg: NLLGridFromMetaConfig) -> None:
    meta = load_meta(cfg.meta_json)

    # Load already-uniform projected km grids
    x_km, y_km, z_km, VP, VS = load_axes_and_values(meta)

    # Step is constant 5 km (or whatever in meta)
    step = float(meta["spacing_km"])

    slow_P = velocity_to_slow_len(VP, grid_step_km=step, eps_vel=cfg.eps_vel)
    slow_S = velocity_to_slow_len(VS, grid_step_km=step, eps_vel=cfg.eps_vel)

    write_nll_grid_slow_len(slow_P, x_km, y_km, z_km, cfg.out_basename_p, wave_type="P")
    print(f"Wrote P: {cfg.out_basename_p}.P.mod.hdr/.buf")

    write_nll_grid_slow_len(slow_S, x_km, y_km, z_km, cfg.out_basename_s, wave_type="S")
    print(f"Wrote S: {cfg.out_basename_s}.S.mod.hdr/.buf")

    if cfg.dump_npz:
        out = Path(cfg.out_basename_p).with_suffix("").parent / "slow_len_debug.npz"
        np.savez(out, x_km=x_km, y_km=y_km, z_km=z_km, slow_P=slow_P, slow_S=slow_S)
        print(f"Saved debug npz: {out}")


if __name__ == "__main__":
    cfg = NLLGridFromMetaConfig(
        meta_json="run_fm3d/data/xyz_vp_vs_meta.json",
        out_basename_p="run_fm3d/data/slow_P_cubic",
        out_basename_s="run_fm3d/data/slow_S_cubic",
        dump_npz=False,
    )
    main(cfg)
