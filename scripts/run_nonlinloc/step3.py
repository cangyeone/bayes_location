#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Grid2TimeConfig:
    # 固定：所有文件都放这里；Grid2Time 也在这里运行
    work_dir: str = "run_fm3d/data"

    # ---------- Inputs（相对 work_dir） ----------
    station_file: str = "loc.synth_stations.new.txt"

    # 输入慢度网格 root（不带 .hdr/.buf）
    slow_p_root: str = "slow_P_cubic"
    slow_s_root: str = "slow_S_cubic"

    # ---------- Outputs（相对 work_dir） ----------
    out_dir: str = "time_synth_test"
    out_ps_root: str = "time_synth_test/tt_PS"   # 你外部定义的统一输出 root

    # 两个输入文件：各自只包含一条 GTFILES
    grid2time_p_in: str = "grid2time_P.in"
    grid2time_s_in: str = "grid2time_S.in"

    # ---------- NLLoc Grid2Time Common Controls ----------
    control_1: int = 1
    control_seed: int = 54321
    lat0: float = 27.5
    lon0: float = 102.5
    z0: float = 0.0

    gtmode: str = "GRID3D"
    angles: str = "ANGLES_NO"
    gt_plfd_1: float = 1.0e-3
    gt_plfd_2: int = 0

    # ---------- Station handling ----------
    z_srce_km: float = 0.0
    keep_elev_sign: bool = True

    # ---------- Execution ----------
    grid2time_bin: str = "/Users/yuziye/machinelearning/location/NonLinLoc/src/bin/Grid2Time"
    iswap: int = 0
    run_grid2time: bool = True   # True: 生成后顺序跑 P、S；False: 只生成 .in 文件

    # ---------- Safety checks ----------
    verify_outputs: bool = True


def _parse_station_lines(station_file: str) -> List[Tuple[str, str, float, float, float]]:
    stations = []
    p = Path(station_file)
    if not p.is_file():
        raise FileNotFoundError(f"Station file not found: {station_file}")

    for ln, raw in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            raise ValueError(f"Bad station line (need 5 columns): line {ln}: {raw}")
        net, sta = parts[0], parts[1]
        lon = float(parts[2])
        lat = float(parts[3])
        elev_km = float(parts[4])
        stations.append((net, sta, lon, lat, elev_km))

    if not stations:
        raise ValueError(f"No stations parsed from: {station_file}")
    return stations


def _format_grid2time_single_in(
    cfg: Grid2TimeConfig,
    slow_root: str,
    out_ps_root: str,
    wave_type: str,
    stations: List[Tuple[str, str, float, float, float]],
) -> str:
    """
    One Grid2Time .in containing ONE GTFILES line only (P or S).
    """
    if wave_type not in ("P", "S"):
        raise ValueError(f"wave_type must be 'P' or 'S', got: {wave_type}")

    lines = []
    lines.append("# ========= 通用控制参数 =========")
    lines.append(f"CONTROL {cfg.control_1} {cfg.control_seed}")
    lines.append(f"TRANS SIMPLE {cfg.lat0:.6f} {cfg.lon0:.6f} {cfg.z0:.1f}")
    lines.append("")
    lines.append("# GTFILES inputFileRoot outputFileRoot waveType iSwapBytesOnInput")
    lines.append(f"GTFILES {slow_root} {out_ps_root} {wave_type} {cfg.iswap}")
    lines.append("")
    lines.append(f"GTMODE {cfg.gtmode} {cfg.angles}")
    lines.append(f"GT_PLFD {cfg.gt_plfd_1:.1e} {cfg.gt_plfd_2}")
    lines.append("")
    lines.append("# ========= 台站列表 =========")
    lines.append("# GTSRCE label LATLON  lat  lon  z(km, 正向向下)  elev(km)")
    lines.append("")

    for net, sta, lon, lat, elev_km in stations:
        label = f"{net}_{sta}"  # 不能有空格
        elev_out = elev_km if cfg.keep_elev_sign else (-elev_km)
        lines.append(
            f"GTSRCE {label:<10s} LATLON  {lat:8.4f} {lon:9.4f}  {cfg.z_srce_km:4.1f}  {elev_out:7.3f}"
        )

    lines.append("")
    lines.append("END")
    lines.append("")
    return "\n".join(lines)


def _run_grid2time_in_place(grid2time_bin: str, in_dir: Path, in_name: str) -> str:
    cmd = [grid2time_bin, in_name]
    print("[RUN]", " ".join(cmd), f"(cwd={in_dir})")
    r = subprocess.run(cmd, cwd=str(in_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"Grid2Time failed with code {r.returncode}: {' '.join(cmd)} (cwd={in_dir})")
    return r.stdout


def _verify_has_phase(time_dir: Path, out_ps_root_rel: str, phase: str) -> None:
    """
    Hard check for outputs under the unified out root.
    We expect filenames like: tt_PS.P.mod.* or tt_PS.S.mod.* (common pattern).
    """
    root_name = Path(out_ps_root_rel).name  # tt_PS
    if phase == "P":
        pat_hdr = f"{root_name}.P.mod.hdr"
        pat_buf = f"{root_name}.P.mod.buf"
    elif phase == "S":
        pat_hdr = f"{root_name}.S.mod.hdr"
        pat_buf = f"{root_name}.S.mod.buf"
    else:
        raise ValueError("phase must be P or S")

    hdr = time_dir / pat_hdr
    buf = time_dir / pat_buf
    if hdr.exists() and buf.exists():
        print(f"[OK] Found {phase} model grids: {hdr.name}, {buf.name}")
        return

    # fallback: station files
    cand = sorted([p.name for p in time_dir.glob(f"{root_name}.{phase}.*.time.hdr")])
    if cand:
        print(f"[OK] Found {phase} station grids (examples): {', '.join(cand[:5])}")
        return

    raise RuntimeError(f"[FAIL] No {phase} outputs detected under {time_dir} for root '{root_name}'")


def main():
    cfg = Grid2TimeConfig()

    work_dir = Path(cfg.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    station_path = (work_dir / cfg.station_file).resolve()
    out_dir_path = (work_dir / cfg.out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    p_in_path = (work_dir / cfg.grid2time_p_in).resolve()
    s_in_path = (work_dir / cfg.grid2time_s_in).resolve()

    # CLI override station file (optional)
    if len(sys.argv) >= 2:
        station_path = Path(sys.argv[1]).expanduser()
        if not station_path.is_absolute():
            station_path = (Path.cwd() / station_path).resolve()

    stations = _parse_station_lines(str(station_path))

    # Use relative paths inside .in (Grid2Time runs in work_dir)
    slow_p_rel = os.path.relpath(str((work_dir / cfg.slow_p_root).resolve()), str(work_dir))
    slow_s_rel = os.path.relpath(str((work_dir / cfg.slow_s_root).resolve()), str(work_dir))
    out_ps_rel = os.path.relpath(str((work_dir / cfg.out_ps_root).resolve()), str(work_dir))

    # Write P.in
    txt_p = _format_grid2time_single_in(
        cfg=cfg,
        slow_root=slow_p_rel,
        out_ps_root=out_ps_rel,
        wave_type="P",
        stations=stations,
    )
    p_in_path.write_text(txt_p, encoding="utf-8")

    # Write S.in
    txt_s = _format_grid2time_single_in(
        cfg=cfg,
        slow_root=slow_s_rel,
        out_ps_root=out_ps_rel,
        wave_type="S",
        stations=stations,
    )
    s_in_path.write_text(txt_s, encoding="utf-8")

    print(f"[OK] wrote: {p_in_path}")
    print(f"[OK] wrote: {s_in_path}")
    print(f"[OK] stations: {len(stations)}")
    print(f"[OK] output dir: {out_dir_path}")
    print(f"[OK] unified tt root: {(work_dir / cfg.out_ps_root).resolve()}")

    if not cfg.run_grid2time:
        return

    # IMPORTANT: run P then S as two separate Grid2Time runs
    _run_grid2time_in_place(cfg.grid2time_bin, work_dir, p_in_path.name)
    _run_grid2time_in_place(cfg.grid2time_bin, work_dir, s_in_path.name)

    if cfg.verify_outputs:
        _verify_has_phase(out_dir_path, out_ps_rel, "P")
        _verify_has_phase(out_dir_path, out_ps_rel, "S")

    print("[DONE] Grid2Time finished (P and S).")


if __name__ == "__main__":
    main()
