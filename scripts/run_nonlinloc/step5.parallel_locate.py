#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Optional


@dataclass
class ParallelNLLocConfig:
    out_dir: Path = Path("run_fm3d/data/nlloc_sc_crop")

    nlloc_bin: str = "/Users/yuziye/machinelearning/location/NonLinLoc/src/bin/NLLoc"                 # or full path
    nlloc_in_template: str = "nlloc.in"
    obs_file: str = "real.obs"
    stations_file: str = "sc.stations"
    tt_root: str = Path("run_fm3d/data/time_sc")        # travel-time grid root

    work_root: str = "work"                 # out_dir/work/<event_id>/
    out_root: str = "out"                   # out_dir/out/
    n_workers: int = 12

    keep_workdir: bool = True               # delete per-event work dirs on success if False


# -----------------------------
# Split OBS: exactly your format
# -----------------------------

# Optional sanity: allow EventID to be purely numeric (your example is numeric)
EVENT_ID_RE = re.compile(r"^\d{14,18}_\d{6,10}$")


def split_obs_nll_format(obs_text: str) -> List[Tuple[str, str]]:
    """
    Split an NLLOC_OBS file (one pick per line) into per-event blocks.

    Expected line style (tabs/spaces both ok):
      STA  ? ? ?  P 0  YYYYMMDD HHMM  SS.SSS  GAU  ...  >  -1 -1  EventID

    We group by the last token of each non-comment, non-empty line (EventID).
    Return list of (event_id, block_text), where block_text contains ONLY pick lines for that event.
    """

    lines = obs_text.splitlines(keepends=True)

    # Keep insertion order of first appearance
    order: List[str] = []
    buckets: Dict[str, List[str]] = {}

    for ln in lines:
        if not ln.strip():
            continue
        if ln.lstrip().startswith("#"):
            continue

        # Split by any whitespace to be robust to tabs/spaces mixture
        toks = ln.strip().split()
        if len(toks) < 2:
            continue

        ev_id = toks[-1]

        # Optional: validate EventID; if your EventID may be non-numeric, remove this check.
        # If it doesn't look like EventID, skip (or raise).
        #if not EVENT_ID_RE.match(ev_id):
        #    # If you prefer strict behavior, replace with: raise RuntimeError(...)
        #    continue

        if ev_id not in buckets:
            buckets[ev_id] = []
            order.append(ev_id)
        buckets[ev_id].append(ln if ln.endswith("\n") else ln + "\n")

    blocks: List[Tuple[str, str]] = []
    for ev_id in order:
        block = "".join(buckets[ev_id])
        if not block.endswith("\n"):
            block += "\n"
        blocks.append((ev_id, block))

    return blocks


# -----------------------------
# Patch LOCFILES: your template has 3 paths
# LOCFILES <obs> <stations> <outprefix>
# -----------------------------

def patch_locfiles_line(template_text: str,
                        obs_path: str,
                        tt_root: str,
                        out_root: str,
                        obs_type: str = "NLLOC_OBS",
                        event_id: str = "",
                        iswap: str | None = None) -> str:
    """
    Patch LOCFILES line for template style:

      LOCFILES <obs> NLLOC_OBS <tt_root> <out_root> [iSwapBytes]

    - Replaces obs, obsType, ttRoot, outRoot.
    - Preserves newline style and surrounding spacing as much as practical.
    - If template has 5th arg (iswap), keep it unless 'iswap' is provided.
    """

    # Capture:
    #   1: prefix "LOCFILES +"
    #   2: obs
    #   3: spaces
    #   4: obsType
    #   5: spaces
    #   6: ttRoot
    #   7: spaces
    #   8: outRoot
    #   9: optional (spaces + iswap)
    #   10: line end
    pat = re.compile(
        r"(^\s*LOCFILES\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+\S+)?(\s*$)",
        flags=re.IGNORECASE | re.MULTILINE
    )

    def repl(m: re.Match) -> str:
        # Keep the optional 5th arg (iswap) if present and caller didn't override
        opt = m.group(9) or ""
        if iswap is not None:
            # normalize to " <iswap>" (keep one leading space)
            opt = f" {iswap}"

        return (
            m.group(1)
            + obs_path + m.group(3)
            + obs_type + m.group(5)
            + tt_root + m.group(7)
            + out_root + f"/{event_id}"
            + opt
            + m.group(10)
        )

    new_text, n = pat.subn(repl, template_text, count=1)
    if n != 1:
        raise RuntimeError("Failed to patch LOCFILES line (not found or unexpected template LOCFILES format).")
    return new_text


# -----------------------------
# Run one event
# -----------------------------
def run_one_event(args) -> Dict:
    cfg, event_id, obs_block = args

    base_dir = cfg.out_dir.resolve()
    work_root = base_dir / cfg.work_root
    out_root = base_dir / cfg.out_root
    work_dir = work_root / event_id

    work_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    #print("OUTROOT", out_root)
    #return {}
    # Project root = .../(project)/  where nlloc_script/ lives.
    # Because your TTTYP uses "nlloc_script/..." relative paths.
    project_root = base_dir.parent.parent.resolve()
    #print("PROJECT ROOT", project_root)
    #return {}
    # Write per-event obs into work dir
    obs_path = (work_dir / f"{event_id}.obs").resolve()
    obs_path.write_text(obs_block, encoding="utf-8")
    
    tt_root = cfg.tt_root.resolve() 
    # Stations path (absolute, safer)
    sta_path = (base_dir / cfg.stations_file).resolve()
    if not sta_path.exists():
        return {
            "event_id": event_id, "ok": False, "returncode": None,
            "error": f"stations file missing: {sta_path}",
            "stdout": "", "stderr": "", "work_dir": str(work_dir)
        }

    # Load template and patch LOCFILES
    tpl_path = (base_dir / cfg.nlloc_in_template).resolve()
    tpl_text = tpl_path.read_text(encoding="utf-8", errors="ignore")

    # Output prefix: make it unique per event. Keep trailing underscore to mimic your sc_loc_ style.
    out_prefix = (out_root / f"{event_id}_").resolve().as_posix()

    patched = patch_locfiles_line(
        template_text=tpl_text,
        obs_path=obs_path.as_posix(),
        tt_root=tt_root.as_posix()+"/tt_PS",
        out_root=out_root.as_posix(),
        event_id=event_id,
    )

    in_path = (work_dir / "nlloc.in").resolve()
    in_path.write_text(patched, encoding="utf-8")

    # Run NLLoc with cwd=project_root so relative "nlloc_script/..." paths resolve
    cmd = [cfg.nlloc_bin, str(in_path)]
    #print(cmd, f"(cwd={project_root})", f"event_id={event_id}")
    try:
        p = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        return {
            "event_id": event_id, "ok": False, "returncode": None,
            "error": f"NLLoc binary not found: {cfg.nlloc_bin}",
            "stdout": "", "stderr": str(e), "work_dir": str(work_dir)
        }

    ok = (p.returncode == 0)

    # Find likely output files (depends on your NLLoc settings; hyp is common)
    hyp = None
    hyp_candidates = list(out_root.glob(f"{event_id}_*.hyp")) + list(out_root.glob(f"{event_id}_*.HYP"))
    if hyp_candidates:
        hyp = str(hyp_candidates[0])

    if ok and (not cfg.keep_workdir):
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

    return {
        "event_id": event_id,
        "ok": ok,
        "returncode": p.returncode,
        "out_prefix": out_prefix,
        "hyp": hyp,
        "stdout": p.stdout[-4000:],
        "stderr": p.stderr[-4000:],
        "work_dir": str(work_dir),
    }


def main():

    import time 
    cfg = ParallelNLLocConfig(
        keep_workdir=True,
    )


    base_dir = cfg.out_dir.resolve()
    obs_path = base_dir / cfg.obs_file
    tpl_path = base_dir / cfg.nlloc_in_template
    sta_path = base_dir / cfg.stations_file

    if not obs_path.exists():
        raise FileNotFoundError(f"obs file not found: {obs_path}")
    if not tpl_path.exists():
        raise FileNotFoundError(f"template nlloc.in not found: {tpl_path}")
    if not sta_path.exists():
        raise FileNotFoundError(f"stations file not found: {sta_path}")

    obs_text = obs_path.read_text(encoding="utf-8", errors="ignore")
    blocks = split_obs_nll_format(obs_text)

    index_path = base_dir / "split_index.json"
    index_path.write_text(json.dumps([{"event_id": eid, "n_lines": txt.count('\n')} for eid, txt in blocks], indent=2),
                          encoding="utf-8")

    print(f"[INFO] split {len(blocks)} events from {obs_path}")
    if not blocks:
        print("[WARN] no events detected (no 'ID <event_id>' headers found).")
        return

    n_workers = cfg.n_workers if cfg.n_workers > 0 else max(1, cpu_count())
    print(f"[INFO] n_workers={n_workers}")

    tasks = [(cfg, eid, txt) for eid, txt in blocks]
    t0 = time.perf_counter()
    results: List[Dict] = []
    with Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_one_event, tasks):
            results.append(r)
            status = "OK" if r["ok"] else "FAIL"
            print(f"[{status}] {r['event_id']} rc={r['returncode']} hyp={r.get('hyp')} out_prefix={r.get('out_prefix')}")

    ok_cnt = sum(1 for r in results if r["ok"])
    fail = [r for r in results if not r["ok"]]

    summary_path = base_dir / "parallel_loc_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    t1 = time.perf_counter() 
    print("========== Summary ==========")
    print(f"Total={len(results)} OK={ok_cnt} FAIL={len(fail)}")
    print(f"Summary JSON: {summary_path}")
    print(f"Elapsed time: {t1 - t0:.1f} seconds")
    print(f"Average time per event: {(t1 - t0)/len(results):.1f} seconds")
    if fail:
        print("Failed (first 20):")
        for r in fail[:20]:
            print(" -", r["event_id"])
        print("Check stderr/stdout tail in summary JSON and per-event work dirs.")
#360.1 s
#818.1 seconds

#1153.2
#Elapsed time: 1030.7 seconds

if __name__ == "__main__":
    main()
