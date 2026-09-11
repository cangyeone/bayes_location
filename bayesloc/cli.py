"""Command-line interfaces; run `python -m bayesloc --help`."""

from __future__ import annotations

import argparse


def parser():
    p = argparse.ArgumentParser(description="Travel-time training, robust earthquake location, and catalog QC")
    sub = p.add_subparsers(dest="command", required=True)

    def device_options(command):
        command.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    def observation_options(command):
        command.add_argument("--stations", required=True, help="station CSV (elevation_m)")
        command.add_argument("--events", required=True, help="event CSV with reference_time")
        command.add_argument("--picks", required=True, help="pick CSV with absolute arrival_time")

    prep = sub.add_parser("prepare", help="convert labeled catalog observations to supervised NPZ pairs")
    observation_options(prep)
    prep.add_argument("--geometry", required=True)
    prep.add_argument("--output", required=True, help="output .npz file")

    fmm = sub.add_parser("generate-fmm", help="generate labeled pairs from a regular 3-D velocity grid")
    fmm.add_argument("--grid", required=True, help="NPZ with x_km,y_km,z_km,vp,vs; velocities (Nz,Ny,Nx)")
    fmm.add_argument("--geometry", required=True)
    fmm.add_argument("--n-sources", type=int, default=200)
    fmm.add_argument("--receivers-per-source", type=int, default=32)
    fmm.add_argument("--noise-s", type=float, default=0.0)
    fmm.add_argument("--seed", type=int, default=1234)
    fmm.add_argument("--output", required=True, help="output .npz file")

    train = sub.add_parser("train", help="train a supervised P/S travel-time network")
    train.add_argument("--pairs", required=True)
    train.add_argument("--validation", help="optional source-disjoint labeled NPZ")
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=20260911)
    device_options(train)

    evaluate = sub.add_parser("evaluate", help="report held-out phase-specific MAE/RMSE")
    evaluate.add_argument("--pairs", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--geometry", help="required for legacy checkpoints without domain metadata")
    evaluate.add_argument("--batch-size", type=int, default=8192)
    evaluate.add_argument("--output", required=True)
    device_options(evaluate)

    loc = sub.add_parser("locate", help="initialize and sample a catalog; export CSV and posterior chains")
    observation_options(loc)
    loc.add_argument("--checkpoint", required=True)
    loc.add_argument("--geometry", help="required for legacy checkpoints without domain metadata")
    loc.add_argument("--output", required=True, help="new/empty output directory")
    loc.add_argument("--mode", choices=["gaussian", "student_t", "student_t_z"], default="student_t_z")
    loc.add_argument("--samples", type=int, default=4000, help="total iterations per chain, including burn")
    loc.add_argument("--burn", type=int, default=2000)
    loc.add_argument("--thin", type=int, default=2)
    loc.add_argument("--chains", type=int, default=3)
    loc.add_argument("--seed", type=int, default=20260911)
    loc.add_argument("--nu", type=float, default=4.0)
    loc.add_argument("--proposal-km", type=float, default=2.0)
    loc.add_argument("--pool-size", type=int, default=16)
    loc.add_argument("--refine-top", type=int, default=4)
    loc.add_argument("--init-steps", type=int, default=100)
    loc.add_argument("--min-stations", type=int, default=3)
    loc.add_argument("--min-phases", type=int, default=4)
    device_options(loc)

    qc = sub.add_parser("filter", help="rank by posterior widths and optionally apply absolute thresholds")
    qc.add_argument("--catalog", required=True)
    qc.add_argument("--top-k", type=int)
    qc.add_argument("--max-horizontal-km", type=float, help="upper bound on hypot(width_x90,width_y90)")
    qc.add_argument("--max-depth-km", type=float, help="upper bound on width_z90")
    qc.add_argument("--max-rhat", type=float, help="upper bound on classical split R-hat for x,y,z,t0")
    qc.add_argument("--output", required=True)

    inverse = sub.add_parser("reproject", help="convert x/y in km back to WGS84 longitude/latitude")
    inverse.add_argument("--catalog", required=True)
    inverse.add_argument("--geometry", required=True)
    inverse.add_argument("--output", required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if hasattr(args, "device"):
            import torch
            from .locator import require_device
            name = args.device
            if name == "auto":
                name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            device = require_device(name)
        if args.command in {"prepare", "generate-fmm"} and not args.output.endswith(".npz"):
            raise ValueError("pair output must use a .npz extension")
        if args.command == "prepare":
            import numpy as np
            from .io import read_json, read_observations, save_pairs, sha256, validate_geometry, write_json
            meta = validate_geometry(read_json(args.geometry))
            data = read_observations(args.stations, args.events, args.picks, meta, training=True)
            save_pairs(args.output, **{k: data[k] for k in ("xr", "xs", "tp", "ts")},
                       event_id=np.array([data["events"][i]["event_id"] for i in data["event_ids"]]), geometry=meta)
            write_json(args.output + ".provenance.json", {
                "arguments": vars(args), "pairs": len(data["xr"]),
                "input_sha256": {k: sha256(getattr(args, k)) for k in ("stations", "events", "picks", "geometry")},
            })
            print(f"Saved {len(data['xr'])} supervised pairs: {args.output}")
        elif args.command == "generate-fmm":
            from .fmm import generate
            generate(args)
        elif args.command == "train":
            from .training import train
            train(args, device)
        elif args.command == "evaluate":
            import numpy as np
            from .io import load_pairs, sha256, write_json
            from .training import loader, metrics
            from .workflow import load_model
            model, meta = load_model(args.checkpoint, device, args.geometry)
            data, data_meta = load_pairs(args.pairs)
            if meta != data_meta:
                raise ValueError("checkpoint and evaluation geometry differ")
            result = metrics(model, loader(data, np.arange(len(data["xr"])), args.batch_size), device)
            write_json(args.output, {"metrics": result, "pairs_sha256": sha256(args.pairs),
                                     "checkpoint_sha256": sha256(args.checkpoint)})
            print(result)
        elif args.command == "locate":
            from .workflow import locate
            locate(args, device)
        elif args.command == "filter":
            from .workflow import filter_catalog
            filter_catalog(args)
        elif args.command == "reproject":
            from .workflow import reproject
            reproject(args)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        p.error(str(exc))
