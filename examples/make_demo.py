"""Generate fictitious CSV observations and a small 3-D velocity grid.

Run from the repository root: python examples/make_demo.py --output work/demo
The homogeneous analytic CSV arrivals exercise data plumbing, not real-data accuracy.
The separate heterogeneous grid exercises the FMM path.
"""

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from bayesloc.io import read_json, transformers, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="work/demo")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_json(Path(__file__).with_name("geometry.json"))
    write_json(out / "geometry.json", meta)
    inv = transformers(meta)[1]
    receiver = np.array([[-25, -25, 0], [25, -25, 0], [25, 25, 0], [-25, 25, 0], [0, 0, 0]], dtype=float)
    stations = []
    for i, (x, y, _) in enumerate(receiver):
        lon, lat = inv.transform(x * 1000.0, y * 1000.0)
        stations.append(dict(station_id=f"DEMO.S{i:02d}", longitude=lon, latitude=lat, elevation_m=0))
    write_csv(out / "stations.csv", stations)
    rng = np.random.default_rng(20260911)
    source = rng.uniform([-20, -20, 2], [20, 20, 18], size=(80, 3))
    training_events, location_events, picks = [], [], []
    start = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    for i, xyz in enumerate(source):
        origin = start + dt.timedelta(seconds=i * 60)
        event_id = f"demo-{i:03d}"
        lon, lat = inv.transform(xyz[0] * 1000.0, xyz[1] * 1000.0)
        training_events.append(dict(event_id=event_id, reference_time=origin.isoformat(),
                                    longitude=lon, latitude=lat, depth_km=xyz[2]))
        location_events.append(dict(event_id=event_id, reference_time=(origin + dt.timedelta(seconds=1)).isoformat()))
        for j, distance in enumerate(np.linalg.norm(receiver - xyz, axis=1)):
            for phase, speed in (("P", 6.0), ("S", 3.5)):
                travel_time = distance / speed + rng.normal(0, 0.01)
                picks.append(dict(event_id=event_id, station_id=stations[j]["station_id"], phase=phase,
                                  arrival_time=(origin + dt.timedelta(seconds=travel_time)).isoformat()))
    write_csv(out / "training_events.csv", training_events)
    write_csv(out / "events.csv", location_events[:3])
    write_csv(out / "training_picks.csv", picks)
    write_csv(out / "picks.csv", [p for p in picks if p["event_id"] in {r["event_id"] for r in location_events[:3]}])
    x, y, z = np.arange(-30., 31., 5.), np.arange(-30., 31., 5.), np.arange(0., 21., 5.)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    vp = 5.8 + 0.02 * zz + 0.1 * np.sin(xx / 20.) * np.cos(yy / 20.)
    np.savez_compressed(out / "velocity.npz", x_km=x, y_km=y, z_km=z, vp=vp, vs=vp / 1.73)
    print(f"Created fictitious demonstration inputs: {out}")


if __name__ == "__main__":
    main()
