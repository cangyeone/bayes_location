"""Explicit units and schemas shared by preparation, training and location."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

from .locator import MISSING


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path, required):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty table")
    return rows


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def timestamp(value):
    when = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if when.tzinfo is None:
        raise ValueError(f"time must include a UTC offset or Z: {value}")
    return when.astimezone(dt.timezone.utc)


def validate_geometry(meta):
    if meta.get("projection") != "AEQD":
        raise ValueError("geometry.projection must be AEQD")
    lon0, lat0 = float(meta["lon0"]), float(meta["lat0"])
    if not (-180 <= lon0 <= 180 and -90 < lat0 < 90):
        raise ValueError("invalid projection origin")
    for name in ("source_bounds_km", "receiver_bounds_km"):
        b = np.asarray(meta[name], dtype=float)
        if b.shape != (3, 2) or not np.isfinite(b).all() or np.any(b[:, 0] > b[:, 1]):
            raise ValueError(f"{name} must be finite increasing [x,y,z] bounds")
        if name == "source_bounds_km" and np.any(b[:, 0] == b[:, 1]):
            raise ValueError("source bounds must have positive extent on every axis")
    return meta


def transformers(meta):
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={meta['lat0']} +lon_0={meta['lon0']} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return (
        Transformer.from_crs(4326, local, always_xy=True),
        Transformer.from_crs(local, 4326, always_xy=True),
    )


def within_bounds(values, bounds, name):
    values = np.asarray(values, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite (N,3) coordinates in km")
    # Roundtrip projection and float32 checkpoints introduce sub-millimetre noise.
    valid = ((values >= bounds[:, 0] - 1e-5) & (values <= bounds[:, 1] + 1e-5)).all(axis=1)
    if not valid.all():
        raise ValueError(f"{name}: {int((~valid).sum())} rows outside configured bounds")


def project_rows(rows, meta, station=False):
    lon = np.array([float(r["longitude"]) for r in rows])
    lat = np.array([float(r["latitude"]) for r in rows])
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()) or np.any(abs(lon) > 180) or np.any(abs(lat) > 90):
        raise ValueError("longitude/latitude are invalid")
    x, y = transformers(meta)[0].transform(lon.tolist(), lat.tolist())
    z = [-float(r["elevation_m"]) / 1000.0 if station else float(r["depth_km"]) for r in rows]
    xyz = np.column_stack([np.asarray(x) / 1000.0, np.asarray(y) / 1000.0, z])
    within_bounds(xyz, meta["receiver_bounds_km" if station else "source_bounds_km"], "stations" if station else "sources")
    return xyz.astype(np.float32)


def index_rows(rows, key):
    result = {}
    for i, row in enumerate(rows):
        value = row[key]
        if not value or value in result:
            raise ValueError(f"empty or duplicate {key}: {value!r}")
        result[value] = i
    return result


def read_observations(stations_path, events_path, picks_path, meta, training=False):
    stations = read_csv(stations_path, ["station_id", "longitude", "latitude", "elevation_m"])
    required = ["event_id", "reference_time"]
    if training:
        required += ["longitude", "latitude", "depth_km"]
    events = read_csv(events_path, required)
    picks = read_csv(picks_path, ["event_id", "station_id", "phase", "arrival_time"])
    sta_ids, ev_ids = index_rows(stations, "station_id"), index_rows(events, "event_id")
    station_xyz = project_rows(stations, meta, station=True)
    refs = [timestamp(e["reference_time"]) for e in events]
    paired = {}
    for row in picks:
        if row["station_id"] not in sta_ids or row["event_id"] not in ev_ids:
            raise ValueError(f"unknown station/event in pick: {row['station_id']} / {row['event_id']}")
        phase = row["phase"]
        if phase not in {"P", "S"}:
            raise ValueError(f"phase must be P or S; explicitly convert branch labels first: {phase}")
        key = (ev_ids[row["event_id"]], sta_ids[row["station_id"]])
        record = paired.setdefault(key, {})
        if phase in record:
            raise ValueError(f"duplicate event/station/phase: {row['event_id']}/{row['station_id']}/{phase}")
        value = (timestamp(row["arrival_time"]) - refs[key[0]]).total_seconds()
        if value == MISSING or not np.isfinite(value):
            raise ValueError("invalid relative arrival time")
        if training and value < 0:
            raise ValueError("training arrival precedes the known origin time")
        record[phase] = value
    keys = sorted(paired)
    ids = np.array([k[0] for k in keys], dtype=np.int64)
    station_indices = np.array([k[1] for k in keys], dtype=np.int64)
    result = dict(
        xr=station_xyz[station_indices],
        tp=np.array([paired[k].get("P", MISSING) for k in keys], dtype=np.float32),
        ts=np.array([paired[k].get("S", MISSING) for k in keys], dtype=np.float32),
        event_ids=ids, events=events, stations=stations,
        station_indices=station_indices, reference_times=refs,
    )
    if training:
        result["xs"] = project_rows(events, meta)[ids]
    return result


def save_pairs(path, *, xr, xs, tp, ts, event_id, geometry):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xr=xr, xs=xs, tp=tp, ts=ts,
                        event_id=np.asarray(event_id, dtype=str), geometry_json=json.dumps(geometry))


def load_pairs(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in ("xr", "xs", "tp", "ts", "event_id")}
        meta = validate_geometry(json.loads(str(data["geometry_json"].item())))
    n = len(result["xr"])
    if n < 1 or result["xs"].shape != (n, 3) or any(result[k].shape != (n,) for k in ("tp", "ts", "event_id")):
        raise ValueError("pair arrays must share N; coordinates (N,3), phases/event_id (N,)")
    within_bounds(result["xr"], meta["receiver_bounds_km"], "receivers")
    within_bounds(result["xs"], meta["source_bounds_km"], "sources")
    # Public pair files use NaN or the sampler sentinel for a missing target.
    for k in ("tp", "ts"):
        v = np.asarray(result[k], dtype=np.float32)
        v[v == MISSING] = np.nan
        if np.isinf(v).any() or np.any(v[np.isfinite(v)] < 0):
            raise ValueError("training travel times must be nonnegative seconds or NaN")
        result[k] = v
    if np.any(~np.isfinite(result["tp"]) & ~np.isfinite(result["ts"])):
        raise ValueError("each training row requires at least one phase label")
    return result, meta
