"""Executable contracts for scientific units, masking, inference, FMM and the CLI."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from bayesloc.cli import main
from bayesloc.fmm import travel_time_field
from bayesloc.io import (load_pairs, project_rows, read_csv, read_json,
                         read_observations, timestamp, transformers, write_csv)
from bayesloc.locator import MISSING, TravelTimeNet, run_sampler
from bayesloc.training import masked_mse, split_sources
from bayesloc.workflow import load_model, score_and_select, split_rhat


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "examples/make_demo.py"), "--output", str(tmp_path)], check=True)
    return tmp_path


def test_projection_roundtrip_and_signed_elevation():
    meta = read_json(ROOT / "examples/geometry.json")
    meta["receiver_bounds_km"][2] = [-2, 1]
    row = dict(longitude="0.01", latitude="-0.01", elevation_m="1500")
    xyz = project_rows([row], meta, station=True)
    assert xyz[0, 2] == -1.5
    lon, lat = transformers(meta)[1].transform(xyz[0, 0] * 1000, xyz[0, 1] * 1000)
    np.testing.assert_allclose([lon, lat], [0.01, -0.01], atol=1e-7)
    row["elevation_m"] = "-100"
    assert project_rows([row], meta, station=True)[0, 2] == pytest.approx(0.1)


def test_timezone_and_reference_time(demo):
    assert timestamp("2020-01-01T08:00:00+08:00") == timestamp("2020-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="offset"):
        timestamp("2020-01-01T00:00:00")
    meta = read_json(demo / "geometry.json")
    obs = read_observations(demo / "stations.csv", demo / "events.csv", demo / "picks.csv", meta)
    train = read_observations(demo / "stations.csv", demo / "training_events.csv", demo / "training_picks.csv", meta, training=True)
    # Location reference time is deliberately one second after true origin.
    np.testing.assert_allclose(train["tp"][:15] - obs["tp"], 1.0, atol=1e-6)


def test_duplicate_and_unknown_pick_rejected(demo):
    meta = read_json(demo / "geometry.json")
    rows = read_csv(demo / "picks.csv", ["phase"])
    write_csv(demo / "bad.csv", rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        read_observations(demo / "stations.csv", demo / "events.csv", demo / "bad.csv", meta)
    rows[0]["station_id"] = "UNKNOWN"
    write_csv(demo / "bad.csv", rows)
    with pytest.raises(ValueError, match="unknown"):
        read_observations(demo / "stations.csv", demo / "events.csv", demo / "bad.csv", meta)


def test_masked_training_and_source_disjoint_split():
    pred = torch.tensor([[2., 100.], [100., 4.]], requires_grad=True)
    target = torch.tensor([[1., float("nan")], [float("nan"), 2.]])
    loss = masked_mse(pred, target)
    assert float(loss.detach()) == 2.5
    loss.backward()
    assert pred.grad[0, 1] == pred.grad[1, 0] == 0
    xs = np.repeat(np.array([[0, 0, 1], [2, 2, 2], [3, 3, 3]]), 5, axis=0)
    train, valid = split_sources(xs, 0.3, 17)
    assert not ({tuple(x) for x in xs[train]} & {tuple(x) for x in xs[valid]})


def test_qc_ties_invalid_and_thresholds():
    rows = [dict(event_id=i, width_x90_km=x, width_y90_km=y, width_z90_km=z)
            for i, x, y, z in [("a", 8, 8, 1), ("b", 10, 0, 2), ("c", 1, 1, 9), ("d", np.nan, 1, 1)]]
    scored = score_and_select(rows, top_k=2)
    np.testing.assert_allclose([r["qc_score"] for r in scored[:3]], [1, 2/3, 1])
    assert {r["event_id"] for r in scored if r["retained"]} == {"a", "b"}
    assert scored[3]["rejection_reason"] == "invalid_width"
    score_and_select(rows, top_k=3, max_z=1.5)
    assert [r["event_id"] for r in rows if r["retained"]] == ["a"]


def test_rhat_detects_separated_chains():
    values = np.random.default_rng(7).normal(size=(3, 400, 2, 4))
    assert split_rhat(values).max() < 1.03
    values[0] += 10
    assert split_rhat(values).min() > 3
    assert np.isinf(split_rhat(np.zeros((3, 20, 1, 4)))).all()
    assert np.isnan(split_rhat(values[:1])).all()


def test_qc_invalid_first_row_and_empty_selection(tmp_path):
    source = tmp_path / "catalog.csv"
    rows = [dict(event_id="invalid", width_x90_km="nan", width_y90_km=1, width_z90_km=1),
            dict(event_id="valid", width_x90_km=1, width_y90_km=1, width_z90_km=1)]
    write_csv(source, rows)
    output = tmp_path / "empty.csv"
    main(["filter", "--catalog", str(source), "--top-k", "0", "--output", str(output)])
    assert len(output.read_text().splitlines()) == 1
    audit = read_csv(str(output) + ".audit.csv", ["rejection_reason"])
    assert [r["rejection_reason"] for r in audit] == ["invalid_width", "above_top_k"]


def test_public_examples_are_fictitious_and_real_catalogs_are_not_tracked():
    meta = read_json(ROOT / "examples/geometry.json")
    assert (meta["lon0"], meta["lat0"]) == (0, 0)
    for name in ("nlloc_sc", "nlloc_sc_crop"):
        assert not (ROOT / "scripts/data" / name / "all.locfiles.csv").exists()


class HomogeneousModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, xr, xs):
        distance = torch.linalg.vector_norm(xr - xs, dim=-1)
        return torch.stack([distance / 6, distance / 3.5], dim=-1)


@pytest.mark.parametrize("mode", ["gaussian", "student_t", "student_t_z"])
def test_sampler_missing_phase_bounds_and_time_reference(mode):
    model = HomogeneousModel()
    xr = np.array([[-15, -15, 0], [15, -15, 0], [15, 15, 0], [-15, 15, 0], [0, 0, 0]], dtype=np.float32)
    truth = np.array([[2., -3., 7.]], dtype=np.float32)
    travel = model(torch.tensor(xr), torch.tensor(truth)).numpy()
    tp, ts = travel[:, 0] - 1.0, travel[:, 1] - 1.0
    ts[-1] = MISSING
    result = run_sampler(model=model, xr=xr, tp=tp, ts=ts, event_ids=np.zeros(5, dtype=int), n_events=1,
                         xs_init=truth, t0_init=np.array([-1.0]), mode=mode, seed=51,
                         n_samples=601, burn=300, thin=7, physical_bounds=((-20, 20), (-20, 20), (0, 20)), verbose=False)
    assert result["xs_samples"].shape == (43, 1, 3)
    assert result["retained_source_samples_outside_domain"] == 0
    assert np.isfinite(result["xs_samples"]).all()
    assert abs(float(result["t0_samples"].mean()) + 1) < 0.8
    assert np.linalg.norm(result["xs_samples"].mean(axis=0) - truth) < 4
    assert result["inlier_probability_s"][-1] == 0


def test_fmm_homogeneous_travel_time_and_anisotropic_spacing():
    pytest.importorskip("skfmm")
    speed = np.full((21, 21, 21), 6.0)
    field = travel_time_field(speed, (10, 10, 10), (0.5, 1.0, 2.0))
    assert field[10, 10, 10] == 0
    assert field[10, 10, 20] == pytest.approx(20 / 6, rel=0.04)
    assert field[20, 10, 10] == pytest.approx(5 / 6, rel=0.04)
    faster = travel_time_field(speed * 2, (10, 10, 10), (0.5, 1.0, 2.0))
    np.testing.assert_allclose(field / 2, faster)


def test_end_to_end_cli(demo):
    pairs, checkpoint = str(demo / "pairs.npz"), str(demo / "time.pt")
    main(["prepare", "--stations", str(demo / "stations.csv"), "--events", str(demo / "training_events.csv"),
          "--picks", str(demo / "training_picks.csv"), "--geometry", str(demo / "geometry.json"), "--output", pairs])
    data, _ = load_pairs(pairs)
    assert len(data["xr"]) == 400
    main(["train", "--pairs", pairs, "--checkpoint", checkpoint, "--epochs", "1", "--hidden-dim", "8", "--device", "cpu"])
    main(["evaluate", "--pairs", pairs, "--checkpoint", checkpoint, "--output", str(demo / "metrics.json"), "--device", "cpu"])
    main(["locate", "--stations", str(demo / "stations.csv"), "--events", str(demo / "events.csv"),
          "--picks", str(demo / "picks.csv"), "--checkpoint", checkpoint, "--output", str(demo / "run"),
          "--samples", "13", "--burn", "4", "--thin", "2", "--chains", "2", "--pool-size", "6",
          "--refine-top", "1", "--init-steps", "1", "--device", "cpu"])
    catalog = read_csv(demo / "run/catalog.csv", ["origin_time", "width_z90_km"])
    assert len(catalog) == 3
    main(["filter", "--catalog", str(demo / "run/catalog.csv"), "--top-k", "2", "--output", str(demo / "kept.csv")])
    assert len(read_csv(demo / "kept.csv", ["qc_score"])) == 2
    main(["reproject", "--catalog", str(demo / "kept.csv"), "--geometry", str(demo / "geometry.json"), "--output", str(demo / "geo.csv")])
    rows = read_csv(demo / "geo.csv", ["longitude"])
    before = read_csv(demo / "kept.csv", ["longitude"])
    np.testing.assert_allclose([float(r["longitude"]) for r in rows], [float(r["longitude"]) for r in before])
    assert read_json(demo / "run/run.json")["retained_draws_per_chain"] == 5


def test_legacy_checkpoint_requires_geometry(tmp_path):
    # Exercise the historical file contract without redistributing a research model.
    path = tmp_path / "legacy.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        original = TravelTimeNet(hidden_dim=8).eval()
    torch.save({
        "model_state": original.state_dict(),
        "model_hidden_dim": 8,
        "meta": {"projection_meta": {"lon0": 0.0, "lat0": 0.0}},
    }, path)
    with pytest.raises(ValueError, match="legacy checkpoint"):
        load_model(path, torch.device("cpu"))
    meta = read_json(ROOT / "examples/geometry.json")
    from bayesloc.io import write_json
    geometry = tmp_path / "private_geometry.json"
    write_json(geometry, meta)
    model, loaded_meta = load_model(path, torch.device("cpu"), geometry)
    receiver, source = torch.zeros(2, 3), torch.ones(2, 3)
    torch.testing.assert_close(model(receiver, source), original(receiver, source))
    assert loaded_meta == meta
    meta["lon0"] = 0.25  # Deliberately mismatched fictitious projection.
    write_json(geometry, meta)
    with pytest.raises(ValueError, match="conflicts with the legacy checkpoint projection"):
        load_model(path, torch.device("cpu"), geometry)
