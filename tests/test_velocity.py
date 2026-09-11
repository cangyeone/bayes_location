"""Fictitious velocity profiles verify conversion, coverage, and FMM handoff."""

import json

import numpy as np
import pytest

from bayesloc.cli import main
from bayesloc.fmm import read_grid
from bayesloc.io import load_pairs, read_json, transformers, write_json
from bayesloc.velocity import axis_count, read_model


@pytest.fixture
def profile_table(tmp_path):
    # Include an interior point so the horizontal footprint is not a full product grid.
    lonlat = np.array([[-0.3, -0.3], [-0.3, 0.3], [0.3, -0.3], [0.3, 0.3], [0.05, 0.0]])
    east, north = transformers({"lon0": 0.0, "lat0": 0.0})[0].transform(lonlat[:, 0], lonlat[:, 1])
    rows = []
    for (lon, lat), x, y in zip(lonlat, east / 1000, north / 1000):
        for z in (-5.0, 0.0, 10.0, 40.0):
            # Independent affine functions must survive both interpolation stages.
            rows.append([lon, lat, z, 6 + 0.001*x + 0.002*y + 0.01*z,
                         3.5 + 0.002*x - 0.001*y + 0.005*z])
    rows = np.asarray(rows)
    np.random.default_rng(3).shuffle(rows)
    path = tmp_path / "profiles.txt"
    np.savetxt(path, rows, header="lon lat dep Vp Vs", comments="")
    return path


def build_arguments(model, output):
    return ["build-grid", "--model", str(model), "--depth-reference", "sea-level",
            "--xlim-km", "-10", "10", "--ylim-km", "-12", "12", "--zlim-km", "0", "20",
            "--spacing-km", "5", "6", "10", "--output", str(output)]


def test_affine_velocity_and_axis_order(profile_table, tmp_path):
    out = tmp_path / "grid"
    main(build_arguments(profile_table, out))
    grid, spacing = read_grid(out / "velocity.npz")
    assert grid["vp"].shape == (3, 5, 5)
    assert spacing == (10.0, 6.0, 5.0)
    z, y, x = np.meshgrid(grid["z_km"], grid["y_km"], grid["x_km"], indexing="ij")
    np.testing.assert_allclose(grid["vp"], 6 + 0.001*x + 0.002*y + 0.01*z, atol=1e-12)
    np.testing.assert_allclose(grid["vs"], 3.5 + 0.002*x - 0.001*y + 0.005*z, atol=1e-12)
    geometry = read_json(out / "geometry.json")
    assert geometry == grid["geometry"]
    assert geometry["vertical_datum"] == "mean_sea_level"
    assert geometry["receiver_bounds_km"][2] == [0.0, 0.0]
    assert geometry["lon0"] == geometry["lat0"] == 0.0
    provenance = read_json(out / "provenance.json")
    assert provenance["source_profiles"] == 5
    assert provenance["source_depth_levels"] == 4
    assert provenance["extrapolation"] == "disabled"
    with pytest.raises(SystemExit):
        main(build_arguments(profile_table, out))


@pytest.mark.parametrize("defect,match", [("duplicate", "duplicate"), ("missing", "complete depth"),
                                          ("invalid_speed", "Vp greater"), ("nonfinite", "finite")])
def test_reject_invalid_profiles(profile_table, tmp_path, defect, match):
    values = np.loadtxt(profile_table, skiprows=1)
    if defect == "duplicate":
        values = np.concatenate([values, values[:1]])
    elif defect == "missing":
        values = values[1:]
    elif defect == "invalid_speed":
        values[0, 4] = values[0, 3] + 1
    else:
        values[0, 3] = np.nan
    path = tmp_path / "invalid.txt"
    np.savetxt(path, values)
    with pytest.raises(ValueError, match=match):
        read_model(path)


def test_reject_extrapolation_and_wrong_depth_file(profile_table, tmp_path, capsys):
    base = build_arguments(profile_table, tmp_path / "outside")
    with pytest.raises(SystemExit):
        main(base + ["--xlim-km", "-100", "100"])
    assert "convex hull" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(base + ["--zlim-km", "0", "60"])
    assert "vertical extrapolation" in capsys.readouterr().err
    surface = tmp_path / "SWChinaCVMv2.0.txt"
    surface.write_bytes(profile_table.read_bytes())
    with pytest.raises(SystemExit):
        main(build_arguments(surface, tmp_path / "wrong_datum"))
    assert "surface-relative depth" in capsys.readouterr().err
    assert not (tmp_path / "outside").exists()


def test_grid_size_and_spacing_guards(profile_table, tmp_path, capsys):
    for bounds, spacing in [([0, 10], 3), ([0, 1], 1), ([0, 10], 0), ([0, 10], np.nan)]:
        with pytest.raises(ValueError):
            axis_count(bounds, spacing)
    with pytest.raises(SystemExit):
        main(build_arguments(profile_table, tmp_path / "too_large") + ["--max-nodes", "10"])
    assert "max-nodes" in capsys.readouterr().err


def test_private_geometry_and_fmm_projection_contract(profile_table, tmp_path, capsys):
    first = tmp_path / "first"
    main(build_arguments(profile_table, first))
    geometry = read_json(first / "geometry.json")
    geometry["source_bounds_km"][2] = [0, 10]
    geometry["receiver_bounds_km"][2] = [-5, 0]
    path = tmp_path / "private_geometry.json"
    write_json(path, geometry)
    second = tmp_path / "second"
    main(["build-grid", "--model", str(profile_table), "--depth-reference", "sea-level",
          "--geometry", str(path), "--xlim-km", "-10", "10", "--ylim-km", "-12", "12",
          "--zlim-km", "-5", "20", "--spacing-km", "5", "6", "5", "--output", str(second)])
    assert read_json(second / "geometry.json") == geometry
    changed = json.loads(json.dumps(geometry))
    changed["lon0"] = 0.25  # Mismatch between an otherwise valid config and the grid.
    write_json(tmp_path / "mismatch.json", changed)
    fmm = ["generate-fmm", "--grid", str(second / "velocity.npz"),
           "--n-sources", "3", "--receivers-per-source", "5", "--output", str(tmp_path / "pairs.npz")]
    with pytest.raises(SystemExit):
        main(fmm + ["--geometry", str(tmp_path / "mismatch.json")])
    assert "different projections or vertical datums" in capsys.readouterr().err
    pytest.importorskip("skfmm")
    main(fmm + ["--geometry", str(path)])
    pairs, metadata = load_pairs(tmp_path / "pairs.npz")
    assert metadata == geometry
    assert 0 < len(pairs["tp"]) <= 15
    assert np.all(pairs["ts"] > pairs["tp"])
