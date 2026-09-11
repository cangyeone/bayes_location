# SWChinaCVM-V2.0 → velocity grid → FMM labels → training

This example converts the publicly released [SWChinaCVM-V2.0](https://github.com/liuyingustc/SWChinaCVM-V2.0) velocity model into the Cartesian NPZ used by `generate-fmm`. It supplies a physical velocity model, not a station table or earthquake catalog. Downloaded data, regional geometry, generated labels, and trained weights stay in ignored local directories.

For an entirely fictitious example, `python examples/make_demo.py --output work/demo` already generates `work/demo/velocity.npz`. That file is an analytic toy speed field, not SWChinaCVM. This guide provides the separate public-model route.

## 1. Install and obtain the correct model

From the Bayes Location repository root, with your Python environment activated:

```bash
python -m pip install -e ".[fmm,test]"
mkdir -p private_data/SWChinaCVM-V2.0

curl --fail --location \
  https://raw.githubusercontent.com/liuyingustc/SWChinaCVM-V2.0/2d56970f4848e3184a5d6f23c39aedfe387f67c9/SWChinaCVMv2.0.txt.wrst.sea_level \
  --output private_data/SWChinaCVM-V2.0/SWChinaCVMv2.0.txt.wrst.sea_level
```

This pins the input to upstream commit `2d56970f4848e3184a5d6f23c39aedfe387f67c9`. You can alternatively download the file from its [upstream file page](https://github.com/liuyingustc/SWChinaCVM-V2.0/blob/2d56970f4848e3184a5d6f23c39aedfe387f67c9/SWChinaCVMv2.0.txt.wrst.sea_level). The inspected file has SHA-256 `e7f21c9d0834d910953760e56af0a3218875144ac7d7b4fb2ef5ed7244964426`; conversion records the input hash automatically.

The upstream repository contains two depth conventions:

| Actual upstream filename | Depth reference | This workflow |
|---|---|---|
| `SWChinaCVMv2.0.txt` | Local surface | Requires a separate terrain/datum conversion; do not use directly |
| `SWChinaCVMv2.0.txt.wrst.sea_level` | Mean sea level | Use this file |

The shared header is `lon lat dep Vp Vs`: longitude/latitude in degrees, depth in km, and P/S velocities in km/s. The converter requires an explicit `--depth-reference sea-level` declaration and rejects the known surface-relative filename. Renaming a file does not convert its depth datum. These conventions come from the [upstream README](https://github.com/liuyingustc/SWChinaCVM-V2.0#readme).

## 2. Build the Cartesian grid

```bash
python -m bayesloc build-grid \
  --model private_data/SWChinaCVM-V2.0/SWChinaCVMv2.0.txt.wrst.sea_level \
  --depth-reference sea-level \
  --xlim-km -100 100 --ylim-km -100 100 --zlim-km 0 40 \
  --spacing-km 5 5 5 \
  --output work/sw-china-cvm-v2
```

Choose a new or empty output directory. The command writes:

| Output | Contents |
|---|---|
| `work/sw-china-cvm-v2/velocity.npz` | `x_km`, `y_km`, `z_km`, `vp`, `vs`, plus embedded `geometry_json` |
| `work/sw-china-cvm-v2/geometry.json` | AEQD projection, vertical datum, source bounds, and receiver bounds |
| `work/sw-china-cvm-v2/provenance.json` | Input hash, profile/layer counts, interpolation method, grid shape, and geometry |

The example builds a **200 km × 200 km × 40 km** local box with 5 km spacing: `vp.shape == vs.shape == (9,41,41)` in `(z,y,x)` order. `--spacing-km` takes **dx dy dz**, while scikit-fmm receives **dz dy dx** internally. Bounds must span an integer number of cells and each axis needs at least three nodes. `--max-nodes` defaults to 2,000,000 to limit target-grid memory use.

Without `--geometry`, the projection center is calculated from the midpoint of the input model's geographic bounding box. This is a reproducible model-centered crop, not an inferred study area or station-network center. Regional coordinates are written only to local metadata, not hardcoded into public examples or printed by the converter. The default source bounds equal the requested box; default receivers lie at mean sea level (`z=0`). No real station coordinates or event locations are used.

To target a different study area, supply a locally verified geometry file with `--geometry private_data/geometry.json`, including `vertical_datum: "mean_sea_level"`. The x/y limits are then relative to that file's projection origin. Its source and receiver bounds must fit inside the output grid and contain grid nodes. The tool does not silently modify those bounds.

## 3. Generate supervised FMM labels

Use the grid and geometry from the **same** conversion:

```bash
python -m bayesloc generate-fmm \
  --grid work/sw-china-cvm-v2/velocity.npz \
  --geometry work/sw-china-cvm-v2/geometry.json \
  --n-sources 200 --receivers-per-source 32 \
  --noise-s 0 --seed 1234 \
  --output work/sw-china-cvm-v2/all.fmm.npz
```

FMM solves a P and S travel-time field for each sampled source and extracts times at sampled receiver nodes. The receiver nodes are simulated training locations, not an observed station network. Coincident source/receiver pairs are omitted. Two hundred sources with 32 receivers therefore yield **at most 6,400 pairs**.

The pair NPZ contains projected `xr`, `xs`, P/S time labels, source IDs, and geometry; its `.provenance.json` records the grid hash and numerical settings. A grid with embedded metadata rejects a mismatched projection or vertical datum. Do not substitute `examples/geometry.json`, whose fictitious origin belongs to the toy demo.

The number of FMM solves scales with source count. Start with a small run, inspect sampling coverage, and then increase the count using a new output filename. A finer grid is an interpolation/numerical choice; it does not add resolution to the published physical model.

## 4. Train and independently evaluate the travel-time network

First reserve test sources **before** training or model selection. The following creates both `train.fmm.npz` and `test.independent.npz` from the generated pool, keeping all receivers of one source together:

```bash
python - <<'PY'
from pathlib import Path
from bayesloc.io import load_pairs, save_pairs, sha256, write_json
from bayesloc.training import split_sources

root = Path("work/sw-china-cvm-v2")
pool = root / "all.fmm.npz"
data, geometry = load_pairs(pool)
train_rows, test_rows = split_sources(data["xs"], fraction=0.2, seed=20260911)
for name, rows in (("train.fmm", train_rows), ("test.independent", test_rows)):
    output = root / (name + ".npz")
    save_pairs(output, geometry=geometry, **{key: values[rows] for key, values in data.items()})
    write_json(str(output) + ".provenance.json", {
        "parent_sha256": sha256(pool), "seed": 20260911,
        "split": name, "grouping": "source coordinates", "parent_rows": rows.tolist(),
    })
PY
```

The training command then splits its own input into training and validation sources; the reserved test file remains separate:

```bash
python -m bayesloc train \
  --pairs work/sw-china-cvm-v2/train.fmm.npz \
  --checkpoint work/sw-china-cvm-v2/time.fmm.pt \
  --epochs 20 --batch-size 128 --hidden-dim 256 \
  --validation-fraction 0.2 --device auto
```

Training is supervised regression on FMM labels. It does not invert the velocity structure or apply an eikonal loss during network training. Validation groups rows by source coordinates. The best-validation weights and matching geometry are saved in the checkpoint.

Evaluate once model selection is complete, using the test file created above:

```bash
python -m bayesloc evaluate \
  --pairs work/sw-china-cvm-v2/test.independent.npz \
  --checkpoint work/sw-china-cvm-v2/time.fmm.pt \
  --output work/sw-china-cvm-v2/test.metrics.json --device auto
```

The split guarantees distinct source nodes; nearby sources and the shared velocity model still introduce correlations. Consider spatial blocks or other independent test designs for substantive validation. **Generating another pool with a different random seed alone does not guarantee disjoint sources.** Neither 200 sampled sources nor 20 epochs establishes useful regional location accuracy. Evaluate forward errors and adequate source/receiver coverage before using the model for inference.

## 5. Use the trained model for location

After validation, use the station, event, and pick CSV schemas in the [README](../README.md#location):

```bash
python -m bayesloc locate \
  --stations private_data/stations.csv \
  --events private_data/events.csv \
  --picks private_data/picks.csv \
  --checkpoint work/sw-china-cvm-v2/time.fmm.pt \
  --output work/sw-china-cvm-v2/location \
  --mode student_t_z --device auto
```

These CSVs must come from your authorized observations; the CVM file does not supply them. The checkpoint embeds the conversion geometry. Station positions, source coverage, phase definitions, and depth datum must agree with training. The default sea-level receiver layer does not cover elevated stations.

For stations above sea level, use their actual signed `z=-elevation_m/1000`, and train receivers over the required depth range. For example, to extend the grid to negative depths while keeping source sampling below sea level, first save a local geometry configuration:

```python
from bayesloc.io import read_json, write_json

geometry = read_json("work/sw-china-cvm-v2/geometry.json")
geometry["source_bounds_km"][2] = [0, 40]
geometry["receiver_bounds_km"][2] = [-5, 0]
write_json("private_data/sw-china-elevation.geometry.json", geometry)
```

Then rebuild in a new directory and regenerate labels/retrain:

```bash
python -m bayesloc build-grid \
  --model private_data/SWChinaCVM-V2.0/SWChinaCVMv2.0.txt.wrst.sea_level \
  --depth-reference sea-level \
  --geometry private_data/sw-china-elevation.geometry.json \
  --xlim-km -100 100 --ylim-km -100 100 --zlim-km -5 40 \
  --spacing-km 5 5 5 \
  --output work/sw-china-cvm-v2-elevation
```

The bounds above are illustrative local depths, not a list of real station elevations. Choose receiver sampling density for the elevations you will use. Editing geometry alone does not expand an existing checkpoint's learned support.

## 6. Interpolation method and limits

The implementation is [bayesloc/velocity.py](../bayesloc/velocity.py), exposed as `python -m bayesloc build-grid`:

1. Parse the five-column table, ignoring its recognized header, blank lines, and `#` comments. Reject invalid coordinates/speeds, duplicate points, or incomplete depth profiles.
2. Reconstruct profiles by horizontal position and depth. Every profile must have the same depth levels; the horizontal footprint need not be a full longitude/latitude product grid.
3. Project horizontal positions using WGS84 AEQD in metres, then convert to km.
4. Interpolate **velocity**, separately for Vp and Vs, using [SciPy's linear Delaunay interpolation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html) in projected x/y, then linearly interpolate each profile in depth.
5. Reject targets outside the horizontal convex hull or input depth range. Write regular `(Nz,Ny,Nx)` arrays after checking finite positive `Vp > Vs` everywhere.

There is no nearest-neighbour fill or vertical extrapolation. Horizontal triangulation can bridge unsampled gaps inside the convex hull, so inspect model coverage before selecting a large box. The converter does not handle arbitrary scattered 3-D samples, profiles with different depth levels, dateline crossings, anisotropy, or terrain masks.

Using mean-sea-level depths does not add a topographic boundary to FMM. The rectangular grid uses the velocities provided by the source model, including supported negative depths if requested. It may include cells above local terrain. A terrain-aware forward solver and appropriate domain mask require separate implementation and validation.

## Attribution and validation

Model reference: Liu et al. (2023), *The high-resolution community velocity model V2.0 of southwest China, constructed by joint body and surface wave tomography of data recorded at temporary dense arrays*, [Science China Earth Sciences, DOI: 10.1007/s11430-022-1161-7](https://doi.org/10.1007/s11430-022-1161-7). Consult the [upstream repository](https://github.com/liuyingustc/SWChinaCVM-V2.0) for model usage information and attribution. This repository provides the adapter and instructions; it does not redistribute the velocity tables.

See [VALIDATION.md](VALIDATION.md) for checks actually performed. Fictitious affine velocity fields test interpolation accuracy and axis order; the downloaded public model is used only in a small local conversion/FMM/training check. Neither check establishes the accuracy of real earthquake locations.
