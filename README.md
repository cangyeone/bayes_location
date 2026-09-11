# Bayes Location: Travel-Time Training, Bayesian Earthquake Location, and Catalog QC

A supervised neural travel-time surrogate with robust Bayesian earthquake location and event-level quality control.

This project embeds a differentiable P/S travel-time network in a Bayesian locator. It returns hypocenter and origin-time estimates, posterior samples, and uncertainty summaries that can be used to rank and screen candidate events.

The supported entry point is `python -m bayesloc`. It covers **training from observed data**, **generating training labels with FMM and a 3-D velocity structure**, **location**, **catalog filtering**, and **inverse projection to longitude/latitude**. The updated sampler and initializer come from the corrected research workflow. See [algorithm details](docs/ALGORITHM.md), [migration instructions](docs/MIGRATION.md), and [validation performed](docs/VALIDATION.md). Historical scripts remain as references; use the unified interface for new work.

> The main travel-time model uses **supervised regression**. Labels may come from a reference earthquake catalog or an offline numerical solver. Training does not read a velocity field or impose an eikonal loss. The historical class name `PINNTravelTime` does not, by itself, mean the model is physics-informed.

## Contents

1. [Installation and a runnable example](#quickstart)
2. [Coordinates, units, and data contracts](#coordinates)
3. [Route A: training from observed data](#observed-training)
4. [Route B: FMM labels from a 3-D velocity structure](#fmm-training)
5. [Training, validation, and checkpoint reuse](#model)
6. [Earthquake location: inputs, commands, and outputs](#location)
7. [Pick diagnostics and event filtering](#qc)
8. [Inverse projection to longitude/latitude](#projection)
9. [Troubleshooting and limitations](#limitations)
10. [Working with a Coding agent](#agents)
11. [Developers](#developers)

<a id="quickstart"></a>
## 1. Installation and a runnable example

Python 3.10+ is required. Core dependencies are PyTorch, NumPy, SciPy, and pyproj. FMM additionally requires scikit-fmm; dependency declarations are in [pyproject.toml](pyproject.toml).

```bash
git clone https://github.com/cangyeone/bayes_location.git
cd bayes_location
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[fmm,test]"
python -m bayesloc --help
```

On Windows, activate the environment with `.venv\Scripts\activate`. Use `python -m pip install -e .` if you do not need FMM or tests. If no scikit-fmm wheel is available for your platform, installation requires a suitable C/C++ build environment.

Training, evaluation, and location accept `--device auto|cpu|cuda|mps`. Automatic selection prefers CUDA, then Apple MPS, then CPU. An explicitly requested unavailable device raises an error. On MPS, Gamma/Beta random draws use a CPU fallback while network evaluations remain on MPS. CPU supports the complete workflow; large catalogs may benefit from an accelerator.

The example below generates **fictitious** stations and events. It needs no restricted research data:

```bash
# Create CSV inputs and a small 3-D velocity grid.
python examples/make_demo.py --output work/demo

# Convert observations with known sources/origin times to supervised pairs.
python -m bayesloc prepare \
  --stations work/demo/stations.csv \
  --events work/demo/training_events.csv \
  --picks work/demo/training_picks.csv \
  --geometry work/demo/geometry.json \
  --output work/demo/observed_pairs.npz

# Short training run to exercise the interface.
python -m bayesloc train \
  --pairs work/demo/observed_pairs.npz \
  --checkpoint work/demo/time.pt \
  --epochs 3 --hidden-dim 32 --batch-size 128 --device cpu

# Short location run; events.csv supplies reference times, not true locations.
python -m bayesloc locate \
  --stations work/demo/stations.csv \
  --events work/demo/events.csv \
  --picks work/demo/picks.csv \
  --checkpoint work/demo/time.pt \
  --output work/demo/location \
  --samples 80 --burn 40 --thin 2 --chains 2 \
  --pool-size 6 --refine-top 2 --init-steps 5 --device cpu

# Rank events by posterior widths and retain at most two.
python -m bayesloc filter \
  --catalog work/demo/location/catalog.csv \
  --top-k 2 --output work/demo/selected.csv

# Recompute geographic coordinates from the projected estimates.
python -m bayesloc reproject \
  --catalog work/demo/selected.csv \
  --geometry work/demo/geometry.json \
  --output work/demo/selected.geographic.csv

python -m pytest -q
```

**Three training epochs and 80 sampling iterations are only a plumbing check. They do not establish useful location accuracy or convergence.** The demonstration CSV uses analytic homogeneous-medium arrivals. Its accompanying `velocity.npz` is a separate heterogeneous example for the FMM route; these are not training and test data from the same physical model. Production use requires independent travel-time validation and adequate initialization and sampling diagnostics.

Choose a new output directory for each location run. The locator refuses to overwrite a nonempty run directory.

<a id="coordinates"></a>
## 2. Coordinates, units, and data contracts

Training, inference, filtering, and export must use consistent conventions:

| Quantity | Convention |
|---|---|
| `longitude`, `latitude` | WGS84 degrees, longitude first |
| Network inputs `xr`, `xs` | Receiver/source `[x,y,z]`, in **km** |
| `x`, `y` | Local AEQD projected coordinates; east/north near the projection center |
| `z`, `depth_km` | Depth positive downward, in km, relative to a common vertical datum |
| `elevation_m` | Station elevation positive upward, in **m**, relative to that same datum |
| Receiver depth | `z_receiver_km = -elevation_m / 1000`, preserving the input sign |
| Velocity | km/s; grid axes in km |
| Travel time, relative arrival, origin-time correction | Seconds |
| Absolute time | ISO 8601 with `Z` or an explicit UTC offset; converted to UTC internally |

Example configuration, also provided in [examples/geometry.json](examples/geometry.json):

```json
{
  "projection": "AEQD",
  "lon0": 0.0,
  "lat0": 0.0,
  "source_bounds_km": [[-30, 30], [-30, 30], [0, 20]],
  "receiver_bounds_km": [[-30, 30], [-30, 30], [0, 0]]
}
```

The receiver depth range `[0,0]` describes a model trained for zero-depth receivers. To use station elevations, generate or collect training receivers covering the relevant depths, update the configuration, and retrain/validate. **Editing bounds alone does not expand a model's learned support.** Hypocentral depth is relative to the chosen datum; the code does not automatically convert it to depth below local terrain.

New checkpoints embed `geometry`, which location reads automatically. An optional `--geometry` must match it exactly. Projection origin, units, network scaling, elevation convention, phase definitions, and applicable domain are part of the model contract.

<a id="observed-training"></a>
## 3. Route A: training from observed data

### 3.1 Required observations

You need station positions, credible reference hypocenters and origin times, and their associated P/S arrivals:

```text
Training label T = absolute arrival_time − known origin/reference_time
Network input    = [receiver_x, receiver_y, receiver_z, source_x, source_y, source_z]
Network output   = [Tp, Ts]
```

This route does not require a velocity structure or FMM. Unlocated picks alone do not supply the source positions and origin times required for this supervised objective. Catalog errors, clock errors, and incorrect picks become label errors, so inspect their provenance and quality first.

### 3.2 Three CSV tables

`stations.csv`: one row per station. A `NET.STA` identifier is convenient.

```csv
station_id,longitude,latitude,elevation_m
XX.AAA,-0.10,-0.10,0
XX.BBB,0.10,0.10,0
```

`training_events.csv`: one row per event. During training, `reference_time` **must be the known/reference origin time**.

```csv
event_id,reference_time,longitude,latitude,depth_km
event-001,2020-01-01T00:00:00Z,0.00,0.00,10.0
```

`training_picks.csv`: one row per observed phase.

```csv
event_id,station_id,phase,arrival_time
event-001,XX.AAA,P,2020-01-01T00:00:03.100Z
event-001,XX.AAA,S,2020-01-01T00:00:05.400Z
event-001,XX.BBB,P,2020-01-01T00:00:03.300Z
```

These few rows illustrate the schema; they are not a sufficient training dataset. Column order may vary, but names must match. Omit a missing phase row rather than inventing a zero arrival. Duplicate `event_id + station_id + phase` records raise an error and need an explicit upstream policy. A receiver may have only P or only S.

The unified interface accepts only `P` and `S`. It does not silently merge `Pg/Pn/Sg/Sn`. Map branch labels only if the intended forward model treats them as the same target. The historical four-output `pnsn` models are incompatible with this two-output interface.

### 3.3 Prepare and train

Place your authorized inputs under `private_data/` and define the projection and source/receiver bounds for your region:

```bash
python -m bayesloc prepare \
  --stations private_data/stations.csv \
  --events private_data/training_events.csv \
  --picks private_data/training_picks.csv \
  --geometry private_data/geometry.json \
  --output work/train.observed.npz

python -m bayesloc train \
  --pairs work/train.observed.npz \
  --checkpoint work/models/time.observed.pt \
  --epochs 20 --batch-size 128 --lr 1e-4 \
  --hidden-dim 256 --validation-fraction 0.2 --seed 20260911 \
  --device auto
```

`prepare` writes one event–receiver pair per NPZ row, with independent P/S masks. Its `.provenance.json` records arguments and input hashes. Unknown stations, duplicate keys, timezone-free timestamps, nonfinite coordinates, out-of-domain locations, and negative training travel times raise errors. If a source file omits timezone information, establish its actual timezone before adding the correct offset; do not assume UTC.

<a id="fmm-training"></a>
## 4. Route B: FMM labels from a 3-D velocity structure

### 4.1 Workflow

```text
3-D Vp/Vs structure → regular velocity grid in a common projected coordinate system
  → P and S FMM travel-time fields for sampled sources
  → extract times at receivers → supervised pair NPZ
  → the same supervised training used in Route A → neural checkpoint
```

FMM solves the eikonal equation `|∇T(x)| = 1/v(x)` for first-arrival travel times in the specified structure. The network subsequently fits these offline labels. This is not simultaneous velocity inversion, and FMM is not part of network backpropagation.

### 4.2 Velocity grid schema

`velocity.npz` contains:

| Array | Shape | Meaning |
|---|---|---|
| `x_km` | `(Nx,)` | Strictly increasing, evenly spaced x axis |
| `y_km` | `(Ny,)` | Strictly increasing, evenly spaced y axis |
| `z_km` | `(Nz,)` | Strictly increasing, evenly spaced depth axis |
| `vp` | `(Nz,Ny,Nx)` | P velocity in km/s |
| `vs` | `(Nz,Ny,Nx)` | S velocity in km/s |

Each axis needs at least three nodes. Axis spacings may differ; the solver receives `(dz,dy,dx)`. Speeds must be finite and positive with `vp > vs`. Do not use longitude/latitude degrees as Cartesian distances or pass `(Nx,Ny,Nz)` data as `(Nz,Ny,Nx)`.

Project and interpolate geographic or irregular velocity data onto this grid first, then check missing values, boundaries, units, and resolution. Historical conversion examples remain in [step1.py](scripts/run_nonlinloc/step1.py) and [interp_velo.py](scripts/run_nonlinloc/interp_velo.py); their input formats and hardcoded paths require adaptation. See the [migration example](docs/MIGRATION.md#velocity) for converting existing `xyz_vp_vs.npy + axes_km.npz` products. A metadata JSON alone is not a complete velocity field.

### 4.3 Generate and train

Start with the fictitious grid:

```bash
python examples/make_demo.py --output work/demo

python -m bayesloc generate-fmm \
  --grid work/demo/velocity.npz \
  --geometry work/demo/geometry.json \
  --n-sources 200 --receivers-per-source 32 \
  --noise-s 0 --seed 1234 \
  --output work/train.fmm.npz

python -m bayesloc train \
  --pairs work/train.fmm.npz \
  --checkpoint work/models/time.fmm.pt \
  --epochs 20 --batch-size 128 --lr 1e-4 --device auto
```

For your region, replace the grid and geometry and design sufficient sampling coverage. The generator samples **distinct source nodes without replacement**, samples receivers for each source, and produces both P and S labels. `receiver_bounds_km` can restrict receivers to a surface layer or a range of depths. The bounds must contain enough actual grid nodes.

Labels are noise-free by default. `--noise-s` adds Gaussian perturbations with the same standard deviation to P and S. Coincident source–receiver pairs and pairs with negative perturbed travel times are skipped, so output size can be smaller than the requested product. No gross errors are added to the training set. Construct and document a separate contaminated test set for robustness experiments.

Generation solves approximately two full grids per source, sequentially. Memory and runtime depend on grid size; the final training pairs also occupy memory. There is no multiprocessing option in this new interface. Large production datasets may need chunking and caching. The historical `scripts/gen_data_v5.*.py` implementations retain source-group reuse, multiprocessing, and distance-weighted receiver selection as adaptation references.

The new generator sets a single source node to zero, enforcing `T(source)=0`. Historical generators define the zero interface around one negative voxel; their near-source discretization differs, so regenerated labels are not expected to match byte-for-byte. Check homogeneous analytic solutions, grid refinement, and an independent numerical reference. A small neural fitting error does not imply a small physical modeling error.

<a id="model"></a>
## 5. Training, validation, and checkpoint reuse

### 5.1 Common pair interface

You may generate `.npz` pairs directly without using CSV preparation:

| Key | Shape | Convention |
|---|---|---|
| `xr` | `(N,3)` | Receiver xyz in km |
| `xs` | `(N,3)` | Source xyz in km |
| `tp`, `ts` | Each `(N,)` | Travel-time labels in seconds; missing values are NaN or `-12345.0` |
| `event_id` | `(N,)`, string | Source event identifier; not a pickle-dependent object array |
| `geometry_json` | Scalar string | JSON containing the full projection and domain configuration |

Each row needs at least one phase. The training subset must contain labels for both P and S. Use `bayesloc.io.save_pairs` to write the file. Convert historical `-1` missing targets to NaN explicitly; the new loader does not treat arbitrary negative times as missing.

### 5.2 Network and optimization

The architecture matches the two-output `ckpt/time.v1.0.pt` network: `[receiver_xyz,source_xyz]` inputs, seven Tanh hidden layers with 256 units by default, and `[Tp,Ts] = Softplus(output) × 10`. Internally, the network divides the **already-kilometre** coordinates by 1000 for numerical scaling. This is not a metres-to-kilometres conversion. Training and inference must retain the same order and scaling.

The objective is masked MSE over all valid phase targets, in seconds squared. AdamW defaults to `lr=1e-4` and `weight_decay=0.01`. Layer width and batch size are starting settings, not universal choices. After changing hidden width, use the newly generated checkpoint rather than incompatible historical weights.

By default, an 80%/20% split groups rows by **distinct source coordinates**. Receivers belonging to the same source cannot cross the split; at least two sources are required. Real-data validation should additionally consider independent time periods, spatial blocks, or event families to reduce correlated leakage.

Provide a separate validation dataset with:

```bash
python -m bayesloc train \
  --pairs work/train.observed.npz \
  --validation work/validation.observed.npz \
  --checkpoint work/models/time.validated.pt \
  --epochs 20 --device auto
```

Training and validation must have identical geometry and no exactly shared source coordinates. The code checks these conditions; it cannot establish independence of nearby events or their underlying catalogs.

| Training artifact | Contents |
|---|---|
| Requested `.pt` | Weights with the lowest validation MSE, hidden width, geometry, selected epoch, provenance, and validation metrics |
| `.pt.history.json` | Training MSE and validation P/S MAE, RMSE, and label counts by epoch |
| `.pt.split.npz` | Training and validation row indices, referring to their respective files when validation is external |

Each `train` invocation starts from scratch and updates the requested checkpoint path. Use a new filename for each experiment. The unified trainer has no resume option yet. Historical `--resume` flags belong to the historical trainers and are not compatible with the new checkpoint format.

### 5.3 Independent evaluation

```bash
python -m bayesloc evaluate \
  --pairs work/test.independent.npz \
  --checkpoint work/models/time.validated.pt \
  --output work/test.metrics.json --device auto
```

`evaluate` reports phase-specific MAE/RMSE; it does not prove test-set independence. Keep a test set out of training, model selection, and QC threshold selection. Examine error by distance, depth, region, and receiver elevation rather than only reporting an average.

### 5.4 Checkpoint naming and applicability

| Existing filename | Historical workflow |
|---|---|
| `ckpt/time.v1.0.pt` | Supervised two-output travel-time surrogate trained on synthetic labels |
| `ckpt/time.real.v1.0.pt` | Observed-data two-output training variant |
| `ckpt/time.real.pnsn.v1.0.pt` | Four-branch observed-data variant |
| `ckpt/time.real.pnsn.switch.v1.0.pt` | Alternative four-branch training variant |
| `ckpt/time.v1.0.eikonal.pt` | Historical eikonal-related variant; inspect its training and forward definitions before reuse |

A filename identifies provenance, not demonstrated interchangeability. Match the region, velocity/data source, projection, phase definitions, network structure, and elevation treatment before reuse. These are not universal models for arbitrary regions.

Legacy checkpoints lack the full new geometry contract. For the compatible synthetic `time.v1.0.pt`, generate and verify the local configuration described in the [migration guide](docs/MIGRATION.md#checkpoint), then pass it explicitly. Its conservative source-depth limit is 0–50 km, with zero-depth receivers, matching the verified original sampling. This does not establish uniform accuracy throughout the bounding box. Keep model-specific coordinate metadata local rather than publishing it as an example.

<a id="location"></a>
## 6. Earthquake location: inputs, commands, and outputs

### 6.1 Location inputs differ from training labels

Station and pick tables use the schemas in Section 3. Location only requires these event columns:

```csv
event_id,reference_time
event-001,2020-01-01T00:00:00Z
```

Here `reference_time` is a time origin for the event: an approximate origin, a window start, or another nearby instant. No known source coordinates are required. The model uses:

```text
t_observed = arrival_time − reference_time
t_observed = t0 + T(receiver,source) + residual
origin_time = reference_time + posterior_mean(t0)
```

Relative location arrivals may therefore be negative, unlike supervised training travel times. Choose a reference near the arrivals to avoid loss of float32 precision with very large relative seconds.

**Provide event membership or candidate-window IDs.** The unified interface does not associate an entire continuous pick stream automatically or deduplicate events across overlapping windows. For direct-window candidates, organize each window as an `event_id`, then locate and screen it. Implement association and cross-window deduplication in the preparation layer when needed.

### 6.2 Run location

```bash
python -m bayesloc locate \
  --stations private_data/stations.csv \
  --events private_data/events.csv \
  --picks private_data/picks.csv \
  --checkpoint work/models/time.observed.pt \
  --mode student_t_z \
  --samples 4000 --burn 2000 --thin 2 --chains 3 \
  --pool-size 16 --refine-top 4 --init-steps 100 \
  --min-stations 3 --min-phases 4 \
  --seed 20260911 --device auto \
  --output work/location/run01
```

For the compatible legacy synthetic checkpoint, substitute:

```text
--checkpoint ckpt/time.v1.0.pt --geometry work/legacy_geometry.json
```

Generate that configuration using the migration instructions. Inputs must match this model's region, depth range, and receiver layer.

| Option | Meaning |
|---|---|
| `--mode gaussian` | Gaussian residuals, without heavy tails or explicit outlier indicators |
| `--mode student_t` | Student-t heavy-tailed residuals |
| `--mode student_t_z` | Student-t inlier component plus a broad Gaussian outlier component; default |
| `--samples` | Total iterations per chain, **including burn-in** |
| `--burn` | Discarded initial iterations; proposal adaptation is restricted to this period |
| `--thin` | Save one draw every this many post-burn iterations |
| `--chains` | Independently seeded chains initialized at the selected solution |
| `--nu` | Student-t degrees of freedom, default 4 |
| `--proposal-km` | Initial spatial random-walk scale, default 2 km |
| `--pool-size`, `--refine-top` | Candidate count and number of best candidates to refine |
| `--init-steps` | Optimization steps per refined initial candidate |
| `--min-stations`, `--min-phases` | Pre-location observation-count criteria, defaults 3 and 4 |

Each chain retains `ceil((samples-burn)/thin)` draws. The `4000/2000/2` example gives 1,000 per chain, or 3,000 across three chains. Minimum observation counts do not guarantee identifiability: poor azimuthal or depth coverage can still give bad locations.

Initialization combines station centroids, early arrivals, and P–S-informed candidates, then scores/refines them with the neural travel-time model and a robust objective. It does not use reference hypocenters as hidden truth. All chains start at each event's best-ranked candidate. Other ranked candidates are saved, but this **does not guarantee exploration of distant posterior modes**; different candidates may also converge to the same basin.

Spatial sampling uses a weak Gaussian prior truncated to the configured domain and MH random-walk proposals. Origin time and error variables use conditional updates. Out-of-domain proposals are rejected. See [ALGORITHM.md](docs/ALGORITHM.md) for the corrected update order and parameter definitions.

### 6.3 Output files

| File | Contents |
|---|---|
| `catalog.csv` | Posterior location/origin estimates, geographic coordinates, widths, and chain diagnostics |
| `chain_00.npz`, … | Retained draws, per-observation inlier probabilities, acceptance rates, and row mappings |
| `initialization.npz` | Ranked initial sources, time offsets, and robust scores |
| `pick_quality.csv` | Relative arrival and inlier probability for each actual observed phase |
| `rejected_events.csv` | Events failing count criteria, with reasons |
| `run.json` | Input hashes, arguments, geometry, seeds, package versions, and stage runtimes |

Main `catalog.csv` fields:

| Field | Meaning |
|---|---|
| `event_id`, `reference_time` | Original event ID and UTC time reference |
| `origin_time`, `t0_mean_s`, `t0_std_s` | Corrected origin time and posterior offset statistics |
| `longitude`, `latitude`, `depth_km` | Geographic coordinates of the inverse-projected xyz mean; depth equals mean z |
| `x_mean_km`, `y_mean_km`, `z_mean_km` | Posterior mean location |
| `std_x_km`, `std_y_km`, `std_z_km` | Posterior sample standard deviations, using the population convention |
| `q05_x_km`, `q95_x_km`, etc. | Marginal 5th and 95th percentiles |
| `width_x90_km`, `width_y90_km`, `width_z90_km` | `q95-q05`: **central 90% marginal full widths** |
| `width_h90_km` | `sqrt(width_x90² + width_y90²)`, a horizontal ranking summary |
| `n_stations`, `n_phases` | Effective station and P+S observation counts |
| `acceptance_rate` | Spatial MH acceptance over the complete chain, including burn, averaged across chains |
| `rhat_split_x/y/z/t0` | Classical split R-hat; blank for insufficient chains/draws, inf for zero within-chain variance |

The horizontal combination is not a 90% error radius. The widths are not the usual IQR (`q75-q25`). Classical split R-hat is not rank-normalized R-hat; ESS is not computed here. Use saved chains to inspect traces, effective sample sizes, initialization sensitivity, and mode exploration for substantive inference.

Read a single chain with:

```python
import numpy as np

chain = np.load("work/location/run01/chain_00.npz", allow_pickle=False)
xyz = chain["xs_samples"]       # (S,E,3), km
t0 = chain["t0_samples"]        # (S,E), seconds
ids = chain["event_id"]         # (E,), identifiers for the event axis
sigma_p = chain["sigma_p_samples"]  # (S,E), seconds; Student-t scale in that mode
row_event = chain["row_event_index"]      # (N,), event index for each event–receiver row
row_station = chain["row_station_index"]  # (N,), zero-based row in original stations.csv
```

`S` is retained draws per chain, `E` is located events, and `N` is event–receiver rows after pairing P/S. Use the saved IDs and indices rather than assuming a catalog's line order matches an array axis.

<a id="qc"></a>
## 7. Pick diagnostics and event filtering

### 7.1 Pick-level robustness

`student_t_z` jointly infers each observed phase's inlier membership and writes its posterior probability as `inlier_probability`. A small value favors the outlier component conditional on the current forward model, priors, and other observations. It is not the probability that an earthquake exists.

Location does not pre-delete arrivals by residual or automatically remove low-probability picks and rerun. Gaussian and Student-t-only modes have no binary indicator, so that CSV field is blank. If you implement hard pick rejection followed by relocation, record the rule, preserve the original input, and re-evaluate errors and coverage.

### 7.2 Event-level ranking

For each event:

```text
Wx = q95(x) − q05(x)
Wy = q95(y) − q05(y)
Wz = q95(z) − q05(z)
Wh = sqrt(Wx² + Wy²)

rh = rank_average(Wh) / N
rz = rank_average(Wz) / N
Q  = max(rh,rz)              # Lower is preferred.
```

Ranks use all input events with finite, nonnegative widths; `N` is their count. Ties receive average ranks. The maximum balances horizontal and depth constraints rather than rewarding a small width in only one direction. `Q` is a relative within-catalog ordering, **not absolute error, event authenticity probability, or a confidence level transferable between regions**.

For a fixed retained count:

```bash
python -m bayesloc filter \
  --catalog work/location/run01/catalog.csv \
  --top-k 7000 --output work/location/run01/selected.k7000.csv
```

Events are stably ordered by `(Q,event_id)`. At most `min(K,eligible_events)` are retained. The example K is illustrative; choose retention using your catalog size and independent validation.

Optionally apply absolute width and chain-diagnostic criteria before selecting the best K eligible events:

```bash
python -m bayesloc filter \
  --catalog work/location/run01/catalog.csv \
  --max-horizontal-km 10 --max-depth-km 15 --max-rhat 1.1 \
  --top-k 7000 --output work/location/run01/selected.checked.csv
```

The values `10 km / 15 km / 1.1` illustrate the options; they are not validated universal thresholds. Width limits apply to **full widths**. Scores are computed before these hard criteria and are not recalculated after exclusion. When a R-hat criterion is requested, missing/nonfinite diagnostics also fail it. Width-only ranking does not automatically establish convergence.

Each filtering run writes:

- The requested CSV: retained events, sorted by score and ID.
- `.audit.csv`: all inputs, scores, `retained` flags, and `rejection_reason`.
- `.qc.json`: criteria, counts, arguments, and input hash.

Without `--top-k`, all events satisfying the explicit criteria are retained. With no criteria, valid events are scored and ordered. Invalid widths receive `invalid_width`. The R-hat check covers x, y, z, and t0.

The mixture probabilities `pi_P/pi_S` are shared across events within a location run. Splitting a catalog into batches can change those posteriors. Independently calculated percentile scores also cannot simply be concatenated. For large catalogs, document batching, assess its effect on inference, and apply final QC ranking to the combined catalog.

### 7.3 Validate the selection

With synthetic truth, measure errors versus retention and actual interval coverage. With real reference catalogs, report reference-relative agreement rather than treating the reference as absolute truth. Method comparisons need identical inputs, matching rules, and retained counts, including accounting for failed/rejected events.

A narrow posterior can result from a wrong local mode, an unmixed chain, or forward-model bias. The current method can under-cover and miss distant modes. Width alone does not establish a trustworthy event; see [algorithm limitations](docs/ALGORITHM.md).

<a id="projection"></a>
## 8. Inverse projection to longitude/latitude

`locate` already writes geographic coordinates for the xyz posterior mean. To convert a projected catalog explicitly:

```bash
python -m bayesloc reproject \
  --catalog work/location/run01/catalog.csv \
  --geometry private_data/geometry.json \
  --output work/location/run01/catalog.geographic.csv
```

Input needs `x_mean_km,y_mean_km,z_mean_km`. Other columns are preserved and geographic columns are updated. **Use the same geometry as training/location.** The tool cannot infer a projection center or hidden offsets from x/y values alone.

Manual conversion and sample-level export:

```python
import json
import numpy as np
from bayesloc.io import transformers

meta = json.load(open("private_data/geometry.json", encoding="utf-8"))
fwd, inv = transformers(meta)

# Forward: WGS84 degrees → metres → kilometres.
x_m, y_m = fwd.transform(longitude_deg, latitude_deg)
x_km, y_km = x_m / 1000.0, y_m / 1000.0

# Inverse: kilometres → metres → WGS84 degrees.
# always_xy=True keeps longitude first.
longitude_deg, latitude_deg = inv.transform(x_km * 1000.0, y_km * 1000.0)

chain = np.load("work/location/run01/chain_00.npz", allow_pickle=False)
xyz = chain["xs_samples"]
shape = xyz.shape[:2]
flat = xyz.reshape(-1, 3)
lon, lat = inv.transform(flat[:, 0] * 1000, flat[:, 1] * 1000)
lon_samples = np.asarray(lon).reshape(shape)
lat_samples = np.asarray(lat).reshape(shape)
depth_samples_km = xyz[..., 2]
```

Projection does not change the vertical datum. Remove any legacy x/y offsets before inverse transformation. Use appropriate geodesic methods for geographic distances; a constant 111 km per degree is not a general conversion for both axes.

Inverse projection of the xyz mean is generally different from averaging individually transformed samples. For geographic uncertainty, transform samples before summarizing them rather than transforming two marginal-interval corners. Account for longitude wrapping if samples cross the antimeridian.

<a id="limitations"></a>
## 9. Troubleshooting and limitations

| Symptom | Check |
|---|---|
| `legacy checkpoint has no domain metadata` | Generate and verify the local geometry described in the migration guide; pass `--geometry` |
| Stations outside bounds | Longitude/latitude order, elevation units/sign, and receiver training support |
| Negative training times | Timezones, origin times, and event membership; do not hide the issue by taking absolute values |
| Unknown or duplicate pick keys | Fix identifier mapping or define an explicit duplicate-resolution policy |
| Pg/Pn/Sg/Sn or checkpoint shape error | Two-output versus four-branch definitions and network compatibility |
| Wrong FMM speed-array shape | `(Nz,Ny,Nx)` order, flattened row order, and m/s versus km/s |
| Depth at a bound or nearly stationary chains | Initialization, surrogate errors, station elevations, bounds, and proposal scales; thinning alone is not a fix |
| Reasonable acceptance but poor locations | Local modes, forward accuracy, label definitions, and network geometry; acceptance is not accuracy |
| Fewer than K retained events | Input count, invalid widths, and explicit width/R-hat criteria; inspect the audit CSV |
| Unavailable CUDA/MPS | A compatible PyTorch installation or `--device cpu` |
| Nonempty output directory | Use a separate directory for each model/configuration/run |

The unified workflow assumes grouped arrivals or candidate event windows. It does not provide waveform picking, complete automatic association, magnitude estimation, double-difference relocation, anisotropic velocity inversion, or four-branch travel-time modeling. Its posterior is conditional on the specified forward/error models and does not automatically include full velocity-structure uncertainty.

Do not publish real event/station coordinates, picks, identifiers, exact times, posterior clouds, or maps as examples. New examples use fictitious geometry centered on `(0°,0°)`. Keep authorized real inputs and all derived outputs under the ignored `private_data/` or `work/` directories. Real geographic output is needed for authorized local analysis; it must not be copied into public documentation or PR attachments. See [DATA_POLICY.md](DATA_POLICY.md) for history-cleanup scope and precautions against reintroducing old data.

<a id="agents"></a>
## 10. Working with a Coding agent

You can ask a Coding agent to modify this project. The modular code, explicit schemas, and [AGENTS.md](AGENTS.md) support changes to adapters, CLI options, training, initialization, QC, or export. Developers still need to review scientific assumptions and actual validation results.

Example request for an input adapter:

```text
Read README.md, AGENTS.md, docs/ALGORITHM.md, and docs/MIGRATION.md first.
Adapt my phase-file format to the stations/events/picks CSV contracts.
Preserve event/station mappings, specify timezone and elevation units,
and reject ambiguous duplicates. Test with fictitious coordinates and picks;
do not read or display actual event/station coordinates in public examples.
Preserve km coordinates, seconds, positive-down depth, and network input order.
Run the relevant tests and a small end-to-end example, then update the README.
```

Example request for larger FMM datasets:

```text
Add chunking/caching to the FMM generator for larger velocity grids.
Preserve (Nz,Ny,Nx) ordering and (dz,dy,dx) spacing, and record the source
boundary condition. Compare a fictitious homogeneous model with analytic
travel times and the current implementation. Do not overwrite previous runs.
Explain any changes to numerical labels or posterior inference and which
validation must be repeated.
```

| Module | Main responsibilities |
|---|---|
| [bayesloc/io.py](bayesloc/io.py) | CSV/NPZ contracts, projection, units, domain checks |
| [bayesloc/training.py](bayesloc/training.py) | Supervised training, source-grouped validation, checkpoint selection |
| [bayesloc/fmm.py](bayesloc/fmm.py) | Velocity grids and FMM label generation |
| [bayesloc/locator.py](bayesloc/locator.py) | Network and MH-within-Gibbs updates |
| [bayesloc/multistart.py](bayesloc/multistart.py) | Candidate generation, robust scoring, refinement |
| [bayesloc/workflow.py](bayesloc/workflow.py) | Location outputs, catalog screening, inverse projection |
| [bayesloc/cli.py](bayesloc/cli.py) | Commands and arguments |
| [tests/test_workflow.py](tests/test_workflow.py) | Numerical and interface regression tests |

<a id="developers"></a>
## 11. Developers

- **Yuqi Cai** — [caiyuqiming@foxmail.com](mailto:caiyuqiming@foxmail.com)
- **Xin Liu** — [xinliu_geo@outlook.com](mailto:xinliu_geo@outlook.com)
- **Ziye Yu** — [yuziye@cea-igp.ac.cn](mailto:yuziye@cea-igp.ac.cn)
- **Fan Xie** — [xiefan@outlook.com](mailto:xiefan@outlook.com)

## 12. Related work and licensing

Original method: [arXiv:2512.06407](https://arxiv.org/abs/2512.06407). This repository documents runnable algorithms and data interfaces. Consult the corresponding paper and formal reproduction materials for version-specific claims, comparisons, data permissions, and citation metadata. Demonstration runs are not evidence for paper-level conclusions.

This update preserves the repository's existing research-use licensing statement. Contact the developers about other permissions, redistribution, or commercial use. Copyright © 2025–2026, Ziye Yu. All rights reserved.
