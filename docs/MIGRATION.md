# Migrating from the historical scripts

## 1. Entry-point mapping

| Historical workflow | New entry point | Main changes |
|---|---|---|
| `travel_time_train.v1.0.real.py` | `prepare` → `train` | Adapt PHA/station input to three CSV tables; specify known origins and elevation units |
| `travel_time_train.v1.0.py` | `train` | Convert block-format synthetic records to NPZ; explicitly replace missing -1 labels with NaN |
| `scripts/gen_data_v5.fortrain.py` | `generate-fmm` → `train` | Explicit grid contract, source sampling, different discrete source boundary, NPZ output |
| `location.v1.0.real.py`, `location.v1.0.crop.py` | `locate` | Grouped arrivals, explicit configuration, corrected sampler, CSV and chain outputs |
| `location.v1.0.pnsn.crop.py` and related variants | Historical four-branch workflow | Do not load four-output networks or branch labels into the two-output interface |
| Historical quality plots | `filter` | Export central 90% full widths and use a common ranking definition |
| Manual `proj(...,inverse=True)` | `reproject` | Explicit AEQD origin, metres/kilometres conversion, and any prior coordinate offsets |

The filenames above identify historical workflows, not files included in the current source tree. Versioned experiment scripts, comparison/plotting utilities, datasets, and pretrained weights have been removed from the maintained tree. The repository owner retained a private archive during cleanup. New installations need only `bayesloc/`, the fictitious examples, and the declared dependencies.

If you have an old checkout, adapt its authorized inputs locally rather than copying its scripts or data back into this repository. The old README's `python bayes_location.py` did not correspond to a valid current entry point and has been replaced with `python -m bayesloc ...`.

## 2. Different legacy text formats require different adapters

Historical scripts read several formats: REAL-like block catalogs, `#EVENT` PHA files, and grouped window picks. A shared extension does not imply a shared schema. Establish:

1. Station-ID construction, including network and punctuation.
2. Original elevation units, sign, and vertical datum.
3. Origin-time timezone and fractional-second representation.
4. Whether each phase time is absolute, relative to origin, or relative to a window start.
5. Longitude/latitude order, projection status, and coordinate offsets.
6. Duplicate handling and accounting for unrecognized records.

For its own historical synthetic format, `travel_time_train.v1.0.py` reads integer event headers and takes latitude/longitude/depth from whitespace columns 8/9/10, counting from one. In its `SC` phase rows, column 5 is relative travel time and the last two columns are station longitude/latitude. This is not a universal REAL/PHA parser.

The old generator writes textual timestamps, a numerical reference time, and absolute/relative arrivals. Verify the relative-time column before migrating labels; do not mix inconsistent example time headers. Test adapters with fictitious records instead of copying real rows into public source files.

The new interface does not guess elevation units, silently resolve duplicates, or merge phase branches. Some old workflows force receiver depth to zero while others use elevation. Confirm consistency with the training model; a readable old output is not proof that its coordinate conventions are correct.

<a id="velocity"></a>
## 3. Converting flattened velocity arrays

If your starting point is the five-column SWChinaCVM-V2.0 text model, use the [SWChinaCVM worked example](SWCHINA_CVM.md) and `build-grid`; the flattened-array adapter below is for already projected Cartesian products.

If you have:

- `xyz_vp_vs.npy`: `(N,5)` rows of `[x_km,y_km,z_km,vp,vs]`;
- `xyz_vp_vs_axes_km.npz`: `x_km,y_km,z_km` axes;

convert them locally as follows. The explicit row-order check prevents a blind reshape from assigning velocities to the wrong nodes:

```python
from pathlib import Path
import numpy as np
from bayesloc.fmm import read_grid

points = np.load("private_data/xyz_vp_vs.npy", allow_pickle=False)
axes = np.load("private_data/xyz_vp_vs_axes_km.npz", allow_pickle=False)
x, y, z = [axes[key] for key in ("x_km", "y_km", "z_km")]
zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
expected_xyz = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
if points.shape != (len(expected_xyz), 5):
    raise ValueError("flat velocity table shape does not match axes")
if not np.allclose(points[:, :3], expected_xyz, rtol=0, atol=1e-4):
    raise ValueError("flat rows are not ordered (z,y,x); reorder explicitly first")

# Speeds are assumed to be km/s. Divide by 1000 first if the source uses m/s.
shape = (len(z), len(y), len(x))
Path("work").mkdir(exist_ok=True)
np.savez_compressed(
    "work/velocity.npz", x_km=x, y_km=y, z_km=z,
    vp=points[:, 3].reshape(shape),
    vs=points[:, 4].reshape(shape),
)
read_grid("work/velocity.npz")
```

Use the original velocity model's projection metadata for `geometry.json`; do not guess it from a map. Source/receiver bounds must match intended training coverage and fit inside the grid. Bounds metadata cannot reconstruct a missing 3-D speed field; obtain an authorized physical model or create a synthetic one.

<a id="checkpoint"></a>
## 4. Legacy checkpoint compatibility

No pretrained weights are bundled. The locator can still load structurally compatible two-output checkpoints containing `model_state` and `model_hidden_dim`. Store an existing authorized model at, for example, `private_data/legacy/time.pt`. Legacy files lack complete source/receiver domains. Verify and supply those locally rather than substituting the fictitious `(0,0)` example projection.

For readers of older versions, these names describe historical model families:

| Historical filename | Workflow |
|---|---|
| `time.v1.0.pt` | Two-output surrogate trained on synthetic labels |
| `time.real.v1.0.pt` | Two-output observed-data model |
| `time.real.pnsn.v1.0.pt` | Four-branch observed-data model |
| `time.real.pnsn.switch.v1.0.pt` | Alternative four-branch model |
| `time.v1.0.eikonal.pt` | Eikonal-related experimental variant |

A filename is not a compatibility guarantee. Before reusing a two-output model:

1. Confirm `[receiver_xyz,source_xyz]` inputs, kilometre units, the `/1000` network scaling, and `[Tp,Ts]` outputs in seconds.
2. Obtain the original projection, source bounds, receiver bounds, and vertical datum from authorized training metadata. A projection center alone cannot establish a model's domain.
3. Save the full geometry contract described in the [README](../README.md#coordinates) as `private_data/legacy/geometry.json`. If the checkpoint contains `meta.projection_meta`, its `lon0` and `lat0` must agree with this file. Do not print or publish these regional values.
4. Validate on independent arrivals from the intended domain before using the model for location.

Pass the local paths to `evaluate` or `locate`:

```text
--checkpoint private_data/legacy/time.pt --geometry private_data/legacy/geometry.json
```

A deeper physical velocity grid does not mean the network was trained at every grid depth. A model trained with zero-depth receivers needs validation or retraining before using station elevations. If training provenance or domain bounds are unavailable, train a new model rather than inventing geometry.

The new FMM source boundary differs from the old generator, and the trainer adds grouped validation and a different checkpoint-selection rule. Retraining is not a byte-for-byte reproduction of historical weights; retain data versions, solver settings, and validation reports.

Four-output `pnsn` weights are incompatible. For `.pinn`/`.eikonal` variants, inspect architecture, scaling, features, and training objective and validate the forward model before substitution. New checkpoints do not contain the old optimizer state, so historical `--resume` cannot restore their training.

## 5. Adapting old location output

`filter` requires `event_id,width_x90_km,width_y90_km,width_z90_km`. `reproject` requires `x_mean_km,y_mean_km,z_mean_km`. Convert mixed comments, multiline event records, or additional truth rows into a one-event-per-row CSV first.

Some historical `location.v1.0.*.py` files write `#event_id,...` records with three `q95-q05` columns after the position standard deviations. Inspect the exact writer before mapping these to `width_*90_km`. Do not relabel standard deviations, 95% widths, or IQRs as central 90% full widths. NLLoc one-sigma values are not directly comparable to these full widths either.

Keep real converted catalogs in ignored local directories. Follow [DATA_POLICY.md](../DATA_POLICY.md) when migrating clones from before the history rewrite: merging old history can restore removed data.
