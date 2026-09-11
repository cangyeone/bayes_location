# Algorithms, provenance, and validation scope

## 1. Source of the updated implementation

The sampler and initializer were ported from the corrected research workflow:

| New file | Source | Treatment in this update |
|---|---|---|
| `bayesloc/locator.py` | `robust_bayes_location_public/revision_experiments/locator.py` | Preserve probability updates; add interface validation |
| `bayesloc/multistart.py` | `multistart.py` in that directory | Preserve hybrid candidates and robust refinement; clarify that different ranks need not be distinct basins |
| `bayesloc/training.py` | `travel_time_train.v1.0.py` and the public supervised training workflow | Preserve network architecture and masked MSE; add source-grouped validation and best-validation checkpoint selection |
| `bayesloc/fmm.py` | Offline-label approach in `run_fm3d/gen_data_v5.fortrain.py` | Add a portable NPZ interface, unequal axis spacings, and a single-zero-node source condition |
| `bayesloc/workflow.py` | Corrected catalog output and `evaluation.py` ranking definitions | Add explicit CSV contracts, thresholds, and complete selection accounting |

These source paths document provenance; installation does not depend on sibling research directories. Real research inputs, location products, and manuscript maps were not copied into the new workflow. Installing corrected code does not retroactively validate old numerical outputs.

## 2. Supervised travel-time surrogate

For receiver `r` and source `s`:

```text
f(r,s) = 10 × Softplus(MLP([r,s] / 1000))
r and s are in km; outputs are [Tp,Ts] in seconds.
```

The MLP has seven fully connected Tanh hidden layers, default width 256. Historical state-dict names `net_merge.*` preserve structural compatibility with the corresponding two-output weights.

Training minimizes squared errors over valid labels. FMM labels come from offline numerical propagation; observed labels come from arrival time minus a known origin time. Both routes use the same supervised objective, without velocity-field inversion or an eikonal-gradient loss during network training.

Grouping by source avoids splitting receivers from the same source between training and validation. It does not remove correlations from nearby events, repeated catalogs, or shared processing. An independent test design is still required.

## 3. Residual distributions

For event `e`, observation row `i`, and phase `p`:

```text
r_i = t_observed_i − t0_e − T_p(receiver_i,source_e)
```

Supported modes:

- `gaussian`: `r_i ~ Normal(0,sigma²_ep)`.
- `student_t`: introduce `lambda_i ~ Gamma(nu/2,rate=nu/2)`, with conditional residual distribution `Normal(0,sigma²_ep/lambda_i)`. Marginalizing lambda gives Student-t residuals.
- `student_t_z`: additionally use `z_i ~ Bernoulli(pi_p)` to distinguish the Student-t inlier component from a broad `Normal(0,sigma²_out,p)` outlier component.

The latent indicator `z_i` is distinct from the source's depth coordinate z. `pi_P` and `pi_S` are each shared across the catalog in one run; `sigma_ep` varies by event and phase.

Defaults are `nu=4`, `sigma²_ep ~ InverseGamma(alpha=3,beta=0.5)`, initial phase scales of 0.5 s, outlier P/S standard deviations of 15/20 s, and `pi_p ~ Beta(8,2)`. These are implementation settings, not calibrated uncertainties for an arbitrary picker. The CLI exposes mode, nu, and proposal scale. Other priors can be set through Python `run_sampler`; record those settings yourself when using the direct API.

For Student-t residuals, sigma is a scale parameter. When `nu>2`, the inlier marginal standard deviation is `sigma*sqrt(nu/(nu-2))`. It is neither location-posterior spread nor the overall standard deviation of all mixture residuals.

## 4. Consistent MH-within-Gibbs updates

Each iteration updates the indicators using the Student-t density with lambda integrated out, then samples lambda conditional on the new inlier set. Latent updates change the target, so the current state's log density is recomputed before the MH comparison.

The spatial prior is a broad Gaussian truncated to `source_bounds_km`, with proportional density `exp(-||s||²/(2×1000²))`. Spatial proposals use a symmetric random walk. Out-of-domain proposals are rejected without evaluating the surrogate outside its configured domain. Proposal adaptation stops at the end of burn-in.

The outlier residual distribution is still centered on the predicted arrival and therefore depends on t0. Its finite precision must contribute to the origin-time conditional:

```text
Inlier precision:  w_i = lambda_i / sigma²_ep
Outlier precision: w_i = 1 / sigma²_out,p

Var(t0_e | rest)  = 1 / sum_i(w_i)
Mean(t0_e | rest) = sum_i(w_i*(t_observed_i − T_i)) / sum_i(w_i)
```

Gaussian mode uses `lambda=1`. The origin-time prior is flat. Inverse-Gamma variance updates use the inlier **observation count** and lambda-weighted squared residuals; the shape parameter must not substitute the sum of weights for the observation count.

These corrections are preserved in the new interface. Historical `location.v1.0.*.py` outputs should not be described as results from this corrected sampler.

## 5. Initialization and independent chains

Hybrid candidates use station centroids, early arrivals, P–S-informed distance rings, and centroid perturbations. The reference P/S speeds used to construct candidate rings are initialization heuristics, not the final constant-velocity forward model. Candidates are scored and refined with the current neural surrogate.

Independent seeds drive multiple chains from the best selected candidate per event. Summaries pool those chains and mainly describe inference within that selected basin. Saved alternative candidates support later initialization-sensitivity checks; the current workflow does not deduplicate basins, guarantee mode mixing, or estimate evidence weights for different modes.

## 6. QC and probability interpretation

QC uses the maximum of percentile ranks for horizontal and depth widths. It provides relative retention priority within a catalog. Pick-level inlier probabilities, posterior location widths, and event-ranking scores are different quantities.

The corrected research experiments motivate examining uncertainty/error relationships, while also showing limitations in interval coverage and distant-mode exploration. Central 90% marginal intervals are not claimed to have calibrated empirical coverage everywhere. Real reference-catalog agreement is not absolute ground-truth accuracy.

## 7. Verification limits

Tests cover projection roundtrips and signed elevation, UTC conversion, duplicate/unknown picks, masked gradients, source-grouped splits, QC ties and thresholds, split R-hat response to separated chains, nonzero origin-time recovery under three residual models with analytic homogeneous travel times, missing phases, domain constraints, FMM spacing/velocity scaling, and a short end-to-end CLI run.

Small examples do not reproduce a complete regional training campaign or paper comparison. The CUDA path comes from the accelerator-aware corrected implementation; a machine without CUDA cannot establish local CUDA validation. See [VALIDATION.md](VALIDATION.md) for what was actually executed.
