````markdown
# Bayesian Earthquake Location with a Neural Travel-Time Surrogate

**Code for:**  
**“Bayesian Earthquake Location with a Neural Travel-Time Surrogate: Fast, Robust, and Fully Probabilistic Inference in 3-D Media”**  
Yu Ziye et al., 2025  
arXiv: [2512.06407](https://arxiv.org/abs/2512.06407)

**Copyright:** © 2025, yuziye@cea-igp.ac.cn. All rights reserved.

---

## 1. Overview

This repository contains a reference implementation of the Bayesian earthquake-location framework described in the paper:

> *Bayesian Earthquake Location with a Neural Travel-Time Surrogate: Fast, Robust, and Fully Probabilistic Inference in 3-D Media* (arXiv:2512.06407).

The key idea is to **replace numerical ray tracing** in a 3-D heterogeneous velocity model with a **neural travel-time surrogate**, and to embed this fast forward model inside a **Bayesian sampler**. This yields:

- **Fast**: orders-of-magnitude speedup in forward evaluations.
- **Robust**: optional heavy-tailed, Student-t-like modeling of outliers.
- **Fully probabilistic**: full posterior samples of hypocenter, origin time, and phase-specific uncertainties.

This code focuses on **3-D regional earthquake location** using **P and S first-arrival times**, with:

- A pre-trained neural travel-time model `PINNTravelTime`.
- A **multi-event Gibbs + random-walk Metropolis–Hastings sampler**.
- Support for **missing phases** and **per-event noise variances**.
- Optional **GMM-based spatial prior** (through `make_log_prior_xs`).

---

## 2. Main Script and Functionality

The main script is:

- `mcmc_speedup_gmm_mask_nuts_clean.py`

Core components:

- **Neural travel-time surrogate**
  - `PINNTravelTime` loaded from `ckpt/travel_time.ps.v2.pth`.
  - Evaluated via:
    ```python
    forward_tps(xs: Tensor, xr: Tensor) -> Tensor
    forward_tps_multi(xs_batch: Tensor, xr: Tensor, event_ids: Tensor) -> Tensor
    ```
    Returns predicted `(Tp, Ts)` for arbitrary source–receiver combinations in a 3-D velocity model.

- **Likelihood and masking**
  - Supports missing data using a special value:
    ```python
    MISSING_VAL = -12345.0
    ```
  - Masks P/S observations event-by-event:
    ```python
    build_phase_masks(...)
    log_lik_per_event(...)
    ```

- **Bayesian sampler**
  - Main entry point:
    ```python
    gibbs_mh_location_multi(...)
    ```
  - For **NC events** and **N total observations**, jointly samples:
    - Source locations: `xs ∈ ℝ^{NC×3}`
    - Origin times: `t0 ∈ ℝ^{NC}`
    - Phase standard deviations: `σ_P`, `σ_S` per event
  - Uses:
    - Random-walk Metropolis–Hastings in location space.
    - Gibbs updates for `t0`, `σ_P²`, `σ_S²` (Inverse-Gamma priors).
    - Optional Student-t scale-mixture via per-observation latent weights `λ`.

- **Prior on locations**
  - `make_log_prior_xs(...)` supports:
    - Wide 3-D Gaussian prior.
    - Optional **GMM-based prior** (if a GMM is fitted externally and passed in).
    - Or a **mixture** of both.

- **Input / output wrapper**
  - `read_station_file(...)` parses station metadata.
  - `read_event_file(...)` parses an event catalog with phases.
  - The `__main__` block:
    - Reads events and stations.
    - Builds the concatenated vectors `Tp_obs`, `Ts_obs`, `xr`, and `event_ids`.
    - Runs the sampler.
    - Writes relocated results and uncertainties to:
      ```text
      odata/reloc.raw.v3.2.txt
      ```

---

## 3. Data and File Structure

The example script expects the following files:

- **Neural travel-time model checkpoint**
  - `ckpt/travel_time.ps.v2.pth`  
    Pre-trained `PINNTravelTime` model (PyTorch) for the target 3-D velocity model.

- **Velocity model & projection**
  - `data/vel.py.model`  
    A pickle file containing:
    ```python
    grid_vp, grid_vs, grid_x, grid_y, grid_z, proj, *rest = pickle.load(...)
    ```
    - `proj`: a cartographic projection object with a **callable interface**:
      - `x, y = proj(lon, lat)`
      - `lon, lat = proj(x, y, inverse=True)`

- **Station file**
  - `ayrdata/china.loc`  
    Plain-text file with (per line, whitespace-separated):
    ```text
    NETWORK STATION_CODE NETWORK_ID LON LAT ELEV(m) ...
    ```
    Example:
    ```text
    SC JLO 0001 101.8684 29.8570 1234 ...
    ```
    The reader builds a station dictionary:
    ```python
    stations["SCJLO"] = np.array([lon, lat, -elev_km])
    ```

- **Event / phase file**
  - `data/Real.txt`  
    A NLLoc-style catalog with:
    - Header lines starting with an integer event ID, followed by:
      - Origin time (year, month, day, hour, minute, second)
      - Julian day or base time
      - Hypocenter (lat, lon, depth)
      - Magnitudes, etc.
    - Followed by phase lines:
      ```text
      NET STATION PHASE TRAVELTIME WEIGHT ...
      ```

  The parser:
  - Extracts event origin time, latitude, longitude, depth, and magnitudes.
  - For each event:
    - Builds a receiver array `rcv` with 3-D projected station coordinates.
    - Fills `T_p`, `T_s` vectors with P/S travel times, using `MISSING_VAL` for missing phases.

---

## 4. Installation and Requirements

### 4.1. Dependencies

- Python 3.9+ (recommended)
- [PyTorch](https://pytorch.org/) (CPU or GPU)
- NumPy
- tqdm (for progress bars)
- Your own projection object (e.g. `pyproj`) inside `vel.py.model`

Install the basic dependencies with:

```bash
pip install torch numpy tqdm
````

(You may need a specific PyTorch build for CUDA / MPS depending on your hardware.)

### 4.2. Clone and setup

```bash
git clone <this-repo-url>.git
cd <this-repo-name>
```

Place or link the required data:

```text
ckpt/travel_time.ps.v2.pth
data/vel.py.model
data/Real.txt
ayrdata/china.loc
```

---

## 5. Quick Start

Once all required files are in place, simply run:

```bash
python mcmc_speedup_gmm_mask_nuts_clean.py
```

The script will:

1. Read the velocity model and projection from `data/vel.py.model`.
2. Read stations from `ayrdata/china.loc`.
3. Read and parse events and phases from `data/Real.txt`.
4. Construct concatenated arrays `Tp_obs`, `Ts_obs`, `xr`, and `event_ids`.
5. Run the multi-event Gibbs+MH sampler:

   ```python
   out = gibbs_mh_location_multi(
       Tp_obs,
       Ts_obs,
       xr,
       event_ids=event_ids,
       NC=num_events,
       n_samples=4000,
       burn=2000,
       thin=2,
       gmm_prior=None,
       prior_mix=None,
       prop_scale=2.0,
       device_=device,
       dtype=dtype,
       verbose=True,
       alpha0=alpha0,
       beta0=beta0,
       use_student_t=False,  # set True to enable heavy-tailed likelihood
   )
   ```
6. Save relocated events and uncertainties to:

   ```text
   odata/reloc.raw.v3.2.txt
   ```

---

## 6. Output Format

The main output file:

* `odata/reloc.raw.v3.2.txt`

For each event, the script writes:

1. A summary header line:

   ```text
   #EVENT,<origin_time_with_t0_correction>,<lon>,<lat>,<depth>,
          <x>,<y>,<z>,
          <std_x>,<std_y>,<std_z>,
          <IQR_x>,<IQR_y>,<IQR_z>
   ```

   Where:

   * `<origin_time_with_t0_correction>` = catalog origin time + posterior mean `t0`.
   * `<x>,<y>,<z>` = posterior mean source location in projected coordinates.
   * `<std_*>` = posterior standard deviation of each coordinate.
   * `<IQR_*>` = width of the 5–95% interval for each coordinate.

2. The original catalog location (for comparison):

   ```text
   <x_cat>,<y_cat>,<z_cat>,<lon_cat>,<lat_cat>,<z_cat>,<mag1>,<mag2>
   ```

3. 95% credible intervals for each coordinate:

   ```text
    x: [lo_x, hi_x], range: hi_x - lo_x
    y: [lo_y, hi_y], range: hi_y - hi_y
    z: [lo_z, hi_z], range: hi_z - lo_z
   ```

This file can be post-processed to:

* Plot posterior clouds and credible ellipsoids.
* Compare Bayesian locations and catalog locations.
* Compute magnitude-dependent error statistics, etc.

---

## 7. Advanced Options

You can customize:

* **Number of samples / burn-in / thinning**

  ```python
  n_samples=4000, burn=2000, thin=2
  ```

* **Student-t robustness**

  * Set `use_student_t=True` to activate the scale-mixture updates for per-observation `λ`, down-weighting outliers.

* **Location prior**

  * Use a wide Gaussian only:

    ```python
    gmm_prior=None
    prior_mix=None
    wide_sigma=1000.0  # km in projected coordinates
    ```
  * Or include a GMM prior (fitted externally from a catalog) by passing a dictionary:

    ```python
    gmm_prior = {
        "w": ...,        # (K,)
        "mu": ...,       # (K,3)
        "L": ...,        # (K,3,3) Cholesky factors
        "log_det": ...,  # (K,)
    }
    prior_mix = 0.5
    ```

* **RW step-size adaptation**

  * Controlled by:

    ```python
    adapt_steps=1000
    target_accept_rw=0.3
    adapt_eta=0.05
    prop_scale=2.0
    ```

---

## 8. Citation

If you use this code or ideas from this work in your research, please cite:

```bibtex
@article{Yu2025BayesNeuralTTLocation,
  title   = {Bayesian Earthquake Location with a Neural Travel-Time Surrogate: Fast, Robust, and Fully Probabilistic Inference in 3-D Media},
  author  = {Yu, Ziye and coauthors},
  journal = {arXiv preprint},
  eprint  = {2512.06407},
  archivePrefix = {arXiv},
  primaryClass  = {physics.geo-ph},
  year    = {2025}
}
```

---

## 9. License and Copyright

All code in this repository is provided **for research use only** unless otherwise stated.

* **Copyright:**
  © 2025, **Yu Ziye** ([yuziye@cea-igp.ac.cn](mailto:yuziye@cea-igp.ac.cn)). All rights reserved.

Please contact the author for questions about licensing, redistribution, or commercial use.

---

## 10. Contact

For questions, bug reports, or collaboration inquiries, please contact:

* **Email:** [yuziye@cea-igp.ac.cn](mailto:yuziye@cea-igp.ac.cn)
* **Affiliation:** Institute of Geophysics, China Earthquake Administration (CEA-IGP)

```
::contentReference[oaicite:0]{index=0}
```
