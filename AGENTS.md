# Instructions for Coding agents

Read `README.md`, `docs/ALGORITHM.md`, `docs/MIGRATION.md` and `DATA_POLICY.md`
before changing a workflow. The supported portable entry point is
`python -m bayesloc`. Keep implementation in `bayesloc/`, fictitious input
generators in `examples/`, tests in `tests/`, and supporting guides in `docs/`.
Historical experiment scripts and pretrained weights are no longer bundled.
Do not add parallel versioned entry-point scripts at the repository root.

## Scope and scientific contracts

- Make the requested change concrete and runnable. Keep README commands,
  argument names, schemas, metadata and tests synchronized with the code.
- Inputs to the neural network are receiver xyz followed by source xyz, in km.
  The historical division by 1000 is network scaling, not a units conversion.
  Outputs are `[Tp, Ts]` in seconds. Changing order/scaling requires a new
  checkpoint and validation; a state-dict shape match is not sufficient.
- AEQD uses WGS84, explicit lon0/lat0 and `always_xy=True`. Projection returns
  metres; divide by 1000 before the network, multiply by 1000 before inverse
  projection. Depth is positive down. Station z is `-elevation_m/1000`.
- Respect trained source AND receiver domains. Do not silently enlarge the
  configured region, force every station to zero depth or take absolute
  elevation values to make invalid input pass.
- Supervised data require known source coordinates and origin times. Location
  requires only grouped picks and a reference time. Do not use test truth or a
  reference catalog as a hidden initializer or tune QC on the evaluation set.
- Missing phases in sampler arrays use `-12345.0`. Training converts missing
  targets to NaN and masks them. Do not use zero as missing, propagate NaNs
  into the sampler, or silently discard malformed input and duplicate picks.
- FMM speed units are km/s, grid shape is `(Nz,Ny,Nx)`, spacing is `(dz,dy,dx)`.
  Preserve the documented source boundary condition, or explicitly version
  the change and validate its numerical consequences.
- The main network is supervised. Do not call it a PINN unless an actual
  physics loss is implemented and documented. Legacy class names are not
  evidence of a physics constraint.
- Preserve the corrected sampler ordering: update indicators, draw their
  conditional Student-t scales, recompute the current target, then MH.
  Include finite-variance outlier terms in the origin-time conditional.
  Reject out-of-domain proposals and stop adaptation before retained draws.
- Normal/outlier mixture probabilities are shared per phase across a run.
  Changes to event batching can change inference and must be documented.
- Width means `q95-q05`: central 90% marginal FULL width, not IQR, radius,
  joint credible region or calibrated error probability. QC is the maximum
  of average-rank percentiles of horizontal and depth widths. Stable ties use
  event ID. Preserve all-input accounting and rejection reasons.
- Multiple chains from one selected basin do not establish global posterior
  exploration. Classical split R-hat is not rank-normalized R-hat or ESS.
  Report checks performed and their limits, without inventing validation.

## Data privacy

The user explicitly requires that real longitude/latitude not be disclosed.

- Do not read, print, copy into examples, commit or attach real event/station
  coordinates or raw real-data rows when not necessary for the requested task.
  The restriction also covers station IDs, event IDs, exact times, geographic
  plots and posterior clouds that reveal the same restricted observations.
- All new public examples and tests must use fictitious data generated under
  `examples/`, centered at `(0,0)`. Do not substitute a real record and label it
  synthetic. Use summaries of schema/counts when inspecting restricted data.
- Keep authorized real input, derived catalogs, coordinate metadata and input
  fingerprints in `private_data/` or `work/`, both Git-ignored. Do not use
  `git add -f` for these paths. A generated location CSV is not a public demo.
- Do not copy files from sibling research/manuscript/release directories into
  this repository without checking whether they contain restricted material.
  Only needed source algorithms were ported for this update.
- Removing a file from the current tree does not purge old Git history.
  History rewrites need explicit authorization; the user has authorized the
  real-coordinate cleanup in this update. Preserve a private backup, audit
  all affected refs, and use explicit force-with-lease protections when
  publishing the rewrite. Never merge pre-cleanup history back into it.

## Verification and delivery

1. Install with `python -m pip install -e ".[fmm,test]"` in a suitable environment.
2. Run the affected numerical/interface tests with `python -m pytest -q`.
3. For workflow changes, run the fictitious example from README in a fresh
   `work/` output directory. A short training run is only a plumbing check.
4. For sampler changes, test a known analytic travel-time model with missing
   phases and a nonzero origin-time offset, not only a random neural model.
5. For FMM changes, compare against homogeneous analytic travel times and
   test unequal axis spacing. Verify source/receiver domain conventions.
6. Review `git diff --stat`, changed paths and public examples before delivery.
   Inspect sensitive-file changes by names/counts; do not paste deleted real
   rows into a user-visible report. Keep generated data and results untracked.
7. Report what changed, what ran successfully and what remains unvalidated.
   Do not start expensive full training, publish real data, or claim a full
   manuscript reproduction from a small smoke test.
