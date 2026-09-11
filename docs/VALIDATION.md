# Validation record for this update

Date: 2026-09-11. Automated tests use fictitious inputs. The public SWChinaCVM-V2.0 velocity table was additionally converted and used for a small FMM/training check in ignored local storage. No real station/event observation set was used or displayed for workflow validation.

| Check | Result |
|---|---|
| `python -m pytest -q` | 22 tests passed after adding velocity-model conversion |
| Syntax of the maintained package, examples, and tests | Passed |
| Local links in README and supporting documentation | 47 targets checked after adding the CVM guide; none broken |
| Python code snippets in documentation | Syntax checked |
| `git diff --check` | Passed |
| CPU preparation, short supervised training, location, filtering, inverse projection | Passed |
| Small 3-D FMM label generation and one supervised training epoch | Passed |
| SWChinaCVM-V2.0 → grid → FMM → source-disjoint train/test split → short training/evaluation | Passed locally; no observation data used |
| Apple MPS one-epoch training and short-chain location | Passed |
| CUDA execution on this machine | Not performed; no available CUDA device |

Test environment: Python 3.14, PyTorch 2.9.1, NumPy 2.3.4, SciPy 1.16.2, pyproj 3.7.2, scikit-fmm 2025.06.23, and pytest 9.0.2. Declared minimum dependencies do not imply that every version combination has been tested.

The earlier source-tree cleanup reduced the tracked file count from 105 to 20 by archiving historical scripts, weights, and experiment products. At that stage, no `bayesloc/` implementation file changed. The full CPU test suite was rerun after removal, including the CLI pipeline and FMM numerical checks. The legacy-checkpoint test now generates a small fictitious checkpoint and checks prediction preservation and projection mismatch rejection, so tests do not require an archived research weight file. Earlier standalone FMM-training and MPS checks in the table were not rerun for that structural change.

Numerical tests use an independent homogeneous analytic forward model to check a nonzero origin-time correction, all three residual modes, missing phases, and domain constraints. FMM tests cover zero source time, unequal axis spacings, and speed scaling. Short neural training/location runs validate interface execution, not useful checkpoint accuracy or converged inference.

The eight velocity-adapter tests use affine fictitious Vp/Vs fields to verify horizontal/depth interpolation and unequal axis spacing, invalid/duplicate/incomplete profiles, extrapolation rejection, input depth datum, grid-size limits, output protection, custom receiver depths, and projection consistency at the FMM handoff. Tests make no network requests and do not need the public model file.

The separate public-model smoke run used upstream commit `2d56970f4848e3184a5d6f23c39aedfe387f67c9`, the mean-sea-level table (36,512 rows, 2,282 horizontal profiles, 16 shared depth levels), and a `(9,41,41)` converted grid. It sampled 12 sources with 16 receivers each, reserved test sources before training, and ran one CPU epoch with hidden width 8 followed by held-out evaluation. This checks the data path only; those weights are not presented as an accurate regional travel-time model. Projection coordinates, converted arrays, labels, and checkpoints remain in ignored local directories. See the [worked example](SWCHINA_CVM.md).

The update does not retrain a complete regional model or reproduce the full paper experiments. History removal is checked separately from numerical validation; see [DATA_POLICY.md](../DATA_POLICY.md) for the difference between rewritten Git refs and external caches/clones.
