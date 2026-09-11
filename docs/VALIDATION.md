# Validation record for this update

Date: 2026-09-11. New execution examples use fictitious inputs. No real-data location run was executed or displayed for workflow validation.

| Check | Result |
|---|---|
| `python -m pytest -q` | 14 tests passed after source-tree cleanup |
| Syntax of the maintained package, examples, and tests | Passed |
| Local links in README and supporting documentation | 36 targets checked after cleanup; none broken |
| Python code snippets in documentation | Syntax checked |
| `git diff --check` | Passed |
| CPU preparation, short supervised training, location, filtering, inverse projection | Passed |
| Small 3-D FMM label generation and one supervised training epoch | Passed |
| Apple MPS one-epoch training and short-chain location | Passed |
| CUDA execution on this machine | Not performed; no available CUDA device |

Test environment: Python 3.14, PyTorch 2.9.1, NumPy 2.3.4, SciPy 1.16.2, pyproj 3.7.2, scikit-fmm 2025.06.23, and pytest 9.0.2. Declared minimum dependencies do not imply that every version combination has been tested.

The source-tree cleanup reduced the tracked file count from 105 to 20 by archiving historical scripts, weights, and experiment products. No `bayesloc/` implementation file changed. The full CPU test suite was rerun after removal, including the CLI pipeline and FMM numerical checks. The legacy-checkpoint test now generates a small fictitious checkpoint and checks prediction preservation and projection mismatch rejection, so tests do not require an archived research weight file. Earlier standalone FMM-training and MPS checks in the table were not rerun for this structural change.

Numerical tests use an independent homogeneous analytic forward model to check a nonzero origin-time correction, all three residual modes, missing phases, and domain constraints. FMM tests cover zero source time, unequal axis spacings, and speed scaling. Short neural training/location runs validate interface execution, not useful checkpoint accuracy or converged inference.

The update does not retrain a complete regional model or reproduce the full paper experiments. History removal is checked separately from numerical validation; see [DATA_POLICY.md](../DATA_POLICY.md) for the difference between rewritten Git refs and external caches/clones.
