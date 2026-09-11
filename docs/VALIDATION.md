# Validation record for this update

Date: 2026-09-11. New execution examples use fictitious inputs. No real-data location run was executed or displayed for workflow validation.

| Check | Result |
|---|---|
| `python -m pytest -q` | 14 tests passed after the English documentation update |
| Syntax of new modules and edited historical scripts | Passed |
| Local links in README and supporting documentation | No broken targets after translation |
| Python code snippets in documentation | Syntax checked |
| `git diff --check` | Passed |
| CPU preparation, short supervised training, location, filtering, inverse projection | Passed |
| Small 3-D FMM label generation and one supervised training epoch | Passed |
| Apple MPS one-epoch training and short-chain location | Passed |
| CUDA execution on this machine | Not performed; no available CUDA device |

Test environment: Python 3.14, PyTorch 2.9.1, NumPy 2.3.4, SciPy 1.16.2, pyproj 3.7.2, scikit-fmm 2025.06.23, and pytest 9.0.2. Declared minimum dependencies do not imply that every version combination has been tested.

Numerical tests use an independent homogeneous analytic forward model to check a nonzero origin-time correction, all three residual modes, missing phases, and domain constraints. FMM tests cover zero source time, unequal axis spacings, and speed scaling. Short neural training/location runs validate interface execution, not useful checkpoint accuracy or converged inference.

The update does not retrain a complete regional model or reproduce the full paper experiments. History removal is checked separately from numerical validation; see [DATA_POLICY.md](../DATA_POLICY.md) for the difference between rewritten Git refs and external caches/clones.
