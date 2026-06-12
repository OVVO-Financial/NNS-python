# Parity Status

## Current status

- The parity suite lives in `tests/parity/` and compares public PyNNS behavior to R NNS-compatible cached fixtures.
- CI-compatible parity runs use `PYNNS_R_CACHE_ONLY=1` and the committed `tests/_r_cache.json` cache.
- Native-vs-fallback coverage lives in `tests/invariants/test_native_original_src_coverage.py`.
- The PR #6 non-finite native-routing fix is preserved in `src/pynns/core.py` through `_native_safe(...)` checks in `lpm`, `upm`, `lpm_ratio`, and `upm_ratio`.

## Closed parity gaps (branch `close-all-parity-gaps`)

- `nns_boost` depth=None parity: resolved (seed-sensitivity triage; seed pinned in the parity test; seed-invariance regression guard added). The committed cache matches Python to ~3.5e-15.
- `NNS.copula(..., continuous=FALSE)` discrete mode: implemented in `pynns.nns_copula` and adopted against the R fixture.
- `NNS.copula` multivariate / three-column mode: implemented (2-D `(observations, variables)` matrix input, any column count `>= 2`) and adopted for continuous and discrete.
- `PM.matrix` R data-frame naming: optional `pm_matrix(..., names=[...])` echo added (NumPy-first; numeric arrays unchanged) with a parity test.
- Plot/graphics policy: formalized in `docs/plot_parity_policy.md`; graphics-device artifacts are never compared in CI.

## Skipped or deferred cases

- The only remaining parity skips are intentional live-R-only practical examples in `tests/parity/test_practical_examples.py`, which regenerate vignette-scale results from installed R NNS on demand rather than from the committed cache. They are not ordinary cache-backed parity coverage.
- Live R regeneration is not required in CI because many runners do not have `Rscript` or R NNS installed.
- Cache regeneration remains optional and developer-local via `scripts/regenerate_r_cache.py`.
- The NNS-python migration remains out of scope for this branch.

## Regression coverage

- `tests/invariants/test_native_original_src_coverage.py` verifies native smoke behavior only for symbols exported by the currently built optional extension.
- The same file verifies public fallback behavior when native is disabled or unavailable.
- Non-finite partial-moment inputs are covered by a focused regression that monkeypatches native dispatch and proves NaN inputs use the Python fallback.
