# R NNS 13.0 parity cache regeneration provenance

This document records the fresh, from-empty regeneration of
`tests/_r_cache.json` against live vendored R NNS 13.0.

## Environment

| Item | Value |
| --- | --- |
| Date of fresh regeneration | 2026-06-13 (UTC) |
| OS | Ubuntu 24.04.4 LTS (Linux 6.18.5 x86_64) |
| Python | 3.11.15 |
| R | R version 4.3.3 (2024-02-29) "Angel Food Cake" |
| Rscript | Rscript (R) version 4.3.3 (2024-02-29) |
| Vendored NNS source | `tools/NNS` (extracted package directory, preferred over `tools/NNS_13.0.tar.gz`) |
| `packageVersion("NNS")` | `13.0` |

R NNS itself was installed exclusively from the vendored repository source via
`scripts/install_local_r_nns.py` (`R CMD INSTALL tools/NNS`), never from CRAN.
R package dependencies required to load vendored NNS (`data.table`,
`doParallel`, `foreach`, `Rcpp`, `RcppParallel`, `rgl`, `xts`, `zoo`,
`jsonlite`) were installed as Ubuntu binary packages; `Rfast` (plus its `zigg`
dependency) was built from the upstream GitHub release source
`RfastOfficial/Rfast` tag `v2.1.5.1-apollo` because CRAN was unreachable in
the regeneration environment. Only dependencies came from external archives;
NNS came from `tools/NNS`.

## Commands run

```bash
python scripts/install_local_r_nns.py
Rscript -e "suppressPackageStartupMessages(library(NNS)); cat(as.character(packageVersion('NNS')))"
# printed: 13.0

# Fresh regeneration: moves tests/_r_cache.json to tests/_r_cache.json.bak,
# starts from an EMPTY cache, and repopulates every entry with a live R call.
python scripts/regenerate_r_cache.py --fresh -- -n 0 tests/parity

# Replay checks after regeneration
PYNNS_R_CACHE_ONLY=1 python -m pytest -q -n 0 tests/parity
python -m pytest -q tests/parity/test_r13_smoke.py
python -m pytest -q tests/invariants
ruff check .
mypy
python -m build
```

All cache-only/offline toggles (`PYNNS_R_CACHE_ONLY`, `NNS_R_CACHE_ONLY`,
`PYNNS_OFFLINE`, `NNS_OFFLINE`, `CI`) were unset for the regeneration run.
The `--fresh` mode added to `scripts/regenerate_r_cache.py` in this change
refuses to run in CI, verifies a live local R NNS 13.0 install before touching
anything, then moves the existing cache aside so no existing entry can be
reused during regeneration.

The fresh regeneration was run twice: once to surface the true Python parity
gaps against live R 13.0 (216 test failures), and a second time from an empty
cache after the Python fixes below, confirming every parity test passes
against entries produced exclusively by live R calls.

## Results

| Item | Value |
| --- | --- |
| `nns_version` | `13.0` |
| `schema_version` | `1` |
| Final entry count | 2385 |
| Prior entry count | 2406 |

The entry count dropped from 2406 to 2385 because 21 entries in the old
committed cache are stale: an instrumented cache-only replay confirmed that no
current parity test computes those 21 keys, so a from-empty regeneration never
recreates them. All 2385 keys requested by the current test suite were
regenerated from live R.

### Deterministic cache entries that changed

456 of the 2384 shared keys changed value relative to the previously committed
cache, summarized by NNS function (mapped by instrumenting the cache-key
computation during a full replay):

| Function | Changed entries |
| --- | --- |
| `PM.matrix` | 216 |
| `NNS.reg` | 147 |
| `NNS.boost` (numeric harness) | 25 |
| `LPM.VaR` | 12 |
| `UPM.VaR` | 12 |
| `NNS.ANOVA` (custom harness) | 12 |
| `NNS.M.reg` | 10 |
| `NNS.stack` (numeric harness) | 6 |
| `NNS.dep` | 3 |
| `NNS.caus` | 3 |
| `NNS.moments` | 2 |
| `dy.d` (scalar harnesses) | 4 |
| factor-predictor harnesses (boost/stack/reg) | 4 |

The fresh live R 13.0 values were treated as authoritative in every case.

### Python parity fixes required

Fresh regeneration surfaced 216 failing parity tests. All of them traced to
four deterministic divergences between NNS Python and vendored R NNS 13.0, and
Python was fixed to match R in each case:

1. **`LPM.VaR` / `UPM.VaR` integer degrees 1–4** (`src/nns/var.py`):
   R 13.0 replaced the `optimize()` search with an exact polynomial
   root-finding inversion on the located order-statistic interval
   (`.NNS_LPM_VaR_integer`). Python now ports that algorithm (prefix power
   sums, break-ratio interval location, Brent root with
   `tol = .Machine$double.eps^0.5`, and the same boundary/fallback handling).
   This also fixed the `NNS.reg`/`NNS.M.reg` confidence intervals and
   stack/boost prediction intervals built on these helpers.
2. **Distance-kernel lognormal rank weight** (`src/nns/distance.py`):
   R 13.0's bulk path kernels (`NNS_distance_path_single_parallel_cpp`, used
   for fitted values when `n.best > 1` and for multi-point estimates) use the
   population sd of ranks `sqrt((k^2 - 1) / 12)` in the lognormal weight,
   while the single-point `NNS_distance_cpp` kernel keeps the sample sd.
   Python now provides `nns_distance_path_single_bulk` mirroring the bulk
   kernel and keeps `nns_distance` on the single-point formulation.
3. **`NNS.M.reg` out-of-hull multi-point extrapolation**
   (`src/nns/multivariate_regression.py`): R 13.0 vectorized the multi-point
   outsider path (bulk kernel estimates, `pmax(distance, 1e-10)` gradient
   guards) and removed the old dims-dropping behavior for a single outsider
   row that Python previously emulated. Python now mirrors the new path.
4. **`NNS.ARMA` numeric multi-lag seasonal weighting** (`src/nns/arma.py`):
   R weights each numeric seasonal factor by reversing the series with the
   factor's *position* in the `seasonal.factor` vector
   (`variable[seq(length(variable), 1, -i)]`), not its lag value. Python now
   matches, which resolved the two previously `xfail`-ed Sunspots ARMA and
   macro-like VAR practical examples; both now pass against live R and the
   `xfail` markers were removed.

Structural test updates justified by fresh R output: R 13.0's `NNS.boost`
returns only `results`, `pred.int`, `feature.weights`, and
`feature.frequency` (no `n.best`), so the Python return dictionary and the
test assertions that expected `n.best` were updated, and the NaN handling of
final boost estimates now matches R (`stack` falls back to `reg` only when
absent; NaNs are filled with the gravity of the remaining estimates).

The only remaining `xfail` is the balanced Iris `NNS.boost` diagnostic, which
is documented as a stochastic sampling gap (R RNG-driven CV-index, feature
subset, and up/down-sampling draws cannot be reproduced bit-for-bit with
NumPy's RNG). No deterministic parity gap is excluded from the suite.

### Manual comparison harness

No `scripts/compare_nns.py` / `scripts/compare_nns_r13.py` diagnostic scripts
exist in this repository, so there was no `$RPM`-based univariate `NNS.reg`
extraction to fix. The univariate regression-point diagnostic lives in
`tests/parity/test_r13_smoke.py` and already extracts regression points via
`NNS.reg(..., multivariate.call = TRUE)$y`, matching the supported R-side
extraction.

### ARMA nonseasonal nonlinear reconciliation

The suspected live mismatch (Python `[125.25, 107.75, 158.75, 213.66]` vs R
`[128.50, 113.50, 155.50, 213.66]`) was re-run against live vendored R NNS
13.0 before regeneration:

- Live R `NNS.ARMA(series, h = 4, seasonal.factor = FALSE, method = "nonlin")`
  on the 24-point AirPassengers-style series returned
  `128.5, 113.5, 155.5, 213.6666666667`.
- Python `nns_arma(series, h=4, seasonal_factor=False, method="nonlin")`
  returned `128.5, 113.5, 155.5, 213.66666667`.

Python matches live R 13.0 exactly; the previously reported divergent Python
values could not be reproduced with the current implementation. The hardcoded
smoke expectations in `tests/parity/test_r13_smoke.py` (LPM/UPM, copulas,
regression points, seasonal ARMA, nonseasonal nonlinear ARMA, seeded stack)
were each verified against this fresh live R 13.0 run and required no changes.
