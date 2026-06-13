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

## Results

| Item | Value |
| --- | --- |
| `nns_version` | `13.0` |
| `schema_version` | `1` |
| Final entry count | TBD |
| Prior entry count | 2406 |

### Deterministic cache entries that changed

TBD

### Python parity fixes required

TBD

### ARMA nonseasonal nonlinear reconciliation

The suspected live mismatch (Python `[125.25, 107.75, 158.75, 213.66]` vs R
`[128.50, 113.50, 155.50, 213.66]`) was re-run against live vendored R NNS
13.0 before regeneration:

- Live R `NNS.ARMA(series, h = 4, seasonal.factor = FALSE, method = "nonlin")`
  on the 24-point AirPassengers-style series returned
  `128.5, 113.5, 155.5, 213.6666666667`.
- Python `nns_arma(series, h=4, seasonal_factor=False, method="nonlin")`
  returned `128.5, 113.5, 155.5, 213.66666667`.

Python matches live R 13.0 exactly; the previously reported divergent values
could not be reproduced with the current Python implementation and live
vendored R NNS 13.0.
