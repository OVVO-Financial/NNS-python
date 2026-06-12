# Parity Plan

## Target

Retarget Python parity to R NNS 13.0. R NNS 13.0 is the tensorized architecture target, and R NNS 12.1 cache data is superseded. NNS-core is v13.0.0 and remains the native C++ foundation.

## Plan

1. Install R and R dependencies.
2. Install R NNS 13.0 from the vendored package source under `tools/` (never from CRAN).
3. Confirm `packageVersion("NNS") == "13.0"`.
4. Validate the R NNS 13.0 smoke values for partial moments, copula, ARMA, regression points, PM matrix naming, and seeded stack behavior.
5. Regenerate `tests/_r_cache.json` with R NNS 13.0 metadata and values.
6. Run cache-only parity, capture the full failure inventory, and fix Python behavior to R NNS 13.0 without loosening tolerances.
7. Keep full parity claims bounded by tests and cache.
8. Keep plot artifact policy unchanged.

## Installing R NNS 13.0 from local source

The vendored R package source is committed in this repository, so NNS is installed
from local source, not CRAN:

- Extracted package directory: `tools/NNS` (`tools/NNS/DESCRIPTION` reports `Version: 13.0`).
- Vendored tarball: `tools/NNS_13.0.tar.gz`.

Install with the helper script (prefers `tools/NNS`, falls back to the tarball, and
verifies the loaded version):

```bash
python scripts/install_local_r_nns.py
```

Or run the exact command sequence directly:

```bash
R CMD INSTALL tools/NNS
Rscript -e "suppressPackageStartupMessages(library(NNS)); cat(as.character(packageVersion('NNS')))"
# expected output: 13.0
```

Do not run `install.packages("NNS")`; the parity target is the local `tools/NNS`
source, not the CRAN release.

## Regenerating the parity cache

After confirming `packageVersion("NNS") == "13.0"`, regenerate the committed cache
with cache-only/offline toggles unset:

```bash
unset PYNNS_R_CACHE_ONLY PYNNS_OFFLINE CI
python scripts/regenerate_r_cache.py -- -n 0 tests/parity
```

If full regeneration is slow or unstable, regenerate deterministic chunks one file
at a time, for example `python scripts/regenerate_r_cache.py -- -n 0 tests/parity/test_core.py`,
then continue through the remaining parity files. The committed result must remain a
single valid `tests/_r_cache.json` with `nns_version == "13.0"`, `schema_version == 1`,
and non-empty `entries`. `scripts/regenerate_r_cache.py` enforces those guardrails after
the pytest run.

Validate the regenerated cache offline:

```bash
PYNNS_R_CACHE_ONLY=1 python -m pytest -q -n 0 tests/parity
```

A `RuntimeError: R cache miss ...` means the cache is incomplete (regenerate the
missing live R entries); an `AssertionError`/numeric mismatch means Python behavior
differs from R NNS 13.0 and the Python implementation must be fixed without loosening
tolerances.

## Current retarget focus

The first fixed root cause is the `NNS.reg(..., multivariate.call = TRUE)` regression-point construction used by nonlinear ARMA. Python now preserves R NNS 13.0's duplicate central-point contribution during endpoint consolidation.

## Environment note

The committed `tests/_r_cache.json` carries `nns_version == "13.0"` and `schema_version == 1`
with non-empty `entries`, and the full cache-only parity suite passes against it. Where an R
toolchain is unavailable (for example, sandboxed CI or proxy-restricted runners that cannot
install R), the cache cannot be regenerated live; rerun the local-source install and
`scripts/regenerate_r_cache.py` on a host with R when refreshing the cache. Always install NNS
from `tools/NNS` (or `tools/NNS_13.0.tar.gz`), never from CRAN.
