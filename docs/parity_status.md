# Parity Status

## Current target

R NNS 13.0 is the release parity target. R NNS 12.1 cache data has been superseded because R NNS 13.0 is the tensorized architecture target. NNS-core is v13.0.0 and is the native C++ foundation for the Python package.

## What this status does and does not claim

The project does not claim full package parity. Parity status is bounded by the committed tests and cache:

- `tests/_r_cache.json` for cache-only R result fixtures,
- `tests/parity/` for public behavior parity checks,
- `tests/invariants/` for Python-native contracts and invariants, and
- `tests/fixtures/original_tests_expected.json` for adopted original R tests.

Plot artifact policy remains unchanged: plots and `Rplots.pdf` artifacts are not parity outputs in pytest; returned values are.

## R NNS 13.0 cache

The parity cache metadata now records R NNS 13.0. The cache contains 2,406 keyed entries under schema version 1. Cache generation for this retarget used the vendored R NNS 13.0 source tarball during setup, but local R installation was blocked by apt proxy HTTP 403 responses; rerun `python scripts/regenerate_r_cache.py` in an environment with a working R NNS 13.0 installation to refresh every cached value from R.

## Known retarget fix

The univariate regression-point construction path now follows R NNS 13.0's central-point weighting when `multivariate_call=True`. This path is used by nonlinear ARMA. The airline nonseasonal nonlinear smoke case now matches the R NNS 13.0 target `[128.5, 113.5, 155.5, 213.6667]` instead of preserving the older Python/R-12.1-incompatible behavior.
