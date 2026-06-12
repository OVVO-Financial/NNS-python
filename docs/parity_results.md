# Parity Results

## Executive summary

R NNS 13.0 is now the release parity target for NNS Python because R NNS 13.0 is the tensorized architecture target. The earlier R NNS 12.1 cache has been superseded. NNS-core is v13.0.0 and remains the native C++ foundation for accelerated partial-moment routines; Python parity is still bounded by the committed tests and cache rather than a claim of full package equivalence.

During this retarget, cache generation was prepared against the vendored R NNS 13.0 source tarball committed under `tools/`. The local environment could not complete apt installation of R because Ubuntu package downloads were blocked by the proxy with HTTP 403 responses, so the committed cache metadata is retargeted to 13.0 but the full R-backed cache refresh must be rerun in an environment where apt/R package installation can complete.

Plot artifact policy is unchanged: parity tests compare returned values and do not adopt R graphics-device artifacts. See `docs/plot_parity_policy.md`.

## Expected verification commands

```bash
python -m pytest -q tests/invariants
PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity
PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity/test_original_*
ruff check .
mypy
python -m build
```

`python -m build` is a packaging check. If the local environment lacks build tooling and cannot install dependencies, record that as an environment limitation.

## R NNS 13.0 retarget notes

- Target version: R NNS 13.0.
- Superseded target: R NNS 12.1.
- Native foundation: NNS-core v13.0.0.
- Cache file: `tests/_r_cache.json`.
- Cache schema: version `1`.
- Cache entries: 2,406 keyed R result entries.
- Tarball used for retarget setup: vendored R NNS 13.0 source in `tools/`.

## Fixed behavior in this retarget

The first R NNS 13.0 root-cause fix is in the univariate `NNS.reg(..., multivariate.call = TRUE)` regression-point path used internally by ARMA. R NNS 13.0 appends the central regression point again when final endpoint points are consolidated. Python now preserves that weighting, which changes the airline nonseasonal nonlinear ARMA smoke forecast from the old Python value `[125.25, 107.75, 158.75, 213.6667]` to the R NNS 13.0 value `[128.5, 113.5, 155.5, 213.6667]`.

## Coverage boundaries

Full package parity is not claimed. The current evidence is bounded by:

- cache-backed tests in `tests/parity/`,
- invariant/API tests in `tests/invariants/`,
- original-test fixture adoption under `tests/parity/test_original_*`, and
- the committed R-cache contents.

Any cache miss under `PYNNS_R_CACHE_ONLY=1` remains a parity-data gap until the cache is regenerated with Rscript and installed R NNS 13.0.
