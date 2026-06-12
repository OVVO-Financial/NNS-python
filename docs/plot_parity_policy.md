# Plot and Graphics-Device Parity Policy

## Summary

Graphics-device artifacts are **intentionally not compared** in CI parity. The
parity suite validates the **returned values** of NNS functions, never the
generated plots, PDFs, or other graphics-device output.

This is a deliberate, permanent policy decision — not an unresolved migration
blocker. R plotting and Python plotting use different graphics stacks, and a
faithful value-level port does not require byte-identical (or pixel-identical)
plot artifacts.

## What is compared

- Numeric return values (scalars, vectors, matrices, nested result dicts) from
  every ported function, against committed R fixtures and the committed R cache
  (`tests/_r_cache.json`).
- Structural contracts (result keys, shapes, dtypes, finiteness) via the
  invariant suite.

## What is not compared

- `Rplots.pdf` and any other R graphics-device output.
- R `plot = TRUE` side effects (e.g. `NNS.copula(..., plot = TRUE)`,
  `NNS.part(..., plot = TRUE)`, regression/residual plots, `rgl::plot3d`
  3-D scatter overlays).
- Python plotting output. The Python port deliberately exposes computation, not
  a plotting API, so Python parity calls pass the R `plot = FALSE` equivalent
  and assert only on returned values.

When a ported function has an R `plot` argument, the Python API either omits the
argument entirely or treats plotting as out of scope; only the value-bearing
return is asserted in parity tests.

## Inventory of committed graphics artifacts

- `original_tests/testthat/Rplots.pdf` — produced by the upstream R `testthat`
  run as a side effect of `plot = TRUE` calls in the original R test files. It is
  inventoried here for completeness. It is **not** referenced by any Python
  test, is **not** compared in CI, and exists only as a historical artifact of
  the original R test harness. No CI step reads, regenerates, or diffs it.

A repository-wide check confirms no test under `tests/` references `Rplots.pdf`,
any `*.pdf`, `plot3d`, or `rgl`; the CI workflow
(`.github/workflows/native-backend-ci.yml`) runs only the invariant suite, the
cache-only parity suite, `ruff`, `mypy`, and `python -m build`.

## When (and only when) image comparison would be in scope

Image or PDF comparison would only be considered if and when the Python package
grows a real, first-class plotting API that needs validation. There is no such
API today. Until one exists, no PDF/image comparison is attempted, and adding
one is explicitly out of scope.
