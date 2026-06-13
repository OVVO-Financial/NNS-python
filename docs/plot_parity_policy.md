# Plot and Graphics-Device Parity Policy

## Summary

Graphics-device artifacts are **intentionally not pixel-compared** in CI
parity. The parity suite validates the **returned values** of NNS functions,
never byte-/pixel-identical plot, PDF, or other graphics-device output.

This is a deliberate, permanent policy decision — not an unresolved migration
blocker. R plotting and Python plotting use different graphics stacks, and a
faithful value-level port does not require byte-identical (or pixel-identical)
plot artifacts.

## A visual plotting API now exists (`nns.plotting`)

The Python port now ships an **optional** plotting API in the `nns.plotting`
subpackage (`pip install ovvo-nns[plot]`). It is **color/element-faithful to R
but not pixel-diffed**: tests assert *artist colors and which element they sit
on*, never rendered images.

- matplotlib is an optional extra (`[project.optional-dependencies].plot`). The
  NNS core stays NumPy/SciPy-only; matplotlib is imported lazily inside each
  plot function and a clear `ImportError("install ovvo-nns[plot]")` is raised if
  it is absent. matplotlib is **never** imported at package top level.
- Each `plot_*` function takes an already-computed NNS result (or the same raw
  inputs) plus a keyword `ax=None`, returns the `Axes`/`Figure`, and **never**
  calls `plt.show()`. The compute functions' `plot=False` default behavior is
  untouched; plotting is a separate opt-in call.
- Colors are pinned in `nns.plotting.palette` to the exact R `grDevices` hex
  used by `tools/NNS/R/*.R`. R and matplotlib agree on `steelblue`/`red` but
  **disagree** on `green` (R `#00FF00` vs mpl `#008000`) and `grey` (R `#BEBEBE`
  vs mpl `#808080`); the palette pins those so the port stays faithful.
- Plotting tests live in `tests/plotting/`, run on the headless `Agg` backend,
  and assert `mcolors.to_hex(...)` of line/scatter/patch artists — **no**
  pixel/PDF comparison.

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

## Image comparison remains out of scope

Even though a first-class plotting API (`nns.plotting`) now exists, image or PDF
comparison is still **out of scope**. The API is validated by asserting artist
colors and the element each color sits on (faithful to R's `col=` usage), which
is sufficient for a value-level port. No PDF/image diffing is attempted.
