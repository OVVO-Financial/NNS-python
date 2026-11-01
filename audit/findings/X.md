# X (cross-cutting, orchestrator) findings

### X-01: `import nns; nns.nns_reg` pulls in scipy.interpolate eagerly (~0.40 s of a 0.47 s import)
- Type: perf
- Severity: low
- Confidence: verified
- Where: src/nns/smoothing.py:8 (`from scipy.interpolate import BSpline` at module top), imported by nns/regression.py
- Evidence: `python -X importtime -c "import nns; nns.nns_reg"`: nns.smoothing cumulative 402-460 ms, almost all scipy.interpolate; only `smooth=True` needs BSpline.
- Proposed fix: import BSpline inside the smoothing function. No result change.

### X-02: Commits authored with a private email are rejected by GitHub for this account
- Type: process
- Severity: low
- Confidence: verified
- Evidence: push of commits authored `rasheed.khoshnaw@gmail.com` rejected with "push declined due to email privacy restrictions"; commits now use the GitHub noreply address 99349713+gitRasheed@users.noreply.github.com.

### X-03: CI installs unpinned ruff/mypy although pyproject says the dev pins are "the exact versions CI gates on"
- Type: ci
- Severity: low
- Confidence: verified
- Where: .github/workflows/native-backend-ci.yml:39, inspect-r-api-update.yml:165 (`pip install ... ruff mypy`); pyproject.toml dev group pins mypy==2.3.1, ruff==0.16.3 with a comment that they match CI.
- Proposed fix: install `ruff==0.16.3 mypy==2.3.1` in CI (or `pip install --group dev`), so a new upstream release cannot break CI or disagree with local runs.

### X-04: docs dependency group allows MkDocs 2.0, which mkdocs-material warns is incompatible
- Type: docs/packaging
- Severity: low
- Confidence: verified
- Where: pyproject.toml [dependency-groups].docs (`mkdocs-material>=9.5`); `mkdocs build --strict` prints the MkDocs 2.0 incompatibility warning (build still succeeds today).
- Proposed fix: pin `mkdocs<2` in the docs group.

### X-05: `plot=` is inconsistent across the public API
- Type: api
- Severity: low-medium
- Confidence: verified
- Evidence: plot=True works for nns_reg (univariate), nns_cdf, nns_seas, nns_arma, nns_arma_optim. It raises TypeError
  (unexpected keyword) for nns_part, fsd, ssd, tsd, nns_anova, nns_norm, nns_diff, nns_copula, although R has `plot`
  on all of them and nns.plotting already has plot_nns_part/plot_fsd/plot_ssd/plot_tsd/plot_nns_anova/plot_nns_norm/
  plot_nns_diff/plot_nns_copula. Multivariate nns_reg(plot=True) and nns_sd_cluster(dendrogram=True) silently draw nothing.
- Proposed fix: add `plot: bool = False` to those functions delegating to the existing nns.plotting function (default
  False keeps Python's "no side effects" convention, documented divergence from R's plot=TRUE defaults); warn when a
  requested plot is not supported (multivariate reg with >2 regressors, dendrogram).

### X-06: `dy.dx(x, y)` / `dy_dx(x, y)` with the default eval.point = NULL always crashes (R and Python)
- Type: r-bug (Python copies the crash and R's internal message)
- Severity: medium
- Confidence: verified
- Evidence: R `dy.dx(rnorm(60), sin(x))` -> "Error in `[.data.frame`(deriv.points, , "eval.point"): undefined columns selected".
  Python `nns.dy_dx(x, y)` -> "ValueError: some columns are not in the data.table: [eval.point]" (a data.table message in a
  NumPy library) for every input (work/X/fuzz.py).
- Proposed fix: either default eval.point to "overall" (the documented mean-derivative mode) or stop() early with
  "eval.point must be provided (numeric or 'overall')". Python: same, with a clear ValueError.

### X-07: `nns_causation` on a constant series reports "x must contain only finite values"
- Type: api
- Severity: low
- Confidence: verified
- Evidence: NNS.norm of a constant column produces NaN internally; Python surfaces the inner lpm_ratio error about finiteness, R
  surfaces "x has missing values" from NNS.dep. Neither tells the user the input is constant.
- Proposed fix: validate up front: raise "x and y must each have at least 3 observations and non-zero variance".

### X-08: `nns_arma` on a length-2 or length-3 series raises IndexError (R: "[y] must contain at least two observations")
- Type: python-bug
- Severity: low
- Confidence: verified
- Proposed fix: validate series length up front with a clear ValueError.

### X-09: Subnormal x values give NaN point estimates in nns_reg (R returns classes)
- Type: python-bug (edge)
- Severity: low
- Confidence: verified
- Evidence: x = [2.2e-309, 0, 0, 0, 0], y = [1,2,3,1,2], type="class", point_est = [min, mean, max]: R 2 1 1; Python 2 NaN 1.
  Same data scaled by 1e300 gives 2 1 1 in Python. This made tests/property/test_regression.py::
  test_nns_reg_classification_bounds_hold flaky (hypothesis occasionally draws subnormals).
- Fix applied: property strategy excludes subnormals (allow_subnormal=False). Code fix (guard slope = dy/dx when dx is
  subnormal) left as a low-priority item.

### X-10: tests/_r.py nns_stack_numeric seeds R with set.seed() but NNS.stack re-seeds with its own default seed = 123
- Type: test-bug
- Severity: low-medium
- Confidence: verified (reported by reg-smooth agent; R/Stack.R:93 `seed = 123L`, :275 `set.seed(seed)`)
- Evidence: every R reference with seed=7 or 42 actually ran with seed 123; the seeded stack parity tests only assert
  structure, so the mismatch is invisible.
- Proposed fix: pass `seed = as.integer(args$seed)` to NNS.stack in the helper and regenerate the affected cache shard
  entries (cache keys do not include the script text, so old entries must be deleted).

### X-11: Co-moment parity tests key the R cache on random draws from numpy multivariate_normal
- Type: test-fragility
- Severity: low
- Confidence: verified (dep-pm agent: 120 cache misses under the worktree venv's NumPy build)
- Evidence: tests/parity/test_co_moments.py generates inputs with multivariate_normal, whose output can differ in the last
  bits across NumPy/BLAS versions, so the cache key (hash of the inputs) changes and CI (NNS_R_CACHE_ONLY=1) would fail on
  a different NumPy.
- Proposed fix: generate inputs from standard_normal and an explicit Cholesky factor, or round inputs (e.g. to 12 digits)
  before hashing / passing to R.

### X-12: `dy_d_best` is a Python-only public function that returns exactly what `dy_d` returns; both are wrong (REG-04)
- Type: api / dead-code
- Severity: medium (public, documented as "reconciled" estimator)
- Confidence: verified (work: inline script, uniform X 300x2, eval points (0,0),(0.5,-0.5))
- Evidence: y = 2*x1: dy_d and dy_d_best both give d/dx1 = [1.279, 0.939] (truth 2), d/dx2 = [0.825, 0.969] (truth 0).
  y = x1^2 + 3*x2: d/dx1 = [1.285, 1.388] (truth [0, 1]), d/dx2 = [1.23, 1.482] (truth 3). Identical outputs from both
  functions. dy_d also prints "Currently generating NNS.reg finite difference estimates...Regressor k" to stdout by default.
- Proposed fix: deprecate dy_d_best (alias with DeprecationWarning) once dy_d follows R's fixed dy.d_ (R fix cebf750:
  multivariate NNS.reg fit); default messages=False (Python library convention: no stdout noise).

### X-13: The pandas parity tests could never run in CI
- Type: ci
- Severity: medium
- Confidence: verified
- Evidence: tests/parity/test_labels.py imports pandas at module level, while pandas was in neither the dev dependency group
  nor the CI install lines (other pandas tests use importorskip and were silently skipped in CI). `uv sync` removed an ad-hoc
  pandas and collection failed.
- Fix applied: pandas in the dev group and in the three workflows that run the suite (commit "ci: install pandas ...").
