# Independent review of the NNS-python branch (3d8b4e1..8150ed0)

Status after review: finding 1 (_fast_lm fallback) and finding 2 (ncores < 1) were fixed in commits 'fast_lm: fallback uses R's means...' and 'ncores: treat values below 1...'. Findings 3-5 are open.

The diff introduces no serious correctness regressions on any function I tested. I ran a broad differential sweep comparing the branch (native and `NNS_DISABLE_NATIVE=1`), the old Python at 3d8b4e1, and the patched R 13.3. Every main function matched R to about 1e-7, and native matched fallback, with the one exception in finding 1. That leaves five verified findings, all medium or low severity. Nothing in the repo was edited or committed. Scratch scripts and outputs are in `/home/user/audit/work/review/scripts/` and `/home/user/audit/work/review/`, and the base worktree has been removed.

## Findings (most severe first)

**1. Medium: native and fallback give different `nns_meboot` / `nns_arma` / `nns_mc` results for the same seed.**
- **Where:** `src/nns/_helpers.py:74-82` (`_fast_lm` fallback).
- **Repro:** `nns.nns_meboot(np.array([2.390137,2.847274,3.173205]), reps=4, rho=0.6, random_seed=1)`.
  - Native: `ordxx=[1,3,2]`, `xmin=-0.08747066666666714`. This matches R's `NNS.meboot`.
  - `NNS_DISABLE_NATIVE=1`: `ordxx=[3,1,2]`, `xmin=-0.0874706666666667`, and completely different replicates.
- **Knock-on effect:** `nns.nns_arma(ts, h=3, seasonal_factor=[5], pred_int=0.9, random_seed=1)` on `/home/user/audit/work/review/d_ts.csv` gives Lower 90% `[2.274689, …]` native and `[2.267086, …]` fallback. The estimates are identical.
- **Cause:** NNS-core's `fast_lm` uses R's long-double two-pass mean (`mean_vec`) and sequential sums. The fallback uses pairwise `np.mean`/`np.sum`. That makes the intercept differ in the last bit, so two residuals that should be equal stop being equal and sort in a different order.
- **Status:** the code itself predates this diff. Commit 0026881 adds `NNS_DISABLE_NATIVE` and claims the fallbacks now match NNS-core, and this one does not.
- **Fix:** in the fallback, compute `mean_x`/`mean_y` with `nns._rmean.r_mean` and accumulate `var_x`/`cov_xy` sequentially, as `seasonality._seq_mean` already does. Add a native-vs-fallback test using this tied input.

**2. Low–medium: `ncores` is now strictly validated, with no deprecation path.**
- **Where:** `src/nns/_native.py:62-80` (`native_threads`, applied by `@honours_ncores`), and the now-stale docstring at `src/nns/_public_reg.py:92-93` ("ncores … accepted for R compatibility and ignored").
- **Repro:** `nns.nns_reg(x, y, ncores=0)` (also `-1`, `1.5`) raises `ValueError: [ncores] must be NULL or a positive integer.`
  - At 3d8b4e1, `ncores` was simply deleted, so any value worked.
  - R's univariate `NNS.reg(x, y, ncores=0)` and `ncores=-1` run normally and return R2. Only the multivariate R path rejects them.
- **Affects:** `nns_reg`, `nns_m_reg`, `nns_stack`, `nns_boost`, `nns_var`, `nns_arma_optim`. Callers passing scikit-style `ncores=-1` now break.
- **Fix:** for paths where R ignores `ncores`, treat values below 1 as `None` with a `DeprecationWarning` for one release. Update the docstring.

**3. Low: thread-setting API doesn't behave as documented.**
- **Where:** `extern/NNS-core/include/nns/parallel.hpp:25-33` and `src/nns/_native.py:40-52`.
- **Environment variable is read once:**
  - Repro: `nns.get_num_threads()` → 4; then `os.environ['NNS_NUM_THREADS']='2'; nns.set_num_threads(None)` → still 4.
  - `set_num_threads(None)` claims to restore `NNS_NUM_THREADS`, but the value is cached in a function-static on first use.
  - An invalid value (`NNS_NUM_THREADS=abc`) is silently ignored.
- **Overflow:** `nns.set_num_threads(2**40)` raises a nanobind `TypeError: incompatible function arguments` instead of the documented `ValueError`. The same applies to a huge `ncores` going into `_set_call_threads`.
- **Fix:** document that the environment variable is read once at the first native call, or re-read it when `set_num_threads(None)` is called. Range-check `n` and `ncores` against `INT_MAX` in `_integer()`.

**4. Low: inconsistent handling of non-finite targets in `lpm`/`upm`.**
- **Where:** `src/nns/core.py:109-121` (`_moment`).
- **Repro:**
  - `nns.lpm(1, np.nan, x)` → `nan`, and `nns.lpm(1, np.inf, x)` → `inf`.
  - `nns.lpm(0, np.nan, x)` now returns `nan`; at 3d8b4e1 it returned `0.0`.
  - Meanwhile `lpm_ratio(1, nan, x)`, `co_lpm(..., nan, ...)` and `lpm(..., excess_ret=True)` all raise.
  - R (fixed) raises `` `target` must be finite. `` in all of these cases.
- **Cause:** commit 7c30910 made NaN propagate deliberately, which contradicts the R reference and the sibling functions.
- **Fix:** call `_require_finite(targets, "target")` in `_moment`, as `_ratio` does. If propagation is meant to stay, document the difference instead.

**5. Low: submodule names removed without a deprecation shim.**
- **Removed:**
  - The whole `nns.smoothing` module.
  - From `nns.regression`: `nns_part`, `PartResult`, `nns_dep`, `lpm_var`, `upm_var`, `nns_mode`, `encode_factor_codes`, `r_smooth_spline_fixed_spar`.
  - From `nns.dependence`: `co_lpm`, `co_upm`, `d_lpm`, `d_upm`.
  - From `nns.multivariate_regression`: `nns_reg`, `nns_distance`, `nns_mode`, `upm_var`.
- **Impact:** every top-level `nns.*` export and its signature is intact (`get_num_threads`/`set_num_threads` were added), so only code doing `from nns.regression import nns_part` and similar breaks.
- **Fix:** a module-level `__getattr__` that forwards with a `DeprecationWarning`, plus a changelog note.

**Intended behaviour changes worth a changelog line (not bugs):**
- `nns_m_reg(factor_2_dummy=...)` default changed from False to True. This matches R, and factor inputs used to error anyway.
- `nns_dep(X)` (matrix form) is new.
- `fsd` gained a `type=` argument.
- `causal_matrix` and `nns_norm` now return DataFrames for DataFrame input.
- Non-finite `x` now raises in `lpm`, `nns_moments`, `pm_matrix`, `nns_anova`, `nns_cdf`.

## Checked and fine
- **Against R, head matches on both native and fallback for:**
  - 300 random `fsd`/`ssd`/`tsd`/continuous-FSD pairs, each in its plain and `_uni` form, including ties, unequal lengths, 1e6 scale and near-equal samples.
  - 60 matrices × 4 settings for `sd_efficient_set` and × 3 for `nns_sd_cluster`.
  - 48 `nns_reg` cases: order, noise_reduction, dim_red, integer and tied x.
  - 14 `nns_stack` cases: CLASS, method, dim_red, pred_int.
  - 32 `nns_arma` configs (both scripts) and 6 `nns_arma_optim` configs.
  - `nns_seas`, `nns_var` (4 configs), and 54 `nns_cdf` cases (all types and degrees, 1-3D, targets, NaN dropping).
  - `dy_dx`, `dy_d` (mixed, median, scalar, vector wrt), `nns_diff`, `nns_part`, `nns_copula`, `nns_causation`, `nns_dep` (including the asymmetric matrix form), `nns_anova` (seeded, robust, tails, medians, 3 groups), `nns_boost`, `nns_m_reg`, `pm_matrix`, the nD co-moments, `lpm_var`/`upm_var` with vector inputs.
  - Categorical, object, string-class and pandas-Categorical inputs, unseen-level errors, and point_est column-name matching.
- **Edge cases:** n = 1–5, constant inputs, ties, NaN/Inf and empty arrays behave like R or raise. Native and fallback agree on all 312 edge cases.
- **Dtypes:** int64, int32, uint8 and int8 inputs give results identical to float64.
- **Pandas:** non-default, permuted and string indices are handled positionally with no index alignment, across 24 functions.
- **Inputs are never mutated** by 23 public functions (numpy and pandas).
- **Threading:** concurrent `nns_m_reg`/`lpm` calls from 8 threads with mixed `ncores` give bit-identical results. The per-call thread setting is thread-local and is restored after exceptions. The GIL is released only around pure C++ work.
- **Warnings** point at the caller's line (`external_stacklevel`).
- **Performance at n = 20k:** no new O(n²) behaviour. Head is faster than base nearly everywhere, for example the bivariate CDF goes from 7.0s to 0.18s and the efficient set from 0.9s to 0.05s. Two-sample `nns_anova` is roughly linear (2.3s at n = 20k), dominated by R's 1000×n bootstrap draws, as in R.
- **`nns_stack_fit` / `nns_stack_predict`** round-trip exactly.
