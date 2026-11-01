# PM (partial moments core) audit findings

Scope: `nns.core` (lpm/upm/ratios), `pm_matrix`, `co_moments`, `classical`, `central_tendencies`,
`cdf`, `var.lpm_var/upm_var`, `_helpers`, `_native`, matching parts of `_nnscore_bindings.cpp` and
vendored `extern/NNS-core/src/partial_moments.cpp`; R `Partial_Moments.R`, `LPM_UPM_VaR.R`,
`Central_tendencies.R`, `src/partial_moments*.{cpp,h}`, related man pages.
All scripts are in `/home/user/audit/work/PM/` (R bridge: `rrun.py`; R runs with
`RcppParallel::setThreadOptions(1)`). Python = `/home/user/NNS-python/.venv/bin/python`, R = NNS 13.3.

---

### PM-01: R `LPM.ratio`/`UPM.ratio`/`Co.LPM`/`Co.UPM`/`D.LPM`/`D.UPM`/`PM.matrix` silently replace an integer target with `mean(variable)`, so `NNS.CDF` of any integer vector is garbage
- Type: r-bug
- Severity: critical
- Confidence: verified
- Where: R `src/partial_moments_rcpp.cpp:161,202,231,237,266,272,319,325,372,378,413` (`if (is<NumericVector>(target) && !target.isNULL()) ... else target = mean(variable)`); `is<NumericVector>` is FALSE for INTSXP. Triggered by `R/Partial_Moments.R:269-270` (`x <- sort(variable); pval <- LPM.ratio(degree, x, variable)`), which keeps integer type.
- Evidence: `work/PM/t15_r_integer_target.R`
  - `LPM.ratio(0, 2L, c(1,2,3,10))` = 0.75 (that is LPM.ratio at mean 4); `LPM.ratio(0, 2, ...)` = 0.5. `LPM.ratio(0, 1:3, x)` returns ONE value 0.75 instead of `0.25 0.5 0.75`.
  - `Co.LPM(1, x, x, 2L, 2L)` = 3.5 vs 0.25; `D.LPM(1,1,x,rev(x),0L,0L)` = 4.5 vs 0; `PM.matrix(1,1,c(0L,0L),A,FALSE)$cupm[1,2]` = 0 vs 8 (integer target silently becomes colMeans).
  - `NNS.CDF(1:10)$Function$CDF` = `0.5 0.5 ... 0.5`; `NNS.CDF(rpois(200,3))` is the single value 0.645 everywhere; `NNS.CDF(airquality$Ozone)` (integer + NA) is all 0 for degree 0 and all NaN for degree 1.
  - Python is correct: `nns.nns_cdf(np.arange(1,11))` gives `0.1 ... 1.0`, `nns.lpm_ratio(0, np.array([1,2,3]), x)` gives `[0.25, 0.5, 0.75]`.
- Which side is right: Python. A target is a number; its storage type cannot change the answer, and `LPM()` itself coerces with `as.numeric` (so `LPM(0, 2L, x)` = 0.5), proving the intended semantics.
- Proposed fix (R): in every `*_RCPP` wrapper accept `is<NumericVector>(t) || is<IntegerVector>(t)` (or do `target <- as.numeric(target)` in R wrappers as `LPM` does); in `PM.matrix` the same. Changes R results (fixes them). No Python change; worth a Python regression test with integer inputs so a future "parity" fix does not copy the R behaviour.

### PM-02: Multivariate `nns_cdf` builds (n, n, d) tensors: ~25x slower than R and O(n^2 d) memory, while the identical native kernel is already bound
- Type: perf
- Severity: high
- Confidence: verified
- Where: `src/nns/cdf.py:320-352` (`_co_lpm_nd_rows`: `diff = values[None,:,:] - values[:,None,:]`, several n x n x d temporaries); also `_marginal_lpm_ratios` (`cdf.py:355`) builds n x n per column via `_r_lpm_ratio`.
- Evidence: `work/PM/t10_mvcdf_perf.py` (and `t03_cdf.py` tail)
  - n=2000,d=2,deg=1: Python 2.47 s, peak RSS 377 MB; R 0.09 s.
  - n=4000,d=2,deg=1: Python 7.6-7.8 s, peak RSS 1125 MB; R 0.30-0.42 s; `_nnscore.clpm_nd_batch` (already exported by the extension, unused by Python) 0.15 s with identical output (`np.allclose` True, column-major flatten).
  - n=4000 deg=0: Python 0.9 s / 376 MB vs native 0.06 s, R 0.10 s. n=10,000 would need ~5-8 GB.
- Which side is right: numbers agree with R (max diff 6e-16); this is purely a perf/memory problem.
- Proposed fix: `cdf = native_fn("clpm_nd_batch")(values.ravel(order="F"), n, d, values.ravel(order="F"), n, degree, True)` (exactly R's `Co.LPM_nD.batch`); marginals via `lpm_ratio(degree, col, col)` (native, prefix path). No result change.

### PM-03: Univariate `nns_cdf` falls back to a dense n x n matrix for fractional degree, degree >= 4, or any NaN (11-13 s / 2.5 GB at n=1e4; OOM beyond ~3e4)
- Type: perf
- Severity: high
- Confidence: verified
- Where: `src/nns/cdf.py:64-70` (NaN -> `_r_lpm_ratio(degree, x, values)`), `cdf.py:263-264` (degree not in {1,2,3} -> `_r_lpm_ratio(degree, sorted, sorted)`), `_r_partial_moments` (`cdf.py:298-317`) builds `targets[:,None] - values[None,:]`.
- Evidence: `work/PM/t11_cdf_quadratic.py`, n=10,000 t3 data:
  - degree 4: Python 11.4 s, peak RSS 2471 MB; R 0.009 s (prefix path).
  - degree 1.5: Python 13.0 s, 2471 MB; R 2.0 s.
  - degree 1 with one NaN: Python 7.9 s, 1708 MB; R 0.6 s. degree 0 with one NaN: 1.9 s / 1708 MB vs R 0.22 s.
  - Results agree with R (max diff <= 4e-15). At n=1e5 these paths need 80+ GB.
- Proposed fix: call the native `lpm_ratio_v` (`nns.lpm_ratio`) on the finite values for all degrees; for the NaN-in-denominator quirk (see PM-15), rescale by `n_finite/n` instead of materialising n x n. No result change.

### PM-04: `lpm`/`upm`/`lpm_ratio` with any NaN in `x` drop to an (m x n) dense NumPy fallback and silently miscount at degree 0
- Type: python-bug (+perf)
- Severity: medium
- Confidence: verified
- Where: `src/nns/core.py:53-66` (`_native_safe` rejects non-finite input -> broadcast fallback), degree-0 branch `core.py:58-62`.
- Evidence: `work/PM/t12_nan_fallback.py`, `t04_api.py`
  - 1000 targets x 1e5 obs: finite 16 ms; with a single NaN 4506 ms, peak RSS 2.3 GB, result all NaN (degree 1).
  - Degree 0 with a NaN: `lpm(0,0,x)+upm(0,0,x)` = 0.99999 (NaN counted in n but in neither tail); `x=[1,2,3,nan,5]`: `lpm(0,3,x)` = 0.6 (neither 3/4 nor NaN); `lpm(1,2,x)` = NaN. So NaN yields NaN at degree > 0 but a silently wrong number at degree 0.
  - R `LPM`/`UPM` raise "`variable` must be finite".
- Which side is right: R's error (or explicit NaN). A probability that does not sum to 1 is never right.
- Proposed fix: validate like R (`raise ValueError("x must be finite")`), or return NaN for every degree; never take the O(m*n) path just because of NaN. Changes degree-0 results on NaN input (currently wrong).

### PM-05: `nns_cdf` univariate degree 1-3 uses an unshifted prefix-power expansion: catastrophic cancellation on level-shifted data (CDF outside [0,1], non-monotone, inf); R is fine
- Type: parity-mismatch (Python bug)
- Severity: medium
- Confidence: verified
- Where: `src/nns/cdf.py:255-295` `_finite_sorted_grid_lpm_ratio` (raw `sorted_values**power` prefix sums, no centring). R (and the Python native kernel) centre at the median (`partial_moments.h` `shift = sorted[n/2]`).
- Evidence: `work/PM/t03_cdf.py`
  - x = 1e6 + N(0,1), n=2000: degree 2 max |grid - exact| = 1.05e-3, 4 values outside [0,1], 85 non-monotone steps; degree 3: max error = inf, 1254 values outside [0,1]/NaN.
  - vs R `NNS.CDF` (n=300, 1e6 + N(0,1)): degree 2 CDF max diff 6.4e-4, hazard 3.98, cumulative hazard 0.27; degree 1 hazard 3.9e-7.
  - Realistic finance levels are mostly fine (SPY/BTC prices max error <= 2e-14) because the sd/mean ratio is not extreme; failure needs mean/sd around 1e3 or more (e.g. sensor levels, Kelvin temperatures, epoch timestamps at degree 3).
- Which side is right: R (exact to 1e-15).
- Proposed fix: delete `_finite_sorted_grid_lpm_ratio` and call `nns.lpm_ratio(degree, x_sorted, values)` (native prefix backend with median shift; already O(n log n)). Changes results only where they are currently wrong.

### PM-06: `NNS.moments(population = FALSE)` skewness is inflated by (n/(n-1))^1.5 (uses population variance in the sample formula); Python copies it
- Type: r-bug (Python replicates: `src/nns/classical.py:62`)
- Severity: medium
- Confidence: verified
- Where: R `R/Partial_Moments.R:459-460` `(n/((n-1)(n-2))) * ((n*skew_base)/variance^(3/2))` with `variance = mean(z^2)`; the kurtosis branch correctly uses `variance*(n/(n-1))`.
- Evidence: `work/PM/t07_moments.py`: x=c(1,2,4,8,16): R/Python 1.852184; scipy `skew(bias=False)` (= Excel SKEW = adjusted Fisher-Pearson G1) 1.325315; ratio = 1.3975 = (5/4)^1.5. NVDA n=4231: 0.5594661 vs 0.5592678. Kurtosis (sample) matches scipy exactly, which shows the intended estimator.
- Which side is right: neither; the standard G1 = n/((n-1)(n-2)) * sum(z^3)/s^3 with s the sample sd.
- Proposed fix (both): `skewness <- (n/((n-1)*(n-2))) * (n*skew_base) / (variance*n/(n-1))^(3/2)`. Changes sample skewness (by up to 40% at n=5, 0.035% at n=4000).

### PM-07: Multivariate `NNS.CDF` survival/hazard/cumulative hazard use the 2-D inclusion-exclusion formula for any d; wrong for d >= 3. Also only 2 coordinates are bounds-checked
- Type: r-bug (Python replicates, `src/nns/cdf.py:150-175, 194-215`)
- Severity: medium
- Confidence: verified
- Where: R `R/Partial_Moments.R:326-338,347` (`1 - rowSums(marginals) + CDF`), bounds check `R/Partial_Moments.R:248` (only `target[1]`, `target[2]`). The man page says "Multivariate analysis is restricted to CDF", but the code returns the other types anyway.
- Evidence: `work/PM/t13_mv_survival3.py`: 3 correlated normals, n=400, degree 0: max |survival - P(X>x,Y>y,Z>z)| = 0.3675 in R and Python (one row: true 0.3675, returned 0.0). For d=2 the formula is exact (max err 2e-16, `t03_cdf.py`). `target=c(0,0,1e6)` (third coordinate far outside the support) is accepted, value 0.335.
- Which side is right: neither for d >= 3. P(all > t) needs the full inclusion-exclusion over all subsets, or a direct count (degree 0: `mean(all(X > t))`).
- Proposed fix: for degree 0 compute the joint survival directly (a `Co.UPM_nD`-style strict count); for d >= 3 either implement the full alternating sum or raise. Bounds-check every coordinate. Changes d >= 3 results.

### PM-08: `NNS.mode(multi = TRUE)` (R's default) finds spurious modes; their number grows with n on unimodal data
- Type: r-bug (Python replicates exactly in native and fallback)
- Severity: medium
- Confidence: verified
- Where: R `src/central_tendencies.cpp` peak finder (tri-7 smoothing + local max in +/-3 bins, no prominence or count threshold); Python `central_tendencies.py:_peak_indices/_non_maximum_suppress` mirrors it.
- Evidence: `work/PM/t18_mode_spurious.py`: N(0,1), 20 draws each: n=200 -> 1-4 modes; n=1000 -> 5-12 modes; n=5000 -> 19-32 modes; t3 "returns" n=4000 -> 50 modes; NVDA daily returns -> 42 modes (`t08_central.py`). Every isolated tail bin becomes a "mode".
- Which side is right: neither; a normal sample has one mode.
- Proposed fix: require a minimum prominence (e.g. smoothed peak >= some fraction of the global maximum, or a count significance test) before reporting a secondary mode. Changes multi-mode output.

### PM-09: `nns_mode` default `multi=False` differs from R `NNS.mode(multi = TRUE)`; undocumented
- Type: parity-mismatch / api
- Severity: medium
- Confidence: verified
- Where: `src/nns/central_tendencies.py:51-55`; R `R/Central_tendencies.R:19`.
- Evidence: `work/PM/t08_central.py`: `NNS.mode(bimodal)` returns `-2.98 2.77` in R; `nns.nns_mode(bimodal)` returns `-2.98`. The same default mismatch shows on ints, Ozone, exponential and NVDA. All explicit `multi=` calls match R exactly.
- Which side is right: arguably Python's default is the more sensible one (see PM-08), but it is a silent divergence for ported code and not listed in docs/conventions.md.
- Proposed fix: either change the default to `True` for parity, or document the divergence in conventions.md and the docstring. Internal callers already pass `multi=False` explicitly.

### PM-10: `target=None` silently returns NaN in `lpm`/`upm`/`lpm_ratio`/`co_lpm`/`d_lpm`...; R documents and implements a default of `mean(variable)`
- Type: python-bug / api
- Severity: medium
- Confidence: verified
- Where: `src/nns/core.py:116-122` `_as_targets` (`np.asarray(None, dtype=float64)` gives nan); `co_moments.py:69-70`.
- Evidence: `work/PM/t04_api.py`: `nns.lpm(1, None, s)` gives nan; `nns.co_lpm(1, s, s, None, None)` gives nan. R: `LPM.ratio(1, NULL, 1:5)` = 0.5, `Co.LPM(1, 1:5, 5:1, NULL, NULL)` = 0 (mean targets, `partial_moments_rcpp.cpp:161-167,231-242`). Note that R `LPM(1, NULL, x)` returns `numeric(0)`, which is itself inconsistent.
- Proposed fix: treat `None` as `mean(x)` (as R's ratio/co-moment functions and Python's own `pm_matrix` do), or raise `TypeError`. Never return NaN.

### PM-11: `lpm_var`/`upm_var` accept only a scalar percentile (R is vectorised); looping is 21x slower
- Type: api / perf
- Severity: medium
- Confidence: verified
- Where: `src/nns/var.py:781-830` (`float(percentile)`); R `R/LPM_UPM_VaR.R:21-37` vectorises (prefix backend built once, one interval search per p).
- Evidence: `t04_api.py`: `nns.lpm_var(np.array([0.05,0.5]),1,x)` raises `TypeError: only 0-dimensional arrays can be converted`; R `LPM.VaR(c(0.05,0.5),1,0:9)` gives `1.3696 4.5`. `t16_var_vector_perf.py`: 100 percentiles, n=1e5, degree 1: Python loop 1141 ms vs R 53 ms (same values, max diff 6e-16).
- Proposed fix: accept array-like `percentile`, build sort/prefix once in `_lpm_var_integer`, `np.searchsorted` all p, solve each interval; return an ndarray for array input and a float for a scalar. Additive API change.

### PM-12: `skew_pm`, `kurt_pm`, `nns_moments` raise `ZeroDivisionError` on constant input (and `nns_moments(population=False)` for n <= 3); R returns NaN
- Type: python-bug
- Severity: low-medium
- Confidence: verified
- Where: `src/nns/classical.py:34,42,59-66` (Python float division by `variance**1.5`, where variance is 0.0, or by `(n-3)`).
- Evidence: `t04_api.py`, `t07_moments.py`: `nns.nns_moments(np.ones(5))` raises `ZeroDivisionError`; R `NNS.moments(rep(1,5))` gives mean 1, variance 0, skewness NaN, kurtosis NaN. `nns.nns_moments([1,2,4], population=False)` raises; R gives kurtosis NaN and finite skewness. Constant series do occur in practice (halted or stale prices, zero-volume windows).
- Proposed fix: use NumPy float64 arithmetic (`np.float64(skew_base)/variance**1.5` under `np.errstate`), return NaN like R.

### PM-13: Vectorised (>32 targets) LPM/UPM use a binomial prefix-power expansion: large relative errors and negative moments in the tails at degree >= 3; results depend on how many targets are passed
- Type: r-bug (shared by NNS-core and Python)
- Severity: low-medium
- Confidence: verified
- Where: R `src/partial_moments.h` `PrefixPartialMomentBackend::lpm/upm` (degree <= 32, used when `length(target) > 32`); NNS-core `extern/NNS-core/src/partial_moments.cpp:73-188`.
- Evidence: `work/PM/t02_prefix_cancellation.py`, `t02b_high_degree.py` (target = sort(x), the `LPM.ratio(d, sort(x), x)` use case):
  - SPY price levels, degree 4: UPM max rel err 0.273 (31 targets > 1e-6), one negative LPM, one `LPM.ratio` outside [0,1]. R identical (`maxrel 0.2728, ratio_out 1`).
  - NVDA returns: degree 8 max rel err 4.27 and 2 negative LPMs (R also: `neg 2, min -5.6e-24`); degree 12 rel err 9e8.
  - The same target gives different answers depending on call size: R `UPM(4, t[2:33], x)` (32 targets, direct scan) = 0.0039076 vs the same target inside a 33-target call = 0.0039212.
  - Degrees 0-2 are accurate (<= 5e-8 rel), so the common paths are fine.
- Proposed fix: restrict the prefix backend to degree <= 2 (or compute tail targets k < degree+few by direct scan), clamp to >= 0. Changes only tail values that are currently wrong.

### PM-14: `NNS.CDF(type = "hazard")` crashes on data with runs of ties (finite-difference denominator 0); Python crashes too with an opaque message
- Type: r-bug (Python replicates)
- Severity: low-medium
- Confidence: verified
- Where: R `R/Partial_Moments.R:282-285` (`(F[hi]-F[lo])/(x[hi]-x[lo])` is 0/0 when the window lies inside a tie run); Python `cdf.py:_hazard_proxy` (`cdf.py:340-352`).
- Evidence: `work/PM/t03b_hazard_ties.py`: NVDA first 1000 daily returns (contains a run of zero returns): R `Error: [y] contains missing values.`; Python `ValueError: [y] must contain finite values.` (from `nns_reg`, plus RuntimeWarnings). Discrete or rounded data hits this easily.
- Proposed fix: compute the proxy on unique x (collapse ties, weight by multiplicity) or skip zero-width windows. Changes nothing where it currently works.

### PM-15: R ratio/co-moment functions skip the finite check that `LPM`/`UPM` have, so NA is counted in n; `NNS.CDF` of double data with NA tops out below 1 (Python copies this on purpose)
- Type: r-bug (Python replicates in `nns_cdf`)
- Severity: low-medium
- Confidence: verified
- Where: R `src/partial_moments_rcpp.cpp:145-210` (no NA validation; `LPM_C` divides by full n), `R/Partial_Moments.R:269-270`; Python `cdf.py:64-70` emulates it (docs/conventions.md: "NA/Inf comparison quirks are handled").
- Evidence: `work/PM/t19_r_na_ratio.R`, `t04_api.py`: x=c(1,2,3,NA,5): `LPM(0,2,x)` errors but `LPM.ratio(0,2,x)+UPM.ratio(0,2,x)` = 0.8 and `LPM.ratio(0,6,x)` = 0.8 (a CDF beyond the max should be 1). `t14_cdf_na.py`: Python `nns_cdf(airquality Ozone as float, 0)` max CDF = 0.758 = 116/153 (R gives the same for the double version, and garbage for the integer version, see PM-01). Python `nns_cdf(..., target=30)` with NaN raises R's internal message "missing value where TRUE/FALSE needed".
- Which side is right: neither. Drop NA (as `sort()` already does for the grid) or error.
- Proposed fix: R: validate or `na.omit` in `LPM.ratio`/`UPM.ratio`/`Co.*`/`D.*`/`NNS.CDF`. Python: drop NaN in `nns_cdf` and give a clear error message. Changes NA-input results.

### PM-16: Python has no `excess_ret` option; R's `excess_ret` degree-0 LPM uses strict `<`, which contradicts LPM's own `<=` convention
- Type: api (Python) + r-bug (R)
- Severity: low
- Confidence: verified
- Where: R `src/partial_moments_rcpp.cpp:79` (`if (diff > 0)` in the LPM excess branch); Python `core.py:9-14` has no parameter.
- Evidence: `work/PM/t19_r_na_ratio.R`: R `LPM(0, 2, c(1,2,3), excess_ret=TRUE)` = 0.333 vs `LPM(0, 2, c(1,2,3))` = 0.667 and `LPM(0, c(2,2,2), x)` = 0.667. So in excess mode LPM0 + UPM0 = 2/3 with a tie. Python: `nns.lpm(..., excess_ret=True)` raises `TypeError` (`t04_api.py`).
- Proposed fix: R: use `diff >= 0` for LPM degree 0 in the excess branch. Python: add `excess_ret: bool = False` (element-wise `mean(max(t_i - x_i, 0)^d)`, target length 1 or n).

### PM-17: Fractional-degree `LPM.VaR`/`UPM.VaR` invert the ratio with `optimize()` at the default absolute tolerance (~1.2e-4 in x units); inverse is off by up to 0.8% relative at 5%
- Type: r-bug (Python mirrors it via `xatol=_R_OPTIMIZE_TOL`, `var.py:21,795-801`)
- Severity: low
- Confidence: verified
- Where: R `R/LPM_UPM_VaR.R:390-433`.
- Evidence: `work/PM/t06_var.py`: NVDA daily returns, degree 1.5, p=0.05: VaR -0.0283979, `LPM.ratio(1.5, VaR)` = 0.0500779; max |ratio(VaR(p)) - p| = 4.2e-4 (NVDA), 2.4e-4 (BTC). Python equals R bit for bit. The integer-degree paths are exact (no failures for degrees 1-3).
- Proposed fix: root-find `LPM.ratio(d,t)-p` with `uniroot`/`brentq` (it is monotone) at `tol = eps^0.5 * range(x)`. Small result change.

### PM-18: R accepts negative degrees silently (negative integers act as degree 0); Python rejects them
- Type: r-bug
- Severity: low
- Confidence: verified
- Where: R `src/partial_moments.cpp` `LPM_C`: `repeatMultiplication(value, -1)` loops zero times and returns 1.
- Evidence: `work/PM/t19_r_na_ratio.R`: R `LPM(-1, 3, 1:5)` = 0.6 (= `LPM(0, 3, 1:5)`); `LPM.VaR(0.05, -1, 0:9)` = 0.5016. Python raises `ValueError: degree must be non-negative` (`t04_api.py`).
- Which side is right: Python. Proposed fix (R): `stopifnot(degree >= 0)`.

### PM-19: Native kernels always spawn `hardware_concurrency()` threads with no way to limit them
- Type: perf / api
- Severity: low
- Confidence: verified
- Where: `extern/NNS-core/include/nns/parallel.hpp` (`n_threads = -1` gives all cores); `_nnscore_bindings.cpp:1101-1170` never passes `n_threads`.
- Evidence: `work/PM/t09b.py`: `lpm(2, 1000 targets, n=1e4)` 2.26 ms with threads on a shared 4-core box vs 1.07 ms pinned to one core. Inside multiprocessing or joblib pools this oversubscribes (cores x cores threads). R honours `RcppParallel::setThreadOptions`. Timing tables: `work/PM/timing_default.txt`, `timing_py_1core.txt`.
- Proposed fix: thread count from an env var or `nns.set_num_threads()`, passed through the bindings; serial below a work threshold.

### PM-20: `pm_matrix` native fused loop is 2-3x slower than the module's own BLAS fallback for wide inputs
- Type: perf
- Severity: low
- Confidence: verified
- Where: `extern/NNS-core/src/partial_moments.cpp` (fused per-pair row loop); fallback `src/nns/pm_matrix.py:66-71` (`D.T @ D`).
- Evidence: `work/PM/t17_pm_matrix_blas.py` (single-threaded BLAS, one core): 1e4 x 100: native 86 ms vs BLAS 31 ms; 1e5 x 50: 366 vs 161 ms; 1e6 x 11: 458 vs 358 ms; max rel diff <= 4e-14. R is about the same as native (88 ms at 1e4 x 100).
- Proposed fix: compute the deviation matrices natively (or in NumPy) and use `D_l.T @ D_l` and the other three products (or `crossprod` in R). No result change beyond rounding.

### PM-21: `nns_cdf` loses DataFrame/Series column names and emits spurious RuntimeWarnings for fractional degrees
- Type: api
- Severity: low
- Confidence: verified
- Where: `src/nns/cdf.py:229-237` (`_matrix_names` ignores DataFrame columns), `cdf.py:326,330` (`(-diff)**degree` evaluated before masking).
- Evidence: `nns.nns_cdf(pd.DataFrame(A, columns=['spy','tlt']), 0)["Function"]` keys are `['1','2','CDF']`; R gives `spy tlt CDF`. `nns.nns_cdf(A, 1.5)` emits 2 "invalid value encountered in power" warnings (values still match R to 4e-16).
- Proposed fix: take `names` from `variable.columns` when present; use `np.where(mask, np.abs(diff), 0)**degree`.

### PM-22: Edge-case NaN in degree >= 1 `LPM.ratio` when all mass sits at the target (e.g. `NNS.CDF(rep(2,4), 1)` is all NaN)
- Type: r-bug (Python identical)
- Severity: low
- Confidence: verified
- Evidence: R `NNS.CDF(2, 1)` and `NNS.CDF(rep(2,4),1)` give CDF NaN; Python the same; `lpm_ratio(1, 3, [3,3,3,3])` is NaN. `LPM.VaR` already maps this 0/0 to 0 internally (`.NNS_LPM_ratio_at_breaks`), so the package is internally inconsistent.
- Proposed fix: define the 0/0 case as the degree-0 ratio (1 at t >= c) or document it.

### PM-23: Docs and docstrings
- Type: docs
- Severity: low
- Confidence: verified
- Python: `lpm`, `upm`, `lpm_ratio`, `upm_ratio`, `co_lpm`, `co_upm`, `d_lpm`, `d_upm` have no docstrings (`__doc__ == ''`). `docs/api_reference.md:41-45,154,160` calls D.LPM/D.UPM/DPM "distance"/"directional" partial moments (they are divergent). It names R's `NNS.cdf` (actual: `NNS.CDF`). It says `ecdf_pm` is "used by moment routines" (nothing uses it). The `nns_mode` default divergence (PM-09) and `lpm_var` scalar-only (PM-11) are undocumented. `nns.pm_matrix` (the function) shadows the submodule `nns.pm_matrix` (`import nns.pm_matrix as m` yields the function).
- R man pages: `NNS.mode.Rd`/`NNS.gravity.Rd`: "discrete: FALSE (default) for discrete distributions" (inverted); `NNS.mode` "@return a numeric value" but the default returns a vector. `LPM.ratio`, `UPM.ratio`, `Co.LPM`, `Co.UPM`, `D.LPM`, `D.UPM`, `PM.matrix` say the target "defaults to mean", but the signatures have no default (you must pass `NULL`), and `LPM(…, NULL)` returns `numeric(0)`. `PM.matrix`: `pop_adj` is silently ignored when either degree is 0 (undocumented). `NNS.moments`: "population = TRUE performs the population adjustment" (it is the unadjusted statistic). `NNS.CDF`: "Multivariate analysis is restricted to CDF" (it is not, see PM-07); "Function: a data.table" (returns NNS.data.frame). `UPM.ratio` `\dontrun` example (`plot3d(x, y, Co.UPM(0, sort(x), sort(y), x, y), …)`) is not valid R and has wrong arguments. `Co.LPM_nD` "norm: normalize to the maximum observed value" (it divides by CLPM+CUPM+DPM and ignores `norm` at degree 0). `LPM.VaR` degree 0 returns a type-7 interpolated quantile (e.g. 1.6 for p=0.2 on ties data), which is not the inverse of `LPM.ratio(0, ., x)` (gives 0.222), despite "degree 0 for discrete distributions".

---

## Coverage (examined and found fine)

- `lpm`/`upm` scalar and vector targets vs naive `mean(max(t-x,0)^d)` for degrees 0, 0.5, 1, 1.7, 2, 3, 4 on normal, ties, NVDA, BTC, SPY prices, length 1/2, constant: exact for the scalar and <=32-target paths (`t01_core.py`). A/B vs R for degrees 0, 1, 2, 3, 0.5, 1.5 (scalar, 7 targets, >32 targets, ratios): all within 1e-9 (no mismatches).
- Degree-0 tie convention (`<=` for LPM, `>` for UPM; LPM0+UPM0 = 1) in univariate, co-moments (4 quadrants sum to 1 with ties at the target) and PM.matrix.
- Identities: LPM1-UPM1 at the mean = 0, UPM1(0)-LPM1(0) = mean, LPM2+UPM2 about the mean = population variance (all to about 1e-16 relative); `LPM.ratio` degree 0 = ECDF (right-continuous); degree 1 ratio continuous across data points.
- CLPM+CUPM-DLPM-DUPM = population covariance (diff 3e-21 on SPY/TLT); `D.LPM`/`D.UPM` quadrant semantics and mixed/fractional degrees vs naive; `degree_y`; target recycling (50 vs 25 targets) all match R exactly (`t05_comoments.py`).
- `pm_matrix`: `cov.matrix` = `np.cov(ddof=1)` with `pop_adj=True` and `ddof=0` without (diff 9e-19); DUPM = DLPM^T; 12 degree/norm/target/pop_adj combinations match R to 5e-16; pop_adj gating at degree 0 matches R; the pure-Python fallback equals native. n=1 row matches R (zeros).
- `lpm_var`/`upm_var`: exact inverse of `lpm_ratio`/`upm_ratio` for degrees 1, 2, 3 at p in {0.001,...,0.99}; A/B vs R for degrees 0, 1, 2, 3, 4, 0.5, 1.5 on NVDA, BTC, normal, ties, SPY prices (max diff <= 4e-9, uniroot vs brentq tolerance); NaN dropping and percentile clamping match R (`t06_var.py`).
- `nns_moments` population and sample (except PM-06), `var_pm`, `skew_pm`, `kurt_pm`, `mean_pm` vs scipy/R, including level-shifted 1e8 data (`t07_moments.py`).
- `nns_cdf` univariate (CDF, survival, hazard, cumulative hazard; degrees 0, 1, 2; with target) matches R on normal and NVDA-unique data; multivariate d=2 all types and degrees 0, 0.5, 1, 1.5, 2 match R; degree-0 joint CDF/survival = direct counts (`t03_cdf.py`).
- `nns_mode` (all discrete/multi combinations), `nns_gravity` (both), `nns_rescale` (minmax, riskneutral Terminal/Discounted, constant, NA, invalid type, case) match R on 13 data sets, including NA, length 2-4 and constant; the Python fallbacks equal native (`t08_central.py`, `t08b_fallback.py`).
- Timings (`timing_default.txt`, `timing_py_1core.txt`; R single-threaded): Python is on par with or faster than R for lpm/upm/ratio (scalar and 1000 targets at n = 1e4, 1e5, 1e6), co_lpm, pm_matrix (n x 11 and 1e4 x 100), lpm_var degrees 0/1/2/1.5, and univariate nns_cdf degrees 0/1 (6-7x faster). Example n=1e6: lpm(1, scalar) 7 ms vs 11 ms; 1e3 targets x 1e5 obs: lpm deg 2 12 ms vs 9 ms, lpm_ratio deg 1 9-12 ms vs 17 ms; pm_matrix 1e6 x 11 288-437 ms vs 285 ms; lpm_var(0.05, 1) 133-166 ms vs 282 ms. Fractional degree with 1000 targets is O(n*m) in both (1e6: 3.9-10.5 s vs 11.5 s).
- Input handling: pandas Series/list/int/bool inputs, list/Series targets, empty target (returns an empty array like R), 2-D target/x rejected, co-moment length mismatch rejected (documented divergence), `pm_matrix` scalar target broadcast (documented divergence; R errors).
