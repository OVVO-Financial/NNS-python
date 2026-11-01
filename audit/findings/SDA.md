# SDA findings: stochastic dominance, stochastic superiority, ANOVA, normalization

Scripts and outputs are in `/home/user/audit/work/SDA/` (abbreviated `W/` below). Python is
`/home/user/NNS-python/.venv/bin/python`; R is NNS 13.3 run with `RCPP_PARALLEL_NUM_THREADS=1`.
Python paths are relative to `/home/user/NNS-python/src/nns/`, and R paths to
`/home/user/ovvo-financial/nns/`.

---

### SDA-01: `fsd_uni(type="continuous")` builds a dense (n_x+n_y) x n matrix: 2.7 s at n=1e4, out of memory at n>=3e4
- Type: perf (python-bug)
- Severity: high
- Confidence: verified
- Where: `stochastic_dominance.py:789` (`_dominance_curve`): `upper = np.mean(np.maximum(0.0, values - grid[:, np.newaxis]), axis=1)`
- Evidence: `W/time_py.py` and `W/time_r.R`, with x = y + 0.1 so that every gate passes.
  - n=1e3: Python 30 ms vs R `NNS.FSD.uni(...,"continuous")` about 0.1 ms.
  - n=1e4: Python 2658-3384 ms vs R 1.5 ms (R `NNS.FSD(type="continuous")` 10 ms).
  - n=3e4: the process was OOM-killed.
  - n=1e5: `_ArrayMemoryError: Unable to allocate 149. GiB`.
  - `prof_py.py`: 82% of the time is spent on this one line.
- Which side is right: R. Its C++ evaluates the same quantity in O(n) from prefix sums.
- Proposed fix: use `upm(1, grid, values)` from `nns.core` (sort-based native code). Alternatively reuse
  `_pair_curve_values_at_thresholds(..., degree=1, discrete=False)`, which already computes this
  from prefix sums. Results do not change.

### SDA-02: Robust ANOVA resamples only the first `min(n1, n2)` observations of the longer group (R and Python)
- Type: r-bug (Python copies it)
- Severity: high
- Confidence: verified
- Where: R `R/ANOVA.R:111-115` (`l <- min(length(treatment), length(control))`, then
  `control[treatment_p]`); Python `anova.py:239-251` (`sample_size = min(...)`, `control[indices]`).
- Evidence: `W/anova_robust_trunc.R`. The control group is 50 draws of N(0,1) followed by 950 draws of N(5,1);
  the treatment group is 50 draws of N(0,1).

  | | Base Certainty | Robust Certainty Estimate [CI] |
  |---|---|---|
  | R | 0.0000 | 0.6249 [0.2376, 0.8693] |
  | Python | 0.0000 | 0.5929 [0.2729, 0.8721] |

  The robust output says "probably the same" because 95% of the control data is never looked at.
  The result also depends on the row order of the data.
- Which side is right: neither. The book (ch. 15, "Robust Estimation via Bootstrap Resampling")
  says the control and treatment samples are resampled independently with replacement. Each group
  should be resampled from its own full sample.
- Proposed fix: `ci = rng.integers(0, n_c, (n_c, 100))`, `ti = rng.integers(0, n_t, (n_t, 100))`,
  then pair column g with column g. This changes robust results.

### SDA-03: Python robust ANOVA uses paired (identical) resampling indices, unlike R. Robust estimates are biased upward and the output keys differ
- Type: parity-mismatch
- Severity: high
- Confidence: verified
- Where: `anova.py:240-259`
- Evidence:
  - What R does (`R/ANOVA.R:112-114`): it builds `treatment_matrix` with reversed columns, so control
    resample g is paired with treatment resample 102-g. 98 of the 100 replicates are independent.
    Python uses the same index vector for both groups. For correlated series (SPY/QQQ, correlation
    0.94) this keeps the day-by-day co-movement, so the resamples look more similar.
  - SPY vs QQQ over the first 300 days, mean of 30 seeds (`W/anova_py.py` vs `W/anova_r.R`):

    | | Robust estimate | Lower bound | Upper bound |
    |---|---|---|---|
    | R | 0.862 | 0.674 | 0.962 |
    | Python | 0.927 | 0.890 | 0.956 |

  - `W/robust_scheme.py` reproduces both numbers inside Python by switching only the pairing scheme
    (paired: 0.926 / 0.889; R scheme: 0.859 / 0.670).
  - The keys also differ. Python returns `"Lower Bound Robust Certainty"` and
    `"Upper Bound Robust Certainty"`; R returns `"Lower 95% CI"` and `"Upper 95% CI"` (the label
    follows the CI level, and "bound" is used for one tail). R's effect-size keys come out as
    `"Effect_Size_LB.2.5%"` and `"Effect_Size_UB.97.5%"`, because R keeps the quantile names.
  - Python ignores `tails` for the robust CI and always returns both bounds. With
    `confidence_interval=None`, Python still returns 95% bounds, while R returns only
    `Robust Certainty Estimate`.
- Which side is right: R's independent resampling, which is also what the book describes. Paired
  resampling is only valid for paired designs.
- Proposed fix: draw independent index matrices for the two groups (combine with the SDA-02 fix);
  use R's key names; honour `tails`. This changes public robust results.

### SDA-04: ANOVA `tails="left"` or `"right"`: R crashes; Python returns Effect_Size_LB=-inf and Effect_Size_UB=+inf
- Type: r-bug + python-bug
- Severity: high
- Confidence: verified
- Where:
  - R `R/Binary_ANOVA.R:132-152`: for `left`, `min_effect <- treatment_lower - control_upper`, but
    `control_upper` is only computed for `both`/`right`. For `right`, `control_lower` is used but
    only computed for `both`/`left`.
  - Python `anova.py:286-303`: it initialises `control_upper=inf` and `control_lower=-inf`, so
    `left` gives `treatment_lower - inf = -inf`.
- Evidence: x ~ N(0,1), y ~ N(1,1), n=50.
  - R: `tails="left"` gives "object 'control_upper' not found"; `"right"` gives "object
    'control_lower' not found". This applies to every `NNS.ANOVA(control, treatment, tails=...)`,
    including `robust=TRUE` (`W/anova_r.R`).
  - Python: left gives (-inf, inf), right gives (-inf, inf), both gives (0.44, 1.41).
- Which side is right: neither. A one-sided lower bound needs the one-sided control upper
  quantile, so compute both quantiles for every tail choice.
- Proposed fix: always compute `control_upper`, `control_lower`, `treatment_upper` and
  `treatment_lower` with the one-sided alpha. Then set LB = t_lower - c_upper for left (UB=inf),
  and UB = t_upper - c_lower for right (LB=-inf). This changes Python results (finite bounds) and
  fixes the R crash.

### SDA-05: TSD and continuous FSD are only checked at the sample points, so they report false dominance (R and Python)
- Type: r-bug (shared by Python)
- Severity: medium
- Confidence: verified
- Where:
  - R `src/SD.cpp:85-97` and `:145-152` (`for_each_threshold` visits only merged knots), and
    `R/TSD.R:37-43`.
  - Python `_dominates_uni` (`stochastic_dominance.py:770-775`) and all efficient-set paths.
- Evidence:
  - TSD: x = (-0.85, -1.52, 0.68), y = (-0.21, -0.33, -1.67).
    - R `NNS.TSD` says "X TSD Y", `NNS.TSD.uni` gives 1, and Python `tsd`/`tsd_uni` give 1.
    - The largest LPM2_x - LPM2_y over the merged knots is 0, but on a dense grid it is
      **+0.0389 at t=0.16** (`W/tsd_knots_verify.R`). So X does not TSD Y.
    - LPM2 is piecewise quadratic, so its difference can peak between knots.
    - Rate: 8 false positives among 33,154 TSD positives on random small normal pairs
      (`W/tsd_exact.py`). There are none among the 15 finance pairs.
  - Continuous FSD: x = (0.15, 1.55, -0.86, -0.8, 0.25, 0.33),
    y = (-0.15, 1.17, 0.26, -0.04, -2.44, 1.19).
    - R gives "X FSD Y", R `.uni` gives 1, and Python `fsd_uni(type="continuous")` gives 1.
    - The largest ratio_x - ratio_y on a dense grid is +0.00297 at t=0.50; at the knots it is 0
      (`W/fsdc_knots_verify.R`).
    - L1/(L1+U1) is linear-fractional between knots, not linear. About 0.26% of positives were
      false (26 of 10,104, `W/fsdc_knots_search.py`).
- Which side is right: first principles. The book (ch. 15, "Connection to Partial Moments") states
  `LPM_2(t,X) <= LPM_2(t,Y)` "for all t". SSD (piecewise linear) and discrete FSD (step function)
  are exact at the knots; TSD and continuous FSD are not.
- Proposed fix (TSD): in every interval [t_i, t_{i+1}) where a = k_x/n_x - k_y/n_y < 0, also
  evaluate D(t) at its vertex t* = (P1_x/n_x - P1_y/n_y)/a when t* lies inside the interval. That
  is the root of L1_x - L1_y. For continuous FSD, evaluate the turning point of the
  linear-fractional difference in the same way. This is O(1) extra work per interval and changes
  results only in rare cases.

### SDA-06: Python efficient set and cluster results depend on unrelated columns and on a 75-column switch (the small-matrix path uses one global grid)
- Type: parity-mismatch / docs
- Severity: medium
- Confidence: verified
- Where: `stochastic_dominance.py:299-300` and `:829` (`grid = np.unique(sorted_values.reshape(-1))`),
  plus the switch at `:28` and `:282`. `docs/conventions.md` says SD uses "each pair's merged
  threshold grid rather than one global all-column grid".
- Evidence: `W/eff_threshold75.py` and `W/eff_threshold75.R`, using x and y from SDA-05 plus
  z = (-3, 0.16, 0.2).
  - `sd_efficient_set([x, y], 3)` gives [0], but `sd_efficient_set([x, y, z], 3)` gives [0, 1].
    z's knot at 0.16 exposes the violation, and y comes back even though z is itself dominated.
  - Padding with dominated junk columns: 74 columns give [0, 1], and 75 columns give [0].
  - R gives "X_1" in every case.
  - For degree-1 continuous, the same effect appears: a random 120x6 integer matrix gives Python
    [111, 63] vs R [111] (`W/eff_cmp.py`, m39). That case combines this issue with SDA-07.
- Which side is right: R's pairwise semantics, until SDA-05 is fixed. The result should not depend
  on which other columns are present.
- Proposed fix: evaluate each pair on its own merged grid, as the prefix path already does. Or fix
  SDA-05, after which the global grid and the pair grid agree exactly. For degree 1 discrete and
  degree 2 the two grids are already equivalent.

### SDA-07: R `NNS.FSD.uni(type="continuous")` says x dominates y and y dominates x when the ratio curves coincide at the knots; Python says 0 both ways
- Type: r-bug / parity-mismatch
- Severity: medium
- Confidence: verified
- Where: R `src/SD.cpp:105-127`. The only equality guard is `identical_samples`; there is no
  curve-equality guard, unlike `R/FSD.R:71`.
- Evidence: `W/fsdc_probe.R`.
  - x = (0, 0, 3, 3), y = (1, 1, 3, 0):
    - R `.uni` continuous gives 1 for x over y and 1 for y over x, while R `NNS.FSD` continuous
      says "NO FSD EXISTS". Python gives 0 and 0.
    - `NNS.SD.efficient.set(cbind(x, y), 1, "continuous")` gives "x"; Python gives [0, 1].
  - x = (0, 0), y = (0, 3): R `.uni` continuous gives 1 and 1. The degenerate 0/0 ratio is
    mapped to 0 at t=0.
  - Out of 400 random small integer pairs, 44 and 48 cases mismatch in the two directions
    (`W/sd_cases_cmp.py`).
- Which side is right: Python. Mutual dominance between different distributions is impossible.
  R's own bidirectional `NNS.FSD` rejects it.
- Proposed fix (R): add `if (curves identical) return 0` to the continuous branch, as R's
  `NNS.FSD` does.

### SDA-08: R `.uni` routines silently return wrong answers for samples of unequal length
- Type: r-bug
- Severity: medium
- Confidence: verified
- Where: R `src/SD.cpp:88` (`int ia=0, ib=0, m=a.m; // assume same m`). b is walked with a's
  length, so it reads past b's end or ignores b's tail. There is no length check in
  `R/Uni_SD_Routines.R`.
- Evidence: `W/uni_uneq.R` and `W/sd_cases_cmp.py`.
  - x = (1, 2) and y = (1, 1, 2, 2) are identical distributions. Yet `NNS.SSD.uni` and
    `NNS.TSD.uni` return 1 in **both** directions, while R `NNS.TSD` says "NO TSD EXISTS".
  - About 100 of 200 random unequal-length integer pairs disagree with R's own bidirectional test.
  - Python's `*_uni` agrees with R's bidirectional functions and with first principles in all
    cases, as its documentation says.
- Which side is right: Python.
- Proposed fix (R): walk each sample with its own length (`ia < a.m`, `ib < b.m`), or
  `stop()` when the lengths differ.

### SDA-09: Dominance verdicts have no floating-point tolerance, so the textbook SSD case flips on rounding noise (R and Python)
- Type: r-bug (shared design)
- Severity: medium
- Confidence: verified
- Where:
  - R `R/SSD.R:43-58`, `R/TSD.R`, `src/SD.cpp:125,131,140,150`.
  - Python `stochastic_dominance.py:765-775` and the prefix paths.
- Evidence: `W/mps_probe.R` and `W/sd_cases_cmp.py`. The test case is x = qnorm(ppoints(100)) and
  y = 2*x, an exact mean-preserving spread, so the expected answer is "X SSD Y".
  - R `NNS.SSD` says "NO SSD EXISTS": the LPM1 difference reaches +8.9e-16 and
    mean(y) - mean(x) = -3.7e-18.
  - On the **same data**, R `NNS.SSD.uni` gives 1 and R `NNS.TSD` gives "X TSD Y", because the
    prefix-sum arithmetic differs.
  - Shifting both samples by 0.37 changes `.uni` to 0, so the verdict is not translation invariant.
  - x = (0.1, 0.7) vs y = (0, 0.8): "NO SSD", because 0.1 + 0.7 = 0.7999999999999999. Multiplying
    both by 10 gives "X SSD Y".
  - Python agrees with R in most of these cases but not all. For the n=100 quantile case Python
    `tsd` is 0 and R `NNS.TSD` is 1.
- Which side is right: first principles. Ties at the 1e-16 level should not decide dominance.
- Proposed fix: compare with a scale-aware tolerance, for example
  `Lx > Ly + 8*eps*max(1, |t|, range)`, and use the same tolerance in the mean gate. The
  curve-equality guard then also needs a tolerance. This changes public results only in
  near-tie cases.

### SDA-10: ANOVA `Certainty` is documented as a "calibrated probability" and the inverse of a p-value; it is neither
- Type: docs (R and Python)
- Severity: medium
- Confidence: verified
- Where: R `R/ANOVA.R:8-19` (man page), `book/chapter-15...Rmd:583,624-630`, and the Python
  `nns_anova` docstring.
- Evidence: `W/anova_calib.py` (Python matches R to 1e-7 here).
  - Two samples from the same N(0,1) distribution:

    | Mode | n | Median Certainty | 5-95% range | P(C<0.10) |
    |---|---|---|---|---|
    | means.only | 30 | 0.73 | 0.42-0.91 | 0 |
    | means.only | 100 | 0.85 | — | 0 |
    | means.only | 1000 | 0.96 | — | 0 |
    | full | 100 | 0.81 | — | 0 |

  - The values are far from Uniform(0,1) and depend on n.
  - Under a 0.3 sd shift with n=100, the t-test detects the difference 55% of the time, but
    Certainty is below 0.05 in **0%** of runs (median 0.50-0.64).
  - Finance: SPY vs TLT has KS p = 1.8e-9 but Certainty 0.914 ("strong evidence of similarity" by
    the man page's rule). NVDA vs TLT has KS p = 2e-96 but Certainty 0.40.
  - Identical samples give Certainty 0.96 (n=50 each), 0.81 (n=10) and 0.44 (n=3), because of
    `pop.adjustment` (`R/Binary_ANOVA.R:84`). Smaller samples look *more* different, which runs
    against how evidence should behave.
- Which side is right: the documentation overclaims. Certainty is a descriptive similarity index.
- Proposed fix: reword the docs ("similarity index in [0,1], not a probability; not comparable to
  p-value thresholds"). Optionally offer a permutation calibration. The docs-only change does not
  alter results.

### SDA-11: The `identical` k x k x n boolean tensor gives O(k^2 n) memory in `sd_efficient_set` and `nns_sd_cluster`
- Type: perf
- Severity: medium
- Confidence: verified
- Where: `stochastic_dominance.py:334-337`, `:364-367` and `:376-379`
  (`np.all(sorted.T[:, None, :] == sorted.T[None, :, :], axis=2)`).
- Evidence: `W/big_eff.py`, 500 columns x 4232 days (17 years of daily data for 500 stocks).
  - Peak RSS is 1123-1156 MB, almost all of it this tensor (500 * 500 * 4232 bytes = 1.06 GB).
  - 1000 columns would need about 4.2 GB, and 2000 columns about 17 GB.
  - R's per-pair `identical_samples` uses O(1) memory.
  - Timings at k=500 (Python vs R single-thread): degree 1 7.5 s vs 11.7 s, degree 2 1.8 s vs
    6.3 s, degree 3 1.4 s vs 6.6 s. The efficient-set sizes match R (451/27/12).
- Proposed fix: hash each sorted column (`sorted[:, j].tobytes()`) and mark identical only within
  hash buckets, or call `np.array_equal` lazily for candidate pairs. Results do not change.

### SDA-12: `nns_norm(pandas.DataFrame)` crashes with a misleading error
- Type: api (python-bug)
- Severity: medium
- Confidence: verified
- Where: `norm.py:43-46`. A non-ndarray input is iterated as a sequence of series, and a
  DataFrame iterates over its column *names*.
- Evidence: `W/norm_py.py`: `nns.nns_norm(prices_df)` raises `ValueError: could not convert string
  to float: 'SPY'`. A DataFrame is the most common input for price levels, and R accepts a
  data.frame (`R/Normalization.R:44-48`).
- Proposed fix: `if hasattr(x, "to_numpy") and getattr(x, "ndim", 1) == 2: x = x.to_numpy(float)`.
  Optionally return a DataFrame with "<name> Normalized" columns, as R does. Numeric results do
  not change.

### SDA-13: R `NNS.FSD(type="continuous")` returns NA when a sample is constant
- Type: r-bug
- Severity: low
- Confidence: verified
- Where: R `R/FSD.R:55-75`. `LPM.ratio(1, t, x)` is 0/0 = NaN at t = the constant, and
  `any(NaN > ...)` gives NA, which `ifelse` returns.
- Evidence: `W/fsdc_probe.R`: `NNS.FSD(c(0,0), c(0,3), "continuous")` returns `NA`. The discrete
  test gives "Y FSD X". R's C++ `.uni` maps 0/0 to 0. Python has no bidirectional continuous FSD
  (see SDA-17).
- Proposed fix: set the ratio to 0 where L+U == 0, as the C++ does, or use `na.rm`.

### SDA-14: Python ANOVA uses its own optimiser-based LPM/UPM VaR instead of the exact `nns.var` version, so Certainty differs from R by up to about 3e-7
- Type: parity-mismatch / dead-code (duplicate implementation)
- Severity: low
- Confidence: verified
- Where: `anova.py:332-377` (`_lpm_var` / `_upm_var` call `minimize_scalar`). `nns.var.lpm_var`
  already implements R's exact integer-degree inversion.
- Evidence:
  - For SPY at the 25% level, `anova._lpm_var` gives -3.556843063e-03, while `nns.var.lpm_var` and
    R `LPM.VaR` both give -3.556845568e-03.
  - Certainty differences against R: SPY vs TLT 1.3e-7, finance pairwise matrix 2.7e-7, unequal
    list 1e-7 (`W/anova_py.py`).
  - means.only results, which do not use VaR, match to 1e-16.
- Proposed fix: import `lpm_var` and `upm_var` from `nns.var` and delete the local copies. The
  public numbers then match R exactly.

### SDA-15: `sd_efficient_set(degree=1.5)` silently runs third-order dominance; `degree="2"` raises a TypeError
- Type: python-bug
- Severity: low
- Confidence: verified
- Where: `stochastic_dominance.py:270` validates only `1 <= degree <= 3`. In `_sd_curve_table`
  (`:833-846`), a degree that is neither 1 nor 2 falls through to the LPM2 (TSD) curve, and
  `degree > 1` turns on the mean gate.
- Evidence: `W/sd_api.py`. With degree=1.5 Python returns the 7-asset set [5, 0, 4, 6, 7, 1, 8].
  R (`as.integer`, so FSD) returns all 11 assets.
- Proposed fix: `if degree not in (1, 2, 3): raise ValueError`, or apply `int()` like R. The same
  applies to `nns_sd_cluster`, which uses `int(degree)` and is consistent with R.

### SDA-16: `nns_sd_cluster(min_cluster=0)` appends an empty final cluster
- Type: parity-mismatch
- Severity: low
- Confidence: verified
- Where: `stochastic_dominance.py:183-185`. Assigning `[]` creates the key; in R, assigning
  `NULL` to a list element is a no-op.
- Evidence:
  - `W/clu_min.R` with 5 shifted copies: Python returns `{..., 'Cluster_5': ['E'],
    'Cluster_6': []}`, while R ends at `Cluster_5`.
  - The 11 assets give the same extra key (`Cluster_4: []`).
- Proposed fix: only add the final cluster when `active` is not empty.

### SDA-17: API gaps: no `type` argument on `fsd`, DataFrame names ignored, indices returned instead of names, and more
- Type: api
- Severity: low
- Confidence: verified (`W/sd_api.py`, `W/anova_api.py`, `W/ss_py.py`)
- Where: `stochastic_dominance.py:63,106-136,259`, `anova.py:67-82`, `stochastic_superiority.py:25`.
- Evidence:
  - `nns.fsd(x, y, type="continuous")` raises a TypeError. R's `NNS.FSD` has `type`; only
    `fsd_uni` takes it.
  - `sd_efficient_set(DataFrame)` returns positional indices, where R returns column names.
  - `nns_sd_cluster(DataFrame)` labels the columns `X_1..X_11` instead of the DataFrame's column
    names.
  - `nns_anova(DataFrame, pairwise=True)` returns an unlabeled ndarray, where R uses dimnames.
  - A misspelt `type="contin"` silently runs the discrete test (R warns).
  - A single-column DataFrame input to `fsd` raises "x must be 1D"; R accepts single-column
    objects.
  - `nns_ss(confidence_interval=np.True_)` is rejected; `reps=2.5` raises a TypeError.
- Proposed fix: read `.columns` and `.name`; squeeze single-column frames; add `type` to `fsd`
  (R's bidirectional continuous path, with the SDA-13 fix); validate `type`; accept `np.bool_`.

### SDA-18: ANOVA missing and infinite values: R propagates NaN, Python silently drops NaN and ±Inf
- Type: parity-mismatch
- Severity: low
- Confidence: verified
- Where: `anova.py:380-385` (`values[np.isfinite(values)]`).
- Evidence:
  - airquality Ozone vs Solar.R (both contain NA): R `NNS.ANOVA(..)$Certainty` gives NaN; Python
    gives 6.08e-05.
  - Appending `inf` to a group leaves the Python Certainty unchanged (0.768).
  - In the multi-group matrix path, R also drops NAs (`na.omit`), and there the two sides agree.
- Which side is right: Python's dropping is more useful, but dropping ±Inf silently is not, and
  `docs/conventions.md` documents neither behaviour.
- Proposed fix: drop NaN only, raise on ±Inf, and document the behaviour.

### SDA-19: R SD routines emit spurious recycling warnings for samples of unequal length
- Type: r-bug
- Severity: low
- Confidence: verified
- Where: R `R/FSD.R:44`, `R/SSD.R:36`, `R/TSD.R:35`, `R/Uni_SD_Routines.R:36,79,118`
  (`anyNA(cbind(x, y))`).
- Evidence: `W/uni_uneq.R` prints "number of rows of result is not a multiple of vector length".
- Proposed fix: `anyNA(x) || anyNA(y)`.

### SDA-20: R `NNS.SD.cluster` docs say each SD set is peeled off only if its size reaches `min_cluster`; the code never checks
- Type: docs (r)
- Severity: low
- Confidence: verified
- Where: `R/SD_Cluster.R` (@details vs the loop at lines 60-86).
- Evidence: `W/clu_min.R` with `min_cluster=2` gives Cluster_1 = "A", Cluster_2 = "B" and
  Cluster_3 = "C", each of size 1. Python matches R.
- Proposed fix: either enforce the rule (merge undersized sets) or correct the docs.

### SDA-21: `NNS.norm`: a zero-mean column gets a mean of 1e-10, which gives scales of about 1e10; the `linear` docs are inverted
- Type: r-bug / docs (Python mirrors it)
- Severity: low
- Confidence: verified
- Where:
  - R `R/Normalization.R:50` (`m[m==0] <- 1e-10`) and `:6`, where the @param says
    "linear = FALSE (default) Performs a linear scaling normalization". In fact `linear=FALSE` is
    the correlation/dependence-weighted (nonlinear) scaling.
  - Python `norm.py:51-52`.
- Evidence: `W/norm_py.py`: `cbind(a=c(-1,0,1), b=c(1,2,3))` normalises to
  a = (-1e10, 0, 1e10) on both sides. A constant column gives all-NaN output, with a
  `corrcoef` RuntimeWarning in Python.
- Proposed fix: reject or warn on a zero or negative mean (mean-ratio scaling assumes positive
  levels), and fix the @param text.

### SDA-22: R ANOVA doc errors on `robust`
- Type: docs (r)
- Severity: low
- Confidence: verified
- Where: `R/ANOVA.R:29`, which describes `robust` as "100 independent random permutations ... for
  pairwise analysis only".
- Evidence:
  - The code draws bootstrap resamples, not permutations (`sample.int(l, replace=TRUE)`).
  - `robust` only applies when `treatment` is supplied. It is ignored for the multi-group and
    `pairwise=TRUE` paths, which is the opposite of "pairwise only". Python behaves the same way.
- Proposed fix: rewrite the @param.

### SDA-23: Dead code
- Type: dead-code
- Severity: low
- Confidence: verified (grep across `src/` and `tests/`)
- Where:
  - Python functions with zero references: `_dominates_from_curves` (`stochastic_dominance.py:800`),
    `_prefix_directional_candidates_to_target` (`:672`), `_sd_efficient_set_names` (`:303`).
  - Python functions used only by tests: `_dominance_matrix_from_precomputed` (`:498`),
    `_dominance_matrix_from_prefix_pairs` (`:518`), and their helpers
    `_curve_comparison_chunks` and `_prefix_pair_candidate_matrix`.
  - Python `_lower_area_share` (`anova.py:318`) computes ratio/(ratio + (1 - ratio)), which is
    just `lpm_ratio(1, ...)`.
  - R: `LPM_ratio` in `R/ANOVA.R:229-233` is computed and never used, and so is `full_matrix` at
    `:116`.
- Proposed fix: delete them or move them to the test helpers.

### SDA-24: Fixed overhead in the Python pairwise SD tests at small n
- Type: perf
- Severity: low
- Confidence: verified (`W/time_py.py`, `W/time_r.R`, `W/prof_py.py`; noisy shared CPU)
- Evidence:

  | Test | n | Python | R | Ratio |
  |---|---|---|---|---|
  | `ssd_uni` / `tsd_uni` vs R `.uni` C++ | 1e3 | 1.7-2.9 ms | 0.05-0.1 ms | about 20-30x |
  | `ssd_uni` / `tsd_uni` vs R `.uni` C++ | 1e4 | 6-8 ms | 1.5 ms | about 4x |
  | Bidirectional `fsd` / `ssd` / `tsd` vs R-level `NNS.FSD` etc. | 1e3 | 2 ms | 0.6-0.8 ms | about 2.5-3x |
  | Bidirectional `fsd` / `ssd` / `tsd` vs R-level `NNS.FSD` etc. | 1e4 | 4.7 ms | 6 ms | about 0.8x |
  | Bidirectional `fsd` / `ssd` / `tsd` vs R-level `NNS.FSD` etc. | 3e4 | 13-16 ms | 16-17 ms | about 0.8-0.9x |
  | 11-asset efficient set, degrees 1/2/3, full history (n=4232) | — | 17.6 / 22.2 / 25.5 ms | 10.7 / 8.5 / 8.3 ms | about 1.6-3x |
  | 11-asset efficient set, continuous | — | 22 ms | 12 ms | about 2x |
  | `nns_sd_cluster` degree 2 | — | 17 ms | 11 ms | about 1.5x |
  | 200 columns x 1000 rows | — | 63-80 ms | 230-444 ms | Python faster |

  - `cProfile`: 94% of `ssd_uni` time is in `core.lpm` over the 2n merged grid, called twice. The
    merged two-pointer walk, or `_pair_curve_values_at_thresholds` with prefix sums, would be
    O(n) with fewer passes.
- Proposed fix: route `_dominates_uni` through the prefix-sum evaluator that already exists. Results
  do not change apart from SDA-09-level rounding.

---

## Coverage (checked and found fine)
- **FSD/SSD/TSD vs R on finance data.** All 110 ordered pairs of the 11 daily equity/ETF returns
  (n=4232) match exactly for bidirectional FSD, SSD and TSD and for the `.uni` variants (discrete
  and continuous FSD). Counts: 0 FSD, 15 SSD and 15 TSD dominances. The exact interior-extremum
  check (`W/tsd_exact.py`) confirms that all 15 TSD verdicts are true (`W/sd_fin_py.py`).
- **Textbook cases.**
  - Quantile N(0.5,1) vs N(0,1) gives FSD=SSD=TSD=1 at n=10, 100 and 1000.
  - Equal distributions (a permutation of the same sample, constants, length 1) give 0.
  - Mean-preserving spreads are detected except for the rounding cases in SDA-09.
  - Random N(0.5,1) vs N(0,1) *samples* show FSD in only 50% of runs at n=100 and 63% at n=1000,
    and SSD in 78% / 87%. This is expected from finite-sample ECDF crossings and the min gate with
    no tolerance; it is not a port bug.
- **First principles.** Discrete FSD (ECDF ordering) and SSD (LPM1 ordering at the knots, which is
  exact) agree with R and Python on 400 equal-length and 200 unequal-length integer cases with ties
  (`W/sd_cases_cmp.py`), apart from the tolerance cases in SDA-09.
- **Efficient set.**
  - The efficient set for degrees 1, 2 and 3 (discrete) and degree 1 continuous matches R on the
    11 daily assets, the 14 monthly series and 40 random matrices of 5, 20, 80 or 120 columns
    (integer, rounded-normal and t-distributed data). The only mismatch is SDA-06/07 m39.
  - R's ordering (LPM at the global maximum, ties broken by index) is reproduced.
  - At k=500 the set sizes match R.
- **`NNS.SD.cluster`.** It matches R for degrees 1-3 with `min_cluster` 1, 2, 3 and 20, for
  continuous FSD, and for the dendrogram merge, height, order and labels (degree 2). Duplicate
  columns and the single-column dendrogram case also match.
- **`NNS.SS`.** Finance pairs match to 5.6e-16, as do ties (integer samples of lengths 100 and 77),
  NaN omission and ±Inf. The CI structure matches, and CI values agree within bootstrap noise
  (Python [0.442, 0.557] vs R [0.445, 0.556], SPY/TLT n=500).
  - Timings are on par: 0.6 ms vs 1.1 ms (n=4232), 339 ms vs 396 ms (n=1e6), and 4.3 s vs 5.9 s
    with CI (999 reps).
  - A tiny tied sample fails inside meboot on both sides (owned by the meboot group).
- **`NNS.ANOVA`.**
  - Binary, multi-group and pairwise runs match R for every combination of means.only and medians,
    on iris Sepal.Length by Species and on the 11-asset matrix. Agreement is within 1e-16 for
    means.only and 1e-7 otherwise (SDA-14).
  - Unequal-length lists, the constant/degenerate NaN convention, the medians and means effect-size
    CI ("both", 50-seed mean within noise of R), and validation of `tails` and
    `confidence_interval` were all checked. Timings are comparable (pairwise on 11 assets: 665 ms
    vs 651 ms; binary with CI: 111 ms vs 661 ms).
  - Classical comparison: the iris one-way F-test gives p = 1.7e-31, consistent with NNS
    Certainty 0.0011.
- **`NNS.norm`.** Linear and nonlinear normalisation of 11-asset price levels (n=4233, NNS.dep
  branch) and of 5 assets (`cor` branch) match R to 5e-15. The 11x11 dependence matrix matches to
  7e-16. Lists of equal and unequal length, a list of Series, and the zero-mean case all match.
  Time: 213 ms vs 339 ms.
