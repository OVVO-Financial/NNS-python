# DEP findings: dependence, correlation, copula, causation, partition map

Scope: Python `nns.dependence` (`nns_dep`, `nns_cor`), `nns.copula`, `nns.causation`
(`nns_causation`, `causal_matrix`), `nns.part`, and the native bindings they call
(`_nnscore_bindings.cpp`: `nns_dep_pair`, `gravity_exact`, `nns_part_off`, `copula_nd`).
R: `R/Dependence.R`, `Copula.R`, `Causation.R`, `Uni_Causation.R`, `Causal_matrix.R`,
`Partition_Map.R`, `Normalization.R` (helper), `src/NNS_dep.cpp`, `src/NNS_part.cpp`,
`src/central_tendencies.cpp` (gravity helper), man pages, and book chapters 10, 12 and 14.

All scripts are in `/home/user/audit/work/DEP/`. The main A/B harness is `gen_cases.py` (41 cases
written to `cases/*.csv`), `run_py.py`, `run_r.R` and `compare.py`. Run them as
`.venv/bin/python run_py.py; Rscript run_r.R; .venv/bin/python compare.py`.

Short version: Python matches R bit-for-bit on continuous data for `nns_dep`, `nns_copula`
(bivariate and multivariate), `nns_causation` and `causal_matrix`. On `nns_part` it matches on
all 2640 option combinations tested. Most of the serious problems are in the shared algorithm, so
they affect R and Python alike: the sign is lost under perfect anti-dependence, the `tau` lag in
NNS.caus does nothing, causation depends on location, the independence floor is large, the
p-values are wrong, and the dependence measure collapses on step functions. The Python-only
problems are a segfault, NaN passing through silently, a bin cap in the internal gravity, memory
blow-up and slow non-default partitions.

---

### DEP-01: NNS.dep / nns_dep returns Correlation = +1 for perfectly anti-dependent data (sign is lost)
- Type: r-bug (shared; Python reproduces it exactly)
- Severity: critical
- Confidence: verified
- Where: R `src/NNS_dep.cpp:84-86` (`if (d0_Co == 1.0 || d0_Co == 0.0) return 1.0;` inside
  `copula_signed`). Python native `src/nns/_nnscore_bindings.cpp:666`; Python fallback
  `src/nns/dependence.py:139-140`. The same pattern is in the copula code at `Copula.R:80` and
  `copula.py:70`, but there it is harmless because that copula is unsigned.
- Evidence: `work/DEP/repro_core.R`, `run_py.py`/`run_r.R` cases `y_neg_x`, `int_seq_rev`, `len2`.
  - `NNS.dep(x, -x)` with x ~ N(0,1), n=1000 gives Correlation = **1**, Dependence = 1 (Pearson -1).
  - `NNS.dep(1:100, -(1:100))` gives Correlation = **0.370**, Dependence 0.990.
  - `NNS.dep(c(1,2), c(2,1))` gives Correlation = **+1**.
  - Python gives identical numbers (`nns_dep(x, -x)` = `{'Correlation': 0.9999999999999999, ...}`).
  - Mechanism: in any partition cell where every point is discordant, d0_Co == 0. The early
    return then hands back an unsigned `+1.0` and skips `* slope_sign`. With y = -x, every X-only
    cell is fully discordant.
- Which side is right: neither. A signed nonlinear correlation must be -1 for y = -x.
- Proposed fix: `if (d0_Co == 1.0) return 1.0; if (d0_Co == 0.0) return -1.0;`. Better still, return
  `slope_sign` (and handle a constant cell as in DEP-06). With an R-level patch
  (`work/DEP/fix_sign_check.R`) the results become dep(x,-x) = -1/1 and dep(1:100,-(1:100)) =
  -0.990/0.990, while dep(x,x), dep(x,-x^3) and dep(x,x+e) are unchanged. This changes public
  results only for inputs that hit the early return.

### DEP-02: NNS.caus "tau" lag does no lagging; lagged causation (y_t = f(x_{t-tau})) cannot be detected
- Type: r-bug (shared design; Python ports it exactly)
- Severity: critical
- Confidence: verified
- Where: R `R/Uni_Causation.R:14-31` (the lag matrix goes into `NNS.norm(...)[ ,1]`) and
  `R/Normalization.R:44-58` (NNS.norm only multiplies each column by a scalar). Python
  `src/nns/causation.py:109-128` (`_tau_normalized`).
- Evidence: `work/DEP/caus_sim.py` (30 reps per design, n=500) and `work/DEP/caus_r_check.R`
  (R gives the same numbers as Python on the saved reps).
  - `x.norm.tau` is exactly `c * x[(tau+1):n]`: in R, `cor(x.norm.tau, x[t]) = 1` and the ratio is
    the constant 0.279479. So tau only drops the first tau rows and rescales; x and y are never
    offset against each other.
  - Design y_t = 0.8 x_{t-1} + 0.3 e_t with iid x: the share of reps reporting x→y is 0.33 at
    tau=0, **0.43 at tau=1 (the true lag)** and 0.33 at tau=3.
  - With AR(1) x the share is 0.43/0.53/0.57. With nonlinear y_t = x_{t-1}^2 + e it is 0.60/0.70/0.67.
  - Independent noise gives 0.40/0.53/0.43, so the lagged designs cannot be told apart from noise.
- Which side is right: the book (ch.14, "Lagged Co-Partial Moments" and "Removing Internal
  Dynamics") defines CoUPM^(tau) on (X_{t-tau}, Y_t) and describes a lag-aware normalisation.
  The code computes neither.
- Proposed fix: build the lagged pairs explicitly. For the x→y direction use
  `x_lag = x[1:(n-tau)]`, `y_lead = y[(tau+1):n]` (and the reverse for y→x). Optionally pre-whiten
  each series on its own lags (for example residuals of `NNS.reg` on its own lag matrix) before
  the cross-normalisation. This changes every result with tau > 0.

### DEP-03: NNS.caus is not location/scale invariant: NNS.norm can zero or flip a variable's sign, and a constant shift reverses the causal direction
- Type: r-bug (shared)
- Severity: high
- Confidence: verified
- Where: R `R/Normalization.R:44-58`: `RG <- m %o% (1/m)`; `scales <- colMeans(RG * |cor|)`;
  `m[m==0] <- 1e-10`. This is called from `Uni_Causation.R:28-43`. Python `nns.norm.nns_norm` via
  `causation.py:97`.
- Evidence: `work/DEP/norm_sign.R` plus the Python one-liners in the session log.
  - x ~ N(1,1), y = x - 1.5 + noise (cor 0.957, means +0.99 and -0.49). After NNS.norm the
    correlation is **-0.957**, because the scale factor for y is -0.465 and y's sign is flipped.
  - `NNS.caus(x, y)` gives C(x→y) = 0.0095. `NNS.caus(x, y + 10)` gives C(x→y) = **1.016**.
    Python is identical.
  - `NNS.caus(x, 2*x + 1)` (an invertible linear map) gives C(x→y) = **1.49**, while
    `NNS.caus(x, x)` gives 0.
  - With y = -x the scales are about 0, so both series collapse to roughly constant vectors.
    R then returns 0.4999/0.4999 and Python returns 0/0; both are garbage.
  - Finance (`SPY`, `TLT`, last 1000 days): the result is C(TLT→SPY) = 0.250. Adding 1.0 to TLT
    returns flips it to **C(SPY→TLT) = 1.877**.
  - Zero-mean series such as returns make `RG` explode, because it is a ratio of near-zero means.
- Which side is right: a causation measure must not depend on the units or origin of a variable.
  The book (ch.14) says NNS.norm is an "empirical CDF transformation", which would be invariant.
  The code is not (see DEP-21).
- Proposed fix: in Uni.caus, replace NNS.norm with a rank/ECDF transform, as the book describes
  (`LPM.ratio(0, x, x)` per column). At minimum, use `abs(m)` and guard against means near zero.
  This changes all NNS.caus outputs.

### DEP-04: NNS.caus(p.value=TRUE) one-sided p-values ignore direction (the "signed" statistic is always >= 0)
- Type: r-bug (Python has no p.value; see DEP-15)
- Severity: high
- Confidence: verified
- Where: R `R/Uni_Causation.R:99-104` (`signed_from_cp` returns `cp[3]` and ignores its name) and
  `R/Causation.R:71,100-102`.
- Evidence: `work/DEP/caus_p_r.R`, with x ~ N(0,1), y = x^2 + noise, n=300 and nperm=50.
  - `NNS.caus(x, y, p.value=TRUE)` gives a `C(x--->y)` label, observed_signed 1.428 and
    **x_causes_y p = 0.0196**.
  - `NNS.caus(y, x, p.value=TRUE)` gives a `C(y--->x)` label, observed_signed 1.428 and
    **x_causes_y p = 0.0196**. That is the same claim with the roles swapped.
  - Every value in the null distribution is >= 0 (range 0.056 to 1.31).
- Which side is right: the docs say the test is "for the signed causation from x -> y". The code
  must flip the sign when the label is `C(y--->x)`.
- Proposed fix: in `signed_from_cp`, `v <- cp[3]; if (names(cp)[3] == "C(y--->x)") v <- -v`, then
  apply the cap. `NNS.caus.matrix` already does this flip. This changes the p-values reported.

### DEP-05: NNS.dep has a large independence floor that decays only like n^(-1/4), and moderate linear dependence barely clears it
- Type: r-bug (shared design)
- Severity: high
- Confidence: verified
- Where: R `src/NNS_dep.cpp:107-112` (sqrt of a mean of absolute deviations, averaged over
  X-only cells of size about n/16 because `obs.req = max(8, n/8)`). Python is identical.
- Evidence: `work/DEP/indep_bias.py` (40 reps; R matches Python to 1e-15 on spot checks
  `indep_n100`, `indep_n1000`, `indep_n10000`, `indep_exp_10k`). Mean Dependence for independent
  normals:

  | n | 30 | 100 | 300 | 1e3 | 3e3 | 1e4 | 1e5 |
  |---|---|---|---|---|---|---|---|
  | Dependence | 0.649 | 0.600 | 0.447 | 0.323 | 0.242 | 0.180 | 0.105 |
  | NNS.copula | 0.415 | 0.297 | 0.216 | 0.168 | 0.130 | 0.091 | 0.052 |
  | Pearson abs(r) | 0.139 | 0.067 | 0.036 | 0.024 | 0.015 | 0.007 | 0.003 |

  At n=1000: y = x + N(0,1) (Pearson 0.72) gives Dependence **0.443** against a null of 0.32.
  The noisy circle gives 0.291 and the exact circle 0.296, both *below* the null. Daily
  SPY/QQQ returns (Pearson 0.93) give 0.553. In the finance matrix (`work/DEP/fin_r.R`), BTC–GLD
  (Pearson 0.10) scores 0.324, above SPY–TLT (Pearson -0.17) at 0.286.
- Which side is right: a dependence coefficient should be about 0 under independence and should
  rank stronger dependence higher. Users cannot read NNS.dep values without a null reference,
  which the code does not provide (the p.value path is also biased; see DEP-13).
- Proposed fix (design): subtract or standardise by the permutation null, for example
  `(D - E0[D]) / (1 - E0[D])`. Use cells larger than n/16, or debias the per-cell
  abs(deviation) terms, which are positive under the null. At minimum, document the n-dependent
  floor. This changes results.

### DEP-06: Cells where one variable is constant score 0 dependence, so deterministic step functions look independent; both-constant cells score +1
- Type: r-bug (shared)
- Severity: high
- Confidence: verified
- Where: R `src/NNS_dep.cpp:111` (`slope_sign = varx==0 ? 0 : sign(cov)`; a constant y gives
  cov = 0, so the sign is 0) and `:86` (both constant gives d0_Co = 1, which returns +1).
  Python `_nnscore_bindings.cpp:666-688` and `dependence.py:139,173`.
- Evidence: cases `step`, `step3`, `discrete_dep` and `repro_core.R`.
  - With u ~ U(-3,3) and n=1000, `NNS.dep(u, as.numeric(u>0))` gives **0.097/0.097** and
    `NNS.dep(u, floor(u))` gives **0.358**. Independent noise at the same n gives 0.30.
    `NNS.copula` gives 0.988 on the same step data.
  - For discrete x with y = x + {0,1}, every cell has constant x, so it contributes 0. Only cells
    where both variables happen to be constant contribute, and those contribute +1 (Correlation
    0.202 in Python, 0.099 in R; see DEP-08).
- Which side is right: y = f(x) (y constant within an x-cell) is perfect dependence of y on x.
  A cell with no variation carries no information and should not count as either 0 or +1.
- Proposed fix: in `copula_signed`, return NaN for a cell with zero variance in x or y and let the
  caller drop it and renormalise the weights. For the asym x→y direction, treat a constant-y cell
  as fully determined (abs(cop) = 1). This changes results on discrete or step data.

### DEP-07: Python nns_copula segfaults on empty input and silently returns 1.0 for NaN input
- Type: python-bug
- Severity: high
- Confidence: verified
- Where: `src/nns/copula.py:119` (the bivariate path uses `co_moments._as_pair`, which checks
  neither for empty input nor for finite values) and the native `copula_nd`
  (`_nnscore_bindings.cpp:890`), which assumes n >= 1.
- Evidence: `work/DEP/segv_copula.py`: `nns.nns_copula(np.array([]), np.array([]))` crashes with
  exit code **139 (SIGSEGV)**. From `api_py.py`: with one NaN in x, `nns_copula` returns **1.0**,
  while R raises "You have some missing values". With Inf it returns 0.728, while R raises
  "`data` must be finite". The matrix form does reject non-finite input, so the two forms are
  inconsistent.
- Which side is right: R (it rejects the input).
- Proposed fix: use `nns._helpers._as_pair` (which checks size and finiteness) in
  `copula._prepare`, validate `target_x`/`target_y` as finite, and add an `n == 0` guard in
  `copula_nd`. Valid inputs are unaffected.

### DEP-08: R NNS.dep compares a floating-point product to 1 exactly, so dep(x, x) on binary data depends on the group count (Python differs)
- Type: r-bug / parity-mismatch (Python is right)
- Severity: medium
- Confidence: verified
- Where: R `src/NNS_dep.cpp:84-86` (`d0_Co = (cupm+clpm) * (1.0/n); if (d0_Co == 1.0)`). Python
  computes count/n exactly (`_nnscore_bindings.cpp:663`).
- Evidence: `work/DEP/repro_core.R`.
  - `b <- c(rep(0,51), rep(1,49)); NNS.dep(b, b)` gives **0.51 / 0.755** in R, because
    `49*(1/49) != 1`.
  - With 50/50 it gives 1 / 1. Python gives 1/1 for both.
  - 82 of the integers 1..1000 fail the test (49, 98, 103, 107, 161, 187, 196, 412, ...).
  - The same cause explains the mismatches in cases `binary_eq` (R 0.588/0.794, Python 1/1) and
    `discrete_dep` (R 0.099/0.496, Python 0.202/0.548).
- Which side is right: Python. The intended test is "all points concordant".
- Proposed fix (R): compare counts, e.g. `if (d0_cupm + d0_clpm == n || == 0)`. Python needs no
  change. Note that DEP-06 makes the +1 for constant cells questionable anyway.

### DEP-09: Copula on cells with (near-)constant values is decided by floating-point noise, so NNS.caus results on tied or discrete data differ wildly between R and Python
- Type: parity-mismatch (both are ill-conditioned)
- Severity: medium
- Confidence: verified
- Where: R `NNS_dep.cpp:55-57` (the mean is a sequential double sum) against Python's
  `np_mean` (pairwise sum, `_nnscore_bindings.cpp:662`). The comparisons `dx > 0`, `dx <= 0`
  and `cov` sign then flip on residuals of 1e-16.
- Evidence: `compare.py` output and `patched_check.R`.
  - For the same `step` data, NNS.caus(tau=0) gives C(x→y) = **4.02 in R and 34.17 in Python**.
  - For `discrete_indep` it gives R 0.246, Python 1.32, and an R-level re-implementation with
    long-double sums 0.0002.
  - `ties_heavy` and `mono_rank` also differ. Every dep/copula/caus value matches to 1e-15 on
    continuous data.
  - Mechanism: a constant cell of 0.1·k values has a sequential mean minus 0.1 of ±1e-15
    (demonstrated in the session), so every point becomes strictly above or strictly below the mean.
- Which side is right: neither. The result should not depend on summation order.
- Proposed fix: snap deviations with `abs(dx) <= 8*eps*max(abs(x))` to 0 and handle zero-variance
  cells explicitly (DEP-06). Results then agree across implementations.

### DEP-10: Python's internal gravity (`gravity_exact`) caps bins at 1e6, so nns_part and nns_dep diverge from R on data with outliers
- Type: parity-mismatch (Python wrong)
- Severity: medium
- Confidence: verified
- Where: `_nnscore_bindings.cpp:323` (`std::min(..., 1000000L)`) and `dependence.py:389`
  (`_MAX_DENSE_BINS`). These are used by `nns_part` (native `nns_part_off`), `nns_dep` and
  `causation`. The public `nns_gravity` uses a different native routine without the cap and
  agrees with R.
- Evidence: `work/DEP/outlier.csv` (n=1e4 normals with x[0] = -1e5).
  - `nns_gravity(x)` gives -2.510 and R `NNS.gravity` gives -2.510, but internal `_gravity(x)`
    gives **-21558.3**.
  - The `nns_part(x, y, order=1)` regression point is x = **-21558**, y = -20217, against R's
    x = -2.51, y = -2.54.
  - `nns_dep` gives 0.3633 against R's 0.3592.
  - The docs (conventions.md "Dependence") claim the cap is `4*len(input)`, but the code uses 1e6.
- Which side is right: R (exact bins) up to R's own int overflow (DEP-11). Clipping every bulk
  value into the last bin puts the "mode" in the wrong place.
- Proposed fix: replace dense binning with a sorted-run histogram. Only occupied bins matter,
  found with `np.unique(np.floor((v-origin)/width), return_counts=True)` and neighbour lookup on
  the bin index. That is exact, O(n log n), and needs no cap. Results then match R.

### DEP-11: Gravity (the partition centre) breaks with a single huge outlier (R and Python)
- Type: r-bug (shared helper; affects NNS.part/NNS.dep partitions)
- Severity: medium
- Confidence: verified
- Where: R `src/central_tendencies.cpp:92-93` (`int nbins = (int)floor(...)` overflows, then
  `nbins < 1` collapses to 1 bin). Python mirrors this through its `int_max` guard
  (`dependence.py:383`).
- Evidence: with n=1e4 normals and x[1] = -1e9, R `NNS.gravity` gives **-250,025,000** (median
  -0.007, mean -1e5). Python `_gravity` and `nns_gravity` both give -250025000.0015.
  With x[1] = -1e7 instead, R would allocate about 7.4e8 ints (3 GB) inside
  `simple_bin_counts` (not run).
- Which side is right: neither. A "robust" central tendency should not land on the outlier.
- Proposed fix: the same sparse-histogram approach as DEP-10, or fall back to `q2` when the bin
  count exceeds n. This changes results only on extreme data.

### DEP-12: "Correlation" = max(corr_xy, corr_yx) of signed values: positive bias under the null, and the weaker magnitude is reported for negative relationships
- Type: r-bug (shared)
- Severity: medium
- Confidence: verified
- Where: R `src/NNS_dep.cpp:240`, Python `_nnscore_bindings.cpp` (nns_dep_pair) and
  `dependence.py:72`.
- Evidence: from `run_py.py`/`run_r.R`, R and Python are equal.
  - `neg_linear_noise` has corr_xy = -0.210 and corr_yx = -0.267; the reported value is -0.210.
  - For independent normals at n=100 (200 reps), the mean signed Correlation is **+0.091** and 71.5%
    of values are positive.
  - For `y_x2` (symmetric parabola, Pearson -0.01) the Correlation is +0.259; at n=100,
    `NNS.dep(x, x^2)` gives Correlation 0.82.
- Which side is right: a signed coefficient should be symmetric and unbiased in sign. Taking the
  max is reasonable for the unsigned Dependence, but not for a signed value.
- Proposed fix: return the value of larger magnitude (`abs(a) >= abs(b) ? a : b`) or the mean of
  the two directions. This changes the Correlation output.

### DEP-13: NNS.dep(p.value=TRUE) has about double the nominal size, the wrong tail for Dependence, and plotting side effects
- Type: r-bug (Python lacks the option; see DEP-15)
- Severity: medium
- Confidence: verified
- Where: R `R/Dependence.R:85-86,116-121`.
- Evidence: `work/DEP/pval_r.R`, 200 null reps at n=100.
  - P(p < 0.05) is **0.095** for both Correlation and Dependence, and P(p = 0) = 0.025.
  - The p-value is `min(LPM(0, obs, null), UPM(0, obs, null))`, a one-sided minimum without the
    factor 2 and without a +1 correction. For Dependence, which is a non-negative magnitude, only
    the upper tail is meaningful, but an observed value *below* the null also yields a small p.
  - With `print.map=FALSE`, `par(no.readonly=TRUE)` still opens a graphics device. In Rscript this
    writes `Rplots.pdf` and emits "calling par(new=TRUE) with no plot" warnings.
  - `asym` is ignored on this path, and there is no seed argument.
- Proposed fix: Dependence `p = (1 + sum(null >= obs)) / (1 + B)`; Correlation
  `p = (1 + sum(abs(null) >= abs(obs))) / (1 + B)`. Only touch `par()` when `print.map` is TRUE.
  Add `seed` and `nperm` arguments. This changes the p-values.

### DEP-14: NNS.copula does not vanish under independence for skewed marginals, and gives 0.866 when one column is constant
- Type: r-bug (shared)
- Severity: medium
- Confidence: verified
- Where: R `R/Copula.R:96-117`, Python `copula.py:85-100` and native `copula_nd`.
- Evidence: `indep_bias.py`.
  - For independent Exp(1) pairs, NNS.copula is 0.233 at n=1e3, 0.203 at 1e4 and **0.193 at
    1e5**, so it does not decay. The analytic limit is sqrt(2·0.0699/4) = 0.187.
  - The cause: the degree-0 null of 0.5 assumes P(X <= mean) = 0.5, but for Exp(1) this is 0.632,
    so Co_0 = 0.535. The degree-1 terms are correctly null, because E(x-mu)+ = E(mu-x)+.
  - `NNS.copula(cbind(x, 1))` = **0.866** in both R and Python.
  - The early return `discrete_Co_pm == 1` (`Copula.R:80`) compares a sum over d(d-1)/2 pairs to
    1. That only makes sense for d = 2 and can fire spuriously when d >= 3 (not reproduced).
- Proposed fix: use the empirical marginal probabilities as the null
  (`p_j = LPM(0, t_j, X_j)`, Co_null = sum over pairs of `p_i p_j + (1-p_i)(1-p_j)`,
  DPM_null = 1 - prod p - prod(1-p)). Return 0/NA for a zero-variance column. Compare against
  `n_pairs` instead of 1. This changes results for skewed data.

### DEP-15: Python API gaps: no matrix NNS.dep, no p-values, and causal_matrix drops labels
- Type: api
- Severity: medium
- Confidence: verified
- Where: `dependence.py:14` (`nns_dep(x, y, asym)` requires y and 1-D input),
  `causation.py:26,66`.
- Evidence (`api_py.py`):
  - `nns_dep(M)` raises `TypeError: missing 1 required positional argument: 'y'`.
  - `nns_dep(M, None)` raises "x and y must be 1D".
  - `nns_dep(x, y, p_value=True)` raises a TypeError.
  - R's documented `NNS.dep(B)` returns Correlation and Dependence matrices, including `asym=TRUE`.
  - `nns_causation` has no `p_value/nperm/permute/seed/conf_int/factor_2_dummy`. `causal_matrix`
    has no p-value/CI and returns a bare ndarray, even for a labelled DataFrame (R keeps dimnames).
  - The docstrings never say which direction `asym=True` measures. It is y as a function of x:
    `nns_dep(x, x**2, asym=True)` = 0.989 and the reverse is 0.342.
  - A user-side pairwise loop does reproduce R's symmetric and asymmetric matrices on 5-asset
    returns (SPY QQQ TLT GLD BTC, max abs diff 6e-16; `fin_r.R` and the inline check).
- Proposed fix: accept 2-D/DataFrame `x` with `y=None` and return `{"Correlation": M, "Dependence": M}`
  with R's orientation (row→column; see DEP-20). Add a permutation p-value with the corrected
  formula (DEP-13/DEP-04). Return a labelled DataFrame when the input has labels. These are
  additions only.

### DEP-16: nns_part with noise_reduction other than "off" is 8–20x slower than R
- Type: perf
- Severity: medium
- Confidence: verified
- Where: `src/nns/part.py:182-224` (pure-Python loop: `np.unique` on string arrays at every depth,
  a boolean mask per group so cost is O(n·groups), per-group `_aggregate_*`, and
  `_regression_points`, which is also mask-per-group).
- Evidence: `work/DEP/time_py.py` and `time_r.R` (1 thread, best of runs):

  | n | mean (Py / R) | median (Py / R) | mode (Py / R) | off (Py / R) |
  |---|---|---|---|---|
  | 1e4 | 0.046 / 0.006 | 0.091 / 0.008 | 0.062 / 0.010 | 0.017 / 0.010 |
  | 1e5 | **1.13 / 0.057** | **2.65 / 0.18** | **1.50 / 0.125** | 0.32 / 0.22 |

- Proposed fix: extend the native `nns_part_off` with the mean/median/mode aggregators. In Python,
  sort once by an integer group id and use `np.split` or `ufunc.reduceat` instead of per-group
  masks, and keep integer path codes instead of growing strings. Results do not change.

### DEP-17: nns_part allocates string arrays as wide as the requested `order`: memory and time scale with order even though depth is capped at log2(n)
- Type: python-bug / perf
- Severity: medium
- Confidence: verified
- Where: `src/nns/part.py:178-179` (`dtype=f"<U{max_order + 1}"`) on the non-"off" paths.
- Evidence: `work/DEP/part_order_mem.py`, n=1000, `noise_reduction="mean"`, actual depth 3 in every
  run:
  - order=None takes 0.08 s and 1.6 MB.
  - order=2000 takes 0.20 s and 33 MB.
  - order=20000 takes **1.97 s and 328 MB**.
  - Extrapolating, n=1e5 with order=1000 would need about 3.2 GB.
- Proposed fix: `max_order = min(max_order, floor(log2(n)))` before allocating, and use integer or
  object labels. Results do not change.

### DEP-18: nns_causation / causal_matrix are about 2x slower than R because the hot path is pure Python, although a native kernel exists
- Type: perf
- Severity: low
- Confidence: verified
- Where: `causation.py:131-145` (`_asym_dep` uses the pure-Python `_xonly_partition`, which has
  per-element list appends and string concatenation, plus the Python `_copula_signed`). The native
  `nns_dep_pair(x, y, asym=True)` gives the same value (checked equal on all 41 cases).
- Evidence: `time_py.py` / `time_r.R`:
  - `causal_matrix` on 10 columns: Python 2.33 s vs R 1.59 s at n=5000, and 0.77 s vs 0.39 s at
    n=1000.
  - `nns_causation` at n=1000: 0.022 s vs 0.011 s. At 1e5 it is 0.89 s vs 0.91 s.
  - The profile at n=2e4 shows 61% of the time in `_xonly_partition`.
- Proposed fix: `_asym_dep = lambda a, b: nns_dep(a, b, asym=True)["Dependence"]` (the native
  path). Results do not change.

### DEP-19: NNS.dep is not invariant under monotone transforms, despite the copula framing
- Type: r-bug (shared design) / docs
- Severity: low
- Confidence: verified
- Evidence: cases `mono_*` with the same (x, y = x + e) pair, n=1000. Correlation/Dependence:
  - base: 0.385 / 0.422
  - exp(x): 0.369 / 0.453
  - x^3: 0.297 / 0.427
  - exp(y): 0.313 / 0.410
  - ranks: 0.317 / 0.401

  NNS.copula ranges from 0.756 to 0.841 across the same transforms. The book (ch.12) presents
  these measures as copula-space and rank-related.
- Proposed fix: document that the measures depend on the marginals (mean targets, degree-1
  terms), or offer a rank-transformed option.

### DEP-20: R docs: NNS.dep asymmetric matrix orientation is documented backwards, and the `asym` description is confusing
- Type: docs (R)
- Severity: low
- Confidence: verified
- Where: `man/NNS.dep.Rd:14,27` (`R/Dependence.R:7,13`).
- Evidence: for (x, y = x^2), `NNS.dep(cbind(x,y), asym=TRUE)$Dependence[1,2]` = 0.989 =
  `NNS.dep(x, y, asym=TRUE)`, which is dependence of y on x. So the entry at [row, col] means
  row → column. The note says "([column variable] ---> [row variable])".
  `asym` is described as "logical; FALSE (default) Allows for asymmetrical dependencies".
- Proposed fix: correct the note to row → column. Describe `asym=TRUE` as: the partition is on x
  and the result measures y as a function of x.

### DEP-21: Book ch.14 describes a different NNS.caus from the one implemented
- Type: docs (R book vs code)
- Severity: medium
- Confidence: verified (code reading plus the DEP-02/03 runs)
- Where: `book/chapter-14-directional-causation.Rmd:120-140,150-170,232-266` against
  `Uni_Causation.R:55`, `Causation.R:207-228`, `Normalization.R`.
- Evidence:
  1. The book says NNS.norm "implements the empirical CDF transformation". The code multiplies by
     a scalar (DEP-03).
  2. The book's raw statistic is (1/2)[P + max(Δρ, 0)]. The code computes
     `mean(c(P * rho.x.y, max(0, Δρ)))`, so P is multiplied by ρ.
  3. The book's final index is the normalised C_{X→Y} = C~xy/(C~xy + C~yx) in [0,1], with 0.5 as
     neutral. The code returns a signed log-ratio capped at ±100 under a direction-dependent name.
  4. Lagged co-partial moments (X_{t-τ}, Y_t) are never computed (DEP-02).
- Proposed fix: align the code with the book (preferred: an ECDF transform, real lags, and also
  report the normalised index) or rewrite the chapter.

### DEP-22: Python docs are stale or wrong about these functions
- Type: docs (Python)
- Severity: low
- Confidence: verified
- Where:
  - `docs/conventions.md` "Partition" says R only distinguishes NULL from non-NULL `type`, and that
    `order="max"` is rejected with `TypeError`. The code (`part.py:137,146-149`) and R 13.3 now
    validate `type` (ValueError for "X") and support `order="max"`.
  - `conventions.md` "Copula" and `api_reference.md:54,210` say "bivariate scalar form/surface",
    but multivariate input is implemented.
  - `api_reference.md:56,220` names the R API `Causal.matrix`, which does not exist (it is
    `NNS.caus(matrix)` / internal `NNS.caus.matrix`).
  - `conventions.md` "Dependence" says the gravity bin cap is `4*len(input)`; the code uses 1e6
    (DEP-10).
  - The `nns_dep` docstring does not explain `asym`.
- Proposed fix: update the docs.

### DEP-23: Dead or wasted code in R causation and dependence
- Type: dead-code (R)
- Severity: low
- Confidence: verified (code reading)
- Where:
  - `Uni_Causation.R:6`: `xy <- NNS.norm(cbind(x, y), ...)` is never used. It costs one
    `cor` / NNS.norm per call.
  - `Causation.R:168-174,182-188` recomputes `Uni.caus` with identical, deterministic inputs
    whenever the two values are equal or zero, which doubles the cost on those inputs.
  - `Uni_Causation.R:10-17`: `x.vectors <- list(tau+1)` plus named `numeric(0)` placeholders. This
    is harmless because `cbind` drops zero-length vectors, but it is junk.
  - `NNS_dep.cpp:270-427` (`NNS_dep_matrix_cpp`) is exported but never called, and uses a
    different, mean-centred partition, so it would give different numbers if someone wired it in.
  - `Dependence.R:175,186`: the `order`/`degree` arguments and `if(nrow(x) < 20) order <- 2` are
    unused.
- Proposed fix: delete these. Results do not change.

### DEP-24: tau and degenerate-input handling in NNS.caus / nns_causation
- Type: api / r-bug
- Severity: low
- Confidence: verified (`api_py.py`, `api_r.R`, ts check in the session log)
- Evidence:
  - `tau="ts"` on an AR(1) series: R fails with the cryptic "missing value where TRUE/FALSE
    needed", because no NNS.seas period is <= sqrt(n). Python raises a clear ValueError. Both are
    unusable on ordinary autoregressive series.
  - Python silently truncates `tau=1.5` to 1 and accepts `tau="3"`. R gives `tau=1.5` a
    fractional-index meaning, and `tau="foo"` fails with "object 'Causation.y.given.x' not found".
  - R `tau=-1` is silently treated as 0; Python raises.
  - `tau >= n`: Python says "x must be non-empty", R says "only 0's may be mixed with negative
    subscripts".
  - A constant y: R `NNS.caus` errors with "x has missing values, please address" (misleading),
    while Python returns 0/0/0 silently.
- Proposed fix: validate `tau` as a non-negative integer smaller than n-2 in both. Fall back to
  tau=1 (or raise a clear error) when no seasonal period qualifies. Reject constant inputs
  explicitly.

### DEP-25: nns_part argument validation differs from R and rejects numpy integers
- Type: api
- Severity: low
- Confidence: verified (`api_py.py`, `api_r.R`)
- Evidence:
  - `nns_part(x, y, order=np.int64(3))` raises `TypeError: order must be an integer`, as does
    `order=3.0`. R accepts 3.0.
  - `order=0` is accepted as 1; R errors.
  - `obs_req=2.5` is accepted; R errors with "[obs.req] must be a nonnegative integer".
- Proposed fix: accept `numbers.Integral` and integral floats, and reject 0 and non-integral
  `obs_req`, as R does.

### DEP-26: R NNS.dep silently accepts Inf
- Type: parity-mismatch (Python is better)
- Severity: low
- Confidence: verified
- Evidence: `NNS.dep(c(x[-1], Inf), y)` returns 0.551 without warning. Python raises "must contain
  only finite values". `NNS.copula` in R does reject Inf.
- Proposed fix (R): add `if (any(!is.finite(x))) stop(...)`.

### DEP-27: NNS.part man page does not define the quadrant labels or the regression-point semantics
- Type: docs (R; Python docstring too)
- Severity: low
- Confidence: verified (`NNS_part.cpp:153-158,184-200`; R run in the session log)
- Evidence:
  - Labels: in XY mode the digit is `1 + (x<=cx) + 2*(y<=cy)`, so 1 = upper-right,
    2 = upper-left, 3 = lower-right, 4 = lower-left, relative to gravity centres (not means).
    In XONLY mode 1 = low and 2 = high.
  - Regression points are aggregated by **prior.quadrant**, so `order=1` gives 4 final quadrants
    but a single regression point ("q"), and `order=2` gives 16 quadrants and 4 points.
  - The returned `order` can be smaller than requested (order=3 returns 2 at n=200 because of
    `min.obs.stop`).
  - None of this is in `NNS.part.Rd` or the `nns_part` docstring.
- Proposed fix: document it.

---

## Coverage (examined and found consistent between R and Python unless noted above)
- `nns_dep` / `NNS.dep` bivariate, sym and asym, both argument orders, on 41 cases. These covered
  independent noise at n = 10..1e4, y=x, y=-x, x^2, -x^3, sin, circle (exact and noisy), step and
  floor functions, linear ± noise, monotone transforms, ranks, exponential and t2 tails, discrete
  data, binary data, heavy ties, length 2/3/5, constant y, integer sequences, and SPY/QQQ/TLT/GLD
  returns. Everything matches to 1e-15 except discrete/tied cases (DEP-08/09) and outliers (DEP-10).
- Symmetry: `dep(x,y) == dep(y,x)` holds exactly for asym=FALSE on both sides.
- Matrix: R `NNS.dep(M)` and `NNS.dep(M, asym=TRUE)` on 5 assets (SPY QQQ TLT GLD BTC, n=3048) equal
  a Python pairwise loop to 6e-16. `causal_matrix` against `NNS.caus(M)` matches to 2e-15 for tau=0
  and tau=3 (last 1000 days).
- The pure-Python fallbacks (`_dep_pair`, `_asym_dep`) equal the native `nns_dep_pair` on all 41
  cases.
- `nns_copula` / `NNS.copula`: bivariate (all cases), 3/4/8/11-column matrices, `continuous=FALSE`,
  zero targets, discrete, identical and anti columns. All match to 1e-12.
- `nns_causation` / `NNS.caus` with tau = 0, 3 and "ts" on continuous data match exactly.
  `tau="ts"` matches on indep, SPY/TLT and sin series. Python's antisymmetric `causal_matrix` sign
  convention matches R.
- `nns_part` / `NNS.part`: 2640 combinations (11 datasets × type None/XONLY × order
  None/1/3/"max" × obs.req 0/8/50 × min.obs.stop T/F × 5 noise reductions). order, quadrant,
  prior.quadrant, regression-point labels and x/y all match (`part_py.py`, `part_r.R`,
  `part_cmp.py`), including discrete x rounding, NaN dropping and `order="max"`.
- Input handling in Python: pandas Series, int dtype, lists and NaN rejection in nns_dep and
  nns_causation work. Constant input returns 0/0, as in R.
- Timing at n=1e3/1e4/1e5 (one thread): `nns_dep` 0.0014/0.0145/0.16 s against R
  0.001/0.015/0.21 s. `nns_copula` is faster than or equal to R. `nns_part` "off" 0.32 s against
  0.22 s at 1e5. A 10-column dependence matrix (n=5000) takes 0.34 s in Python against 0.39 s in R.
