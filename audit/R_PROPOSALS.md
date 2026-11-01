# NNS (R) proposals: design and methodology questions

These items were left unchanged in the `fix/r-bugs` patch series on purpose. Each one changes what a
statistic means, not just a wrong line of code, so the maintainer should decide. Every section gives
the problem, the evidence (finding ID and repro under `/home/user/audit/work/<GROUP>/`), the options,
a recommended option, and what results would change.

Status of the patch series: every number below was measured on R 13.3 (`82a92eb`) unless it says
"patched", which means the `fix/r-bugs` build installed in `~/Rlib-rfix`.

---

## DEP-02: `NNS.caus(tau = k)` does not lag anything

**Problem.** `Uni.caus` builds the lag matrix `x[(tau-i+1):(n-i)]`, `i = 0..tau`, then keeps only
column 1 of `NNS.norm(lag matrix)`. `NNS.norm` multiplies each column by a scalar, so `x.norm.tau` is
just `c * x[(tau+1):n]`. x and y are never offset against each other: tau only drops the first tau rows
and rescales.

**Evidence.** `work/DEP/caus_sim.py` and `caus_r_check.R`: `cor(x.norm.tau, x[t]) = 1`. Design
`y_t = 0.8 x_{t-1} + 0.3 e_t` (30 reps, n = 500): share of reps reporting x→y is 0.33 at tau = 0, 0.43 at
the true lag tau = 1, and 0.33 at tau = 3. Independent noise gives 0.40, 0.53 and 0.43, so the lagged
design cannot be told apart from noise. Book ch. 14 defines CoUPM^(tau) on the pairs (X_{t-tau}, Y_t).

**Options.**
1. Build the lagged pairs explicitly: for x→y use `x_lag = x[1:(n-tau)]` and `y_lead = y[(tau+1):n]`,
   and the reverse for y→x.
2. Option 1, plus pre-whitening each series on its own lags (for example the residuals of `NNS.reg` on
   its own lag matrix) before the cross-normalisation, which is the "removing internal dynamics" step
   in the book.
3. Keep the code and document that tau only trims and rescales.

**Recommendation.** Option 1 now, with option 2 behind an argument (`prewhiten = FALSE`).
**Expected change.** Every result with tau > 0 or tau = "ts" changes. tau = 0 is unchanged.

## DEP-03: `NNS.caus` depends on location and scale (NNS.norm)

**Problem.** `NNS.norm` scales column j by `colMeans(m %o% (1/m) * |cor|)[j]`, where m is the vector
of column means. That scale can be negative or near zero, and it changes when a constant is added to
a variable, so the causal direction depends on the origin and units.

**Evidence.** `work/DEP/norm_sign.R`:
- x ~ N(1,1) and y = x - 1.5 + noise: after normalisation cor(x, y) is -0.957 because y's scale is -0.465.
- `NNS.caus(x, y)` gives C(x→y) = 0.0095, but `NNS.caus(x, y + 10)` gives 1.016.
- `NNS.caus(x, 2x + 1)` gives 1.49, while `NNS.caus(x, x)` gives 0.
- SPY/TLT returns: C(TLT→SPY) = 0.250. Adding 1.0 to TLT flips the result to C(SPY→TLT) = 1.877.

**Options.**
1. Use a rank/ECDF transform per column (`LPM.ratio(0, x, x)`), which is what book ch. 14 describes as
   NNS.norm.
2. Standardise each column (z-scores), then rescale to a common positive range.
3. Minimum: use `abs(m)`, and stop or warn when a mean is near 0 relative to its sd.

**Recommendation.** Option 1, invariant to any monotone transform of either variable.
**Expected change.** All `NNS.caus` / causation-matrix outputs change, and so do `NNS.reg` and
`NNS.stack` with `dim.red.method = "NNS.caus"` or `"all"`. Results stop depending on shifts of a
variable.

## DEP-05: `NNS.dep` independence floor

**Problem.** Under independence the per-cell `abs(copula)` terms are positive and averaged, so the
Dependence floor decays only like n^(-1/4). Moderate linear dependence barely clears it.

**Evidence.** `work/DEP/indep_bias.py`, mean Dependence of independent normals:

| n | 30 | 100 | 300 | 1e3 | 1e4 | 1e5 |
|---|---|---|---|---|---|---|
| Dependence | 0.649 | 0.600 | 0.447 | 0.323 | 0.180 | 0.105 |

At n = 1000, y = x + N(0,1) (Pearson 0.72) scores 0.443 against a null of 0.32. BTC–GLD (Pearson
0.10) scores 0.324, above SPY–TLT (Pearson -0.17) at 0.286.

**Options.**
1. Report a null-adjusted value `(D - E0[D]) / (1 - E0[D])`, with E0[D] from a cached permutation
   table by n (or computed on the fly with a few permutations).
2. Debias each cell: subtract the expected `abs(copula)` of a cell of that size under independence
   before averaging.
3. Use larger cells (obs.req about n/4 instead of n/8), which lowers the variance and the floor.
4. Documentation only: state the n-dependent floor and recommend `p.value = TRUE`. The patch series
   already makes the Dependence p-value an upper-tail permutation p-value.

**Recommendation.** Option 4 now (documentation), then option 2, which keeps the 0..1 scale and needs
no randomness.
**Expected change.** Options 1 and 2 lower every Dependence value, by a lot on small samples, and change
anything that consumes it: `NNS.reg` default order, `NNS.stack` / `NNS.reg` with
`dim.red.method = "NNS.dep"`, and `NNS.VAR`.

## DEP-12: signed "Correlation" = max(corr_xy, corr_yx)

**Problem.** `NNS_dep_pair_core` returns `max(corr_xy, corr_yx)` of the *signed* directional
correlations. That biases the result upward under the null and, for negative relationships, reports
the weaker magnitude.

**Evidence.** For independent normals at n = 100 (200 reps), the mean Correlation is +0.091 and 71.5% of
values are positive. In `neg_linear_noise`, corr_xy = -0.210 and corr_yx = -0.267, and -0.210 is
reported. For `y = x^2` the result is +0.82.

**Options.**
1. Return the value of larger magnitude: `abs(a) >= abs(b) ? a : b`.
2. Return the mean `(a + b) / 2`, which is symmetric and unbiased in sign.
3. Return the signed value from the direction with the larger Dependence.

**Recommendation.** Option 2 for the symmetric measure. The max stays appropriate for the unsigned
Dependence.
**Expected change.** The Correlation output of `NNS.dep` (and the matrix form) changes whenever the two
directions differ. Dependence is unchanged.

## DEP-13 (follow-up): `NNS.dep(p.value = TRUE)` Correlation p-value

The patch series fixes the Dependence tail (upper tail with a +1 correction), the plotting side effect,
and the dropped `asym` argument. The Correlation p-value is still
`min(LPM(0, obs, null), UPM(0, obs, null))`. That is the smaller of two one-sided tail frequencies, with
no factor 2 and no +1, so its size is about twice nominal (`work/DEP/pval_r.R`: P(p < 0.05) = 0.095).

**Recommendation.** Use the two-sided `(1 + sum(abs(null) >= abs(obs))) / (1 + B)`, and add `nperm` and
`seed` arguments.
**Expected change.** Correlation p-values roughly double for the same data.

## DEP-14: `NNS.copula` does not vanish under independence for skewed marginals

**Problem.** The degree-0 null of 0.5 assumes P(X <= mean) = 0.5. For Exp(1) that probability is
0.632, so the null co-partial-moment frequency is 0.535 and the copula has a non-vanishing floor. A
constant column gives 0.866. The early return `discrete_Co_pm == 1` compares a sum over d(d-1)/2 pairs
with 1 (CORE-10).

**Evidence.** `indep_bias.py`: independent Exp(1) pairs give 0.233 at n = 1e3 and 0.193 at n = 1e5,
against an analytic limit of 0.187. `NNS.copula(cbind(x, 1))` gives 0.866.

**Options.**
1. Use empirical marginal probabilities for the null: with `p_j = LPM(0, t_j, X_j)`,
   `Co_null = sum over pairs of p_i p_j + (1 - p_i)(1 - p_j)` and
   `DPM_null = 1 - prod(p) - prod(1 - p)`.
2. Rank-transform the columns first, so that p_j = 0.5 holds exactly at the median target.
3. Return 0 (or NA) for zero-variance columns, and compare the early return against `n_pairs`.

**Recommendation.** Options 1 and 3.
**Expected change.** The copula drops on skewed data. This flows into `NNS.reg` default order /
`n.best` and smoothing `spar`, which all use `NNS.copula`.

## TS-04: `NNS.seas` has no null calibration

**Problem.** A period counts as "seasonal" when the CV of its reverse-stride subsample is <= the
full-series CV. There is no minimum subsample size and no multiple-testing control, and the smallest CV
wins. Long periods built on 2-3 points win by chance, and on trending data the CV mostly measures the
trend.

**Evidence.** `work/TS/seas_noise.py`: iid N(10,1), n = 1000, flags about 112 of 500 periods, and the
best period is always in 216-498. Return-like noise flags 366 of 500. AirPassengers has best = 50 and 12
ranks 15th. A noise-free period-12 sine gives best = 48. Default `NNS.ARMA` on AirPassengers
(rolling MAE 68.3) loses to the seasonal naive forecast (35.9), while `seasonal.factor = 12` gets 18.9.

**Options.**
1. Require at least 4-5 points per subsample (period <= n/5) before a period can qualify.
2. Compare each CV ratio with a permutation or block-bootstrap null and keep periods whose ratio is
   below the (1 - alpha / #periods) null quantile.
3. Rank by an information-weighted score (for example CV ratio times sqrt(points per subsample)) instead
   of the raw minimum.
4. Documentation: state that auto mode is unreliable and recommend `NNS.ARMA.optim` with an explicit
   candidate set.

**Recommendation.** Option 1 plus option 4 now (cheap, deterministic); option 2 as an opt-in
(`seas.test = "permutation"`).
**Expected change.** `NNS.seas` periods and best.period change. So do default `NNS.ARMA`
(seasonal.factor = TRUE / dynamic) forecasts and the `NNS.VAR` univariate stage, which also gets much
faster on return data (TS-21).

## SDA-10: ANOVA `Certainty` calibration

**Problem.** `Certainty` is presented as a calibrated probability and the inverse of a p-value. It is
neither: under H0 it is far from uniform and depends on n, and small samples look *more* different
through `pop.adjustment`.

**Evidence.** `work/SDA/anova_calib.py`: same-distribution samples give a median Certainty of 0.73 at
n = 30 and 0.96 at n = 1000 (means.only), and P(C < 0.10) = 0. With a 0.3 sd shift at n = 100 the
t-test rejects in 55% of runs, but Certainty is below 0.05 in 0%. SPY vs TLT has KS p = 1.8e-9 and
Certainty 0.914. Identical samples give 0.96 (n = 50), 0.81 (n = 10) and 0.44 (n = 3).

**Done in the patch series.** The docs (roxygen and Rd) now call it a descriptive similarity index and
warn against p-value thresholds. No numbers changed.

**Options.**
1. Add `calibrate = FALSE`. When TRUE, also return `Certainty.p = (1 + #{perm C <= C_obs}) / (1 + B)`
   from B label permutations of the pooled sample. That is a valid permutation p-value for "same
   distribution".
2. Remove `pop.adjustment`, which is the source of the inverted small-sample behaviour, or replace it
   with a variance-based shrinkage toward 0.5.
3. Keep the documentation change only.

**Recommendation.** Option 1, which is additive and keeps the current Certainty unchanged. Then decide
option 2 separately.
**Expected change.** None with option 1 at its default. Option 2 changes every Certainty value for
small samples.

---

## Additional items found while patching (not in the original list)

### NNS.stack Method 1, univariate design: the n.best candidates are not what is deployed
`NNS.reg` with a single regressor ignores `n.best`: `NNS.reg(x, y, n.best = 1 / 5 / "all")` gives
identical `Point.est`. The univariate Method 1 search still scores k-nearest-neighbour means for
k = 1..sqrt(n) out of fold. The final `reg` is then always the plain univariate `NNS.reg` fit, whatever
k is selected, so `OBJfn.reg` and `NNS.reg.n.best` describe a model that is not returned. The patch
series fixes only the "ALL" candidate (STK-03), which now scores the deployed `NNS.reg`.

**Options.**
1. Score the univariate `NNS.reg` fit once, without a k search, and report n.best = NA.
2. Make univariate prediction honour `n.best` as a kNN ensemble over regression points.

**Recommendation.** Option 1, which keeps the deployed model.
**Expected change.** Univariate stack `OBJfn.reg`, `NNS.reg.n.best` and weights change. `reg` is
unchanged.

### dy.d_ (follow-up to REG-04): bandwidth attenuation
After REG-04, `dy.d_` differentiates the multivariate `NNS.reg` fit, so irrelevant regressors get about
0 (y = 2 x1: dy/dx2 is -0.02 to 0.05) and signs are right. Magnitudes are still attenuated
(dy/dx1 = 1.25-1.47, truth 2). The "v0.5.7" bandwidth set `h_s = 1/log(length(x), c(2,10))` and
`10 * h_s` gives probability windows up to ±1, so the outer bands span the whole support, and the
2-D kNN fit smooths across x1. With only the two small bands the same case gives 1.56 / 2.00 / 1.64.

**Recommendation.** Drop the 10x bands (or choose h like `dy.dx` does) and evaluate f(x ± h e_j)
holding the other coordinates at the point. Use `order = "max"` / `n.best = 1` only for interpolation
checks.
**Expected change.** All `dy.d_` values change, toward the truth on the analytic tests.

### NNS.ARMA.optim reports the weights = NULL objective when equal weights are chosen
After TS-08, `$weights` can now be `c(1/k, ...)`, but `$obj.fn` still reports `nns.SSE`, the search
objective with `weights = NULL`. **Recommendation:** report `weight.SSE` when equal weights are
selected. **Expected change:** the `obj.fn` field only.

### Python note (for the NNS-python fixers)
`nns_arma(dynamic=True, seasonal_factor=False, best_periods=2, modulo=12)` on AirPassengers returns
`441.71 0 0 0`. It is the same zero-weight problem that R now handles with equal weights (TS-11), but
Python reaches it through its own dynamic path. Once R is patched, R gives `441.58 406.03 455.48 476.33`.

---

## Second batch

### PM-08: `NNS.mode(multi = TRUE)` reports spurious modes that multiply with n
**Problem.** The peak finder (tri-7 smoothing plus a local maximum in ±3 bins) has no prominence or
count threshold, so every isolated tail bin becomes a "mode". The documentation gives no threshold that
the code was meant to apply ("multi: TRUE (default) returns multiple mode values"), so I did not change
the behaviour.
**Evidence.** On the patched build, N(0,1) samples (mean of 10 seeds) give 2.1 modes at n = 200, 8.5 at
n = 1000 and 23.1 at n = 5000. `work/PM/t18_mode_spurious.py` reports 50 modes on t3 "returns" at
n = 4000 and 42 on NVDA returns.
**Options.**
1. Keep a secondary peak only if its smoothed height is at least a fraction q of the global peak
   (for example q = 0.5).
2. Keep it only if its prominence (height minus the higher of the two surrounding minima) exceeds a
   multiple of the Poisson noise of the bin count, `sqrt(count)`. That rule is scale-free and grows
   correctly with n.
3. Change the default to `multi = FALSE`.

**Recommendation.** Option 2 with a factor of 2 (about a 2-sigma bump), documented in the Rd, plus a
`min.prominence` argument.
**Expected change.** The output of `NNS.mode(multi = TRUE)` shrinks to the genuine modes: 1 for normal
data, 2 for clearly bimodal data. `multi = FALSE`, `NNS.gravity` and the mreg mode reducer
(`multi = FALSE`) are unaffected.

### TS-25: `NNS.meboot(drift = FALSE)` rescales the replicates to the full sd(x), trend included
**Problem.** With `drift = FALSE` the trend is removed (the baseline is flat at the intercept). The
final `expand.sd` step still matches the replicate sd to `sd(x)` of the original, trending series, so
all of the trend variance is pushed into the residual amplitude.
**Evidence.** AirPassengers, patched build (reps = 50, rho = 0):

| | replicate sd | replicate slope | min |
|---|---|---|---|
| `drift = TRUE` | 137 | 3.12 | 19 |
| `drift = FALSE` | 122.5 | 1.21 | 71 |

For comparison, sd(x) is 120, the residual sd about the linear trend is 45.9, and the original slope
is 2.66. `work/TS/meb_cmp.txt` shows replicates down to -124 without `xmin`.
**Why proposal only.** The Rd says only that `drift = TRUE` "preserves the drift" and refers
`expand.sd` to `meboot::meboot`, which expands to the sd of the series it is given. Neither says which
sd a drift-free replicate should have, so the intended behaviour is not clear.
**Options.**
1. When the reconstruction slope differs from the original, expand against
   `sd(baseline + orig_res)` (the series the replicate is meant to imitate) instead of `sd(x)`.
2. Skip the level-scale `expand.sd` when `drift = FALSE` and keep only the residual-scale expansion,
   which already matches `sd(orig_res)`.
3. Document that `drift = FALSE` keeps the sd of the original levels.

**Recommendation.** Option 1, plus a sentence in the `drift` parameter documentation.
**Expected change.** `drift = FALSE` replicates get an sd of about the residual sd (46 rather than 120
on AirPassengers). `drift = TRUE` (the default) and `target_drift` runs are unchanged.

### CORE-09 follow-up: exact tie tests against floating-point cell means
After the two CORE-09 commits, the degree-0 counts partition the sample, both in the nD moments and
inside `NNS.dep`'s per-cell copula. In `NNS.dep` a tie is still detected with an exact `dx == 0`
against a sequentially summed cell mean, so whether a tie is recognised can depend on summation order
(DEP-09). For example, the independent {-1,0,1} case only improves (Dependence 0.288 → 0.160) because
the cell means are exact there.

**Recommendation.** Snap `|dx| <= 8 eps max|x|` to 0 in `copula_signed` (the fix proposed in DEP-09)
and use a long-double (or pairwise) cell mean.
**Expected change.** Only discrete or tied data change, and R and NNS-core / Python then agree on them.

### DEP-14 (remaining part, after CORE-09/CORE-10)
The CORE-10 commit fixes the dimension-generic early return, and the CORE-09 commit removes the
tie double-counting. The skewed-marginal floor (degree-0 null of 0.5 versus P(X <= mean) ≠ 0.5)
remains. For example, independent {-1,0,1} with means 0 still gives a copula of 0.237. The DEP-14
options above (empirical marginal nulls) still apply.
