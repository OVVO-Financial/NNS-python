# Priority list by codebase

Importance is out of 10: how wrong or how costly the problem is for a user on a common path (10 = silently wrong
results on a mainstream function; 1 = cosmetic). Status: **Fixed** = commit on the branch / patch in `r-patches/`;
**Decide** = method question for the author in `R_PROPOSALS.md`, nothing changed; **Open** = not done.
IDs refer to `findings/*.md`.

## 1. R package (OVVO-Financial/NNS) — the reference implementation

| Score | ID | Problem | Status |
|---|---|---|---|
| 10 | TS-01 | `NNS.VAR` multivariate "forecast" is the last h observed rows refitted, not a forecast | Fixed |
| 9 | DEP-01 | `NNS.dep(x, -x)` reports Correlation +1 (sign lost for perfect negative dependence) | Fixed |
| 9 | PM-01 | Integer targets silently replaced by the mean: `NNS.CDF(1:10)` = 0.5 everywhere, CDF of any integer data wrong | Fixed |
| 9 | TS-03 | Nonlinear ARMA forecasts clipped at 0 even with `negative.values = TRUE` (every return/signed-series forecast) | Fixed |
| 8 | TS-04 | `NNS.seas` has no null calibration: flags periods in white noise, picks wrong periods; default ARMA forecast ~4x worse than Holt-Winters on AirPassengers | Decide |
| 8 | DEP-02 | `NNS.caus(tau = k)` never lags the series; lagged causation is undetectable | Decide |
| 8 | REG-04 | `dy.d_` partial derivatives use the row mean of all regressors (y = 2·x1 gives dy/dx2 ≈ 0.9) | Fixed (magnitudes still low: Decide) |
| 8 | TS-05 | ARMA `weights` applied twice (weights c(1,1) gives 4x the forecast) | Fixed |
| 7 | CORE-01 | >32 targets: LPM/UPM at degree ≥ 3 catastrophically wrong (negative moments, 1e12 errors) | Fixed (C++) |
| 7 | TS-02 | `NNS.ARMA(method = "means")` always returns zeros | Fixed |
| 7 | TS-07/08/09 | `NNS.ARMA.optim` scores a different model than it returns; equal-weights candidate never evaluated; `nns.regress` ignored | Fixed |
| 7 | STK-02 | `NNS.boost` on a two-valued numeric response returns class codes 1/2 instead of the values | Fixed |
| 7 | SDA-02 | Robust ANOVA resamples only the first min(n1, n2) rows of the longer group | Fixed |
| 7 | DEP-05 | `NNS.dep` independence floor: ~0.3 on pure noise at n=1e3, decays like n^-1/4 | Decide |
| 7 | DEP-03 | `NNS.caus` depends on location/scale (`NNS.norm` can flip a sign; +10 to y changes the answer) | Decide |
| 6 | REG-05 | `dy.dx` result depends on row order (step from unsorted x) | Fixed |
| 6 | REG-08 | `NNS.part` rounds regression-point x for integer x (y = 2x off by 1); changes ARMA nonlin forecasts | Fixed |
| 6 | PM-06 | `NNS.moments` sample skewness inflated by (n/(n-1))^1.5 | Fixed |
| 6 | PM-07 | Multivariate `NNS.CDF` survival/hazard wrong for d ≥ 3 | Fixed |
| 6 | STK-03 | `NNS.stack` "ALL" candidate scored as a constant mean but deployed as `n.best = "all"` | Fixed |
| 6 | SDA-04 | `NNS.ANOVA(tails = "left"/"right")` crashes | Fixed |
| 6 | SDA-10 | ANOVA `Certainty` documented as a calibrated probability; it is not (0.73–0.96 under the null) | Decide |
| 5 | DEP-04, DEP-13 | `NNS.caus` p-value ignores direction; `NNS.dep` p-value wrong tail, ~2x nominal size, writes Rplots.pdf | Fixed |
| 5 | STK-05 | Character response coded by first appearance: results depend on row order | Fixed |
| 5 | SDA-05, SDA-09 | Dominance only checked at sample points (false TSD); no float tolerance (textbook SSD flips on 1e-16) | Fixed |
| 5 | DEP-12 | Signed "Correlation" = max(xy, yx): biased positive, reports the weaker of two negatives | Decide |
| 5 | TS-10 | ARMA prediction intervals cover 48–97% at nominal 95% | Open |
| 5 | DEP-14 | `NNS.copula` does not go to 0 for independent skewed data | Decide |
| 4 | CORE-09, CORE-10 | Degree-0 tie conventions don't partition the sample; copula uses the bivariate shortcut for any dimension | Fixed |
| 4 | REG-11 / CORE-03 | Multivariate `noise.reduction = "mode"` differs between native and pure-R paths | Fixed |
| 4 | PM-15, X-06 | NA counted in n by ratio functions (CDF tops out < 1); `dy.dx` default `eval.point` always crashes | Fixed |
| 4 | DEP-21 | Book ch.14 describes a different `NNS.caus` from the code | Open (docs) |
| 3 | TS-06, TS-11–14, TS-19 | ARMA/optim edge cases (linear.approximation ignored, zero forecasts on noise, dynamic ignores args, constant series) | Fixed |
| 3 | STK-09, SDA-07, SDA-08, SDA-13 | Balancing drops single-member classes; `.uni` dominance wrong for unequal lengths / coincident curves | Fixed |
| 3 | PM-14, PM-17, PM-16, PM-22 | Hazard crashes on ties; fractional VaR tolerance; `excess_ret` tie rule; 0/0 ratio | Fixed |
| 3 | PM-08, TS-25 | Spurious modes with `multi = TRUE`; `meboot(drift = FALSE)` rescales to full sd | Decide |
| 2 | CORE-07, -08, -12, -13, -16 | int overflow in mode/gravity, heap over-read, generate.vectors OOB, negative degree accepted | Fixed |
| 2 | DEP-23, STK-14, docs | Dead code; man pages vs code | Fixed |

## 2. NNS-core (portable C++)

| Score | ID | Problem | Status |
|---|---|---|---|
| 8 | CORE-01 | Vectorised LPM/UPM prefix backend cancels catastrophically (shared with R) | Fixed |
| 7 | CORE-02 | R 13.1–13.3 M.reg / stack kernels missing, so Python reimplemented them 14–25x slower | Fixed (ported) |
| 6 | CORE-04 | Frozen at R 13.0; R → core → Python sync chain broken both ways | Fixed |
| 6 | CORE-06 | Threads spawned on every call, no thread limit (10–19x slower small calls, oversubscription) | Fixed |
| 5 | DEP-01, DEP-08, CORE-09, CORE-03 | Same dependence/tie/mode bugs as R | Fixed |
| 4 | CORE-05, CORE-18 | Sync gate marked logic changes "ready" after a comment rewrite; script-injection patterns in workflows | Fixed |
| 3 | CORE-07, -08, -12, -13 | Overflow / out-of-bounds edge cases (shared with R) | Fixed |
| 3 | Fidelity coverage | Suite covered 28 scalar cases; now 322 (vectors, nD, partition, dependence, M.reg) | Fixed |
| 2 | CORE-14, CORE-16 | `fast_lm` mean last-bit; negative degree accepted | Fixed |
| 2 | CORE-19 | Unused native bindings remain | Open |
| 2 | CORE-20 | R's M.reg workers call Rmath from threads (speculative hazard, R side) | Not changed |

## 3. NNS-python

| Score | ID | Problem | Status |
|---|---|---|---|
| 9 | REG-03, STK-08 | DataFrame `point_est` / `IVs.test` matched by position, not name: silently wrong predictions (mtcars 8579 vs 22) | Fixed |
| 9 | REG-06, STK-11 | Multivariate `nns_reg` 5–126x slower than R; boost 3.5–4.5x slower | Fixed (now at or faster than R) |
| 8 | STK-04, REG-12–16 | Class/factor levels ignore pandas Categorical and `class_levels`; worse classifiers (iris acc 0.82 vs 0.985); NA in factors became a level | Fixed |
| 8 | DEP-09, REG-09, STK-10 | `nns_dep` differs from R on tied/grid data (cell means summed differently), spreading into reg/stack/boost | Fixed |
| 8 | X-13 | pandas tests never ran in CI | Fixed |
| 7 | REG-19 | ~1,700 lines of dead legacy regression engine with different semantics (a fix had landed in a dead copy) | Fixed |
| 7 | PM-04 | NaN in `x` silently gives wrong degree-0 moments (and a 2.3 GB fallback) | Fixed |
| 7 | PM-02, PM-03, SDA-01 | `nns_cdf` and `fsd_uni` build n x n matrices: GBs of memory, OOM beyond ~3e4 | Fixed |
| 6 | All "mirror" items | Python copied every R bug in section 1; each now follows the fixed R | Fixed |
| 6 | STK-01 | Two Boston parity tests failing (test called R with different settings) | Fixed |
| 6 | DEP-07 | `nns_copula` segfaults on empty input, returns 1.0 for NaN | Fixed |
| 6 | PM-05 | `nns_cdf` loses precision on level-shifted data (values outside [0,1]) | Fixed |
| 6 | REG-02, REG-07 | `dy_dx` steps by bandwidth index not value; boundary fallback corrupts estimates | Fixed |
| 5 | REG-01 | `smooth = True` was an inexact imitation of R's `smooth.spline` (up to 9% off); now exact. Optional: see parity note | Fixed (your call) |
| 5 | SDA-03, SDA-06 | Robust ANOVA resampling differs from R; efficient set changes when unrelated columns are added | Fixed |
| 5 | TS-15–18, TS-23, TS-24 | ARMA intervals, meboot vector drift crash, seas period rules, validation | Fixed |
| 5 | DEP-10 | Internal gravity capped at 1e6 bins: partitions diverge on data with outliers | Fixed |
| 5 | X-11 | Test inputs depended on the LAPACK build, so cached R answers missed on other machines | Fixed |
| 4 | PM-10, PM-11, PM-12 | `target=None` gave NaN; `lpm_var` scalar only (21x slower in a loop); `ZeroDivisionError` on constants | Fixed |
| 4 | X-05, REG-21, REG-25, STK-18 | `plot=` missing on 8 functions; bool predictors; error messages | Fixed |
| 4 | DEP-15–18 | No matrix `nns_dep`; slow `nns_part` / `causal_matrix`; validation | Fixed |
| 3 | Review #4 | `lpm(target=NaN)` returns NaN; R raises | Open |
| 3 | Review #5 | Removed internal submodule names have no deprecation shims | Open |
| 3 | REG-17, X-12, X-01, X-03, X-04 | `nns_diff` flags; `dy_d_best` duplicate (deprecated); import time; CI pins | Fixed |
| 2 | Review #3 | `NNS_NUM_THREADS` read once; huge `n` gives TypeError | Open |
| 1 | X-09 | Subnormal inputs give NaN point estimates | Open (test adjusted) |

## Parity-only commits (optional)

These exist only to match R's arithmetic exactly. Keep the first two (ties on grid/discrete data flip partitions, which
changes real outputs); the rest can be reverted individually if exact R equality is not a goal.

| Commit subject | Effect if reverted |
|---|---|
| `dep: compute the bivariate copula with R's arithmetic` | nns_dep differs from R on tied/grid data (keep) |
| `nns_stack: reproduce R's cor and X* rounding` | Stack partitions/n.best can differ from R (keep) |
| `nns_reg: make smooth=True reproduce R's smooth.spline` | smooth=True up to ~9% off R (a different but valid smoother) |
| `nns_diff: find roots like R's uniroot` | Convergence flags / ~1e-7 differences |
| `anova: bootstrap from R's Mersenne-Twister stream` | Same distribution, not seed-for-seed equal to R |
| `reg/part: use R's long-double means` | Last-digit differences, occasional tie flips |
