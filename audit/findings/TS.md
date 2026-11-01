# TS group findings: time series and resampling

Scope: Python `src/nns/{arma,var,seasonality,meboot,mc}.py` and R `ARMA.R`, `ARMA_optim.R`, `NNS_VAR.R`, `Seasonality_Test.R`, `NNS_meboot.R`, `NNS_MC.R`, `src/NNS_seas.cpp`, plus the ARMA/meboot C++ helpers in `src/internal_functions.cpp`, the man pages and the forecasting/sampling vignettes.

Work dir: `/home/user/audit/work/TS/`. The two quickest repro scripts are `repro_core.R` and `repro_core.py`. They print one line per headline finding.

Overall: deterministic Python vs R parity is **excellent**. 88 of 92 ARMA configurations, all 10 ARMA.optim configurations except one, and every NNS.VAR run (cor, NNS.dep, all, NaN-filled, naive_weights=FALSE) agree to about 1e-15. The meboot deterministic parts match exactly, and the meboot/MC statistics agree. So most problems below are **R algorithm bugs that the port copies faithfully**. The few true divergences are listed separately, and in most of them Python is the side that is right.

---

### TS-01: NNS.VAR "multivariate" forecast is an in-sample refit of the last h *observed* rows, not a forecast (R and Python)
- Type: r-bug (Python replicates it on purpose)
- Severity: critical
- Confidence: verified
- Where: R `R/NNS_VAR.R:397` (`lagged_new_values_train <- head(lagged_new_values, nrow - h)`) and `:435` (`IVs.test = tail(IV, h)`, where `IV` is the *training* lag matrix). Python `src/nns/var.py:511` (`lagged_train = lagged[:-h, :]`) and `:544` (`lagged_iv[-h:, relevant]`). `docs/conventions.md` describes this as intended ("evaluated on the last h lagged training rows (R's tail(IV, h))").
- Evidence: `work/TS/var_proof.py` and `var_proof.R`, on 3 monthly ETF return series with h=6 and tau=1.
  - R and Python both return SPY and GLD "forecasts" that are **exactly the last 6 observed returns** (rows 196-201). TLT is close to them.
  - Python check: adding 1000 to every univariate forecast row leaves `multivariate` bit-identical (`np.array_equal == True`). So the multivariate output carries no information about the forecast period.
  - On the 5-series panel with tau=3 (`var_r_5_12_3_cor.json`), corr(multivariate, last 12 observed) = 1.000 for TLT and GLD (MAE 0), 0.96 for SPY, 0.89 for USO and 0.85 for IWM.
  - `ensemble = 0.5*univariate + 0.5*multivariate`, so the headline ensemble output is half "replay of the last h observations".
- Which side is right: neither. The doc says "Returns the multi-variate forecasts". A forecast for t = n+1..n+h has to be evaluated on the lag-matrix rows for n+1..n+h. Those are the rows built from the univariate forecasts, and they are exactly the rows `head(..., -h)` drops.
- Proposed fix: evaluate on the forecast rows. In R, `IVs.test = tail(lagged_new_values[, -i, drop=FALSE], h)[, rel_vars]`. In Python, `np.delete(lagged, i, axis=1)[-h:, relevant]`. I tested this one-line patch (`work/TS/oos_var.py`, rolling 5 origins). Multivariate RMSE improves from 0.0535 to 0.0495 and ensemble from 0.0486 to 0.0481. Both still lose to the historical mean (0.0427), which is expected for returns. Changes public results.

### TS-02: `method = "means"` always returns a vector of zeros (R and Python)
- Type: r-bug (Python replicates it)
- Severity: high
- Confidence: verified
- Where: R `R/ARMA.R:270-272`. `Estimates[j]` is assigned only for "lin", "both" and "nonlin", so the means estimate computed at `:262-266` is thrown away. Python `src/nns/arma.py:505-506` (`else: estimate = 0.0`), which copies this on purpose.
- Evidence: `repro_core.R` and `repro_core.py`: `NNS.ARMA(AirPassengers, h=4, seasonal.factor=12, method="means")` gives `0 0 0 0` on both sides. ARMA grid configs 12-15 (`arma_py.py`) give all-zero output on both sides. Adding `pred.int` makes R fail with "function cannot be evaluated at initial parameters", because meboot of a constant vector fails, and Python raises the same message.
- Which side is right: neither. The Rd says "Means for each subset are returned with (method = "means")".
- Proposed fix: add `if (method == "means") Estimates[j] <- Lin.estimates`, and the Python equivalent `estimate = lin_estimate`. Changes public results, from 0 to sensible values.

### TS-03: Nonlinear ARMA forecasts are clipped at 0 even when `negative.values = TRUE` or auto-detected (R and Python)
- Type: r-bug (Python replicates it)
- Severity: high
- Confidence: verified
- Where: R `R/ARMA.R:246` (`Regression.Estimates <- pmax(0, Regression.Estimates)` with no `negative.values` guard). The linear branch at `:267` is guarded. Python `src/nns/arma.py:472`.
- Evidence: `repro_core.*`. A zero-mean 12-period sine plus noise (last 12 obs about 0.9 ... -3.7) with `seasonal.factor=12, negative.values=TRUE` gives a nonlin forecast of `0.88 2.20 3.23 2.83 1.28 0 0 0 0 0 0 0`. For SPY monthly returns, nonlin forecasts are all >= 0 (grid configs 58-59). "both" is biased upward by the same mechanism.
  - This makes every nonlin/both forecast of returns, spreads or demeaned data wrong. It also biases `NNS.ARMA.optim`, and through it `NNS.VAR`'s univariate stage, for any series with negative values.
- Which side is right: neither. The `negative.values` docs say the forecast is allowed to be negative.
- Proposed fix: `if(!negative.values) Regression.Estimates <- pmax(0, Regression.Estimates)` (Python: guard with `if not negative_values`). Changes public results for series with negative values.

### TS-04: NNS.seas has no null calibration: it flags 20-75% of all periods as "seasonal" in iid noise and favours periods near n/2.5, and the ARMA defaults inherit this (R and Python)
- Type: r-bug (methodology; Python replicates it)
- Severity: high
- Confidence: verified
- Where: R `src/NNS_seas.cpp:122-163`. A period passes when the sample CV of the reverse-stride subsample is <= the full-series CV. The test has no sample-size or multiple-testing correction, and the minimum CV wins. Python `src/nns/seasonality.py:83-124`.
- Evidence:
  - `work/TS/seas_noise.py`, iid normal n=1000: about 112 of 500 periods are flagged when mean=10, and 366 of 500 when mean=0.0005 and sd=0.01 (return-like, where CV is ill-conditioned). The best period is always 216-498. The documented "result of 1 = no seasonality" never happens.
  - `work/TS/seas_correct.py`:
    - Raw AirPassengers gives best=50, and 12 ranks 15th of 18.
    - A noise-free period-12 sine gives best=48. With noise, 12 ranks 5th.
    - A linear trend 1..100 gives best=34.
  - For daily ETF returns (n=1000), NNS.seas returns 378-495 "seasonal" periods. That is the root cause of the cost of NNS.VAR's univariate stage (TS-21).
  - Effect on forecasts (`work/TS/oos_arma.py` / `oos_arma.txt`, rolling-origin h=12 MAE). Default `NNS.ARMA(x, h)` (auto period):

    | Series | Default NNS.ARMA | Comparison |
    |---|---|---|
    | AirPassengers | 68.3 | seasonal naive 35.9, Holt-Winters 14.8, NNS.ARMA sf=12 lin 18.9 |
    | VIX monthly | 13.1 | naive 5.0, mean 3.1 |

    With an explicit `seasonal.factor = 12` NNS is sane.
- Which side is right: neither. A CV computed on 2-3 points is very noisy, so picking the minimum selects long periods by chance. On trending data CV mostly measures the trend.
- Proposed fix (R and Python): require a minimum number of points per subsample (e.g. >= 4-5, i.e. period <= n/5) and compare against a permutation or bootstrap null for the CV ratio. At minimum, document that the auto mode is unreliable and recommend `NNS.ARMA.optim` with an explicit candidate set. Changes public results.

### TS-05: `weights` are applied twice in the recursive ARMA path, so unnormalised weights scale forecasts by sum(w)^2 (R and Python)
- Type: r-bug (Python replicates it), plus a Python api issue
- Severity: medium
- Confidence: verified
- Where: R `R/ARMA.R:247, 266` (`sum(est * Weights)`) and then again at `:270, 272` (`sum(Lin.estimates * Weights)`, `sum(Nonlin.estimates * Weights)`). Python `src/nns/arma.py:473, 494, 500, 504`.
- Evidence (`repro_core.*`; grid configs 17/18, 40/41, 63/64):
  - AirPassengers with `seasonal.factor=c(12,24)`. `weights=c(.5,.5)` gives 415 389 453. `weights=c(1,1)` gives **1660 1557 1812** (4x) for lin and nonlin, but only 2x for "both", which weights once.
  - A single lag with `weights=0.5` gives 0.5x for lin (vectorised path, weighted once) but 0.25x for nonlin.
  - Python also broadcasts a wrong-length vector without an error: `weights=[1.0]` with 2 lags gives 2x the forecast.
- Which side is right: neither. The vignette says weights should sum to 1, but nothing enforces that, and the same weights give different scalings by method.
- Proposed fix: drop the second multiplication (`Estimates[j] <- Lin.estimates` / `Nonlin.estimates`), normalise user weights to sum 1, and check `length(weights) == length(lag)`. For weights that already sum to 1 this does not change results.

### TS-06: `NNS.ARMA.optim(linear.approximation = FALSE)` never evaluates nonlin or both and silently returns the lin-only answer (R); Python does run the search
- Type: r-bug and parity-mismatch (Python is right)
- Severity: medium
- Confidence: verified
- Where: R `R/ARMA_optim.R:197-254`. `nns.estimates.indiv` is only computed for `j=="lin"` or with `linear.approximation`. For nonlin/both without approximation it stays `numeric()` (reset at `:262`), so `min()` is `Inf` and those methods can never win. Python `src/nns/arma.py:148-165` actually evaluates them.
- Evidence: `optim_py.py` config `ap_noapprox` and `repro_core.*`. R returns `lin 12 0.07894`, identical to `lin.only=TRUE`, in 0.02 s. Python returns `both [12,24] 0.0462` in 1.3 s.
- Which side is right: Python. The docs say FALSE is "a more exhaustive search over the objective space".
- Proposed fix (R): in the nonlin/both branches without approximation, call `NNS.ARMA(..., method = j)` on each `seasonal.combs[[i]]` column. Changes R results.

### TS-07: In NNS.ARMA.optim, the "both" objective scores a different model than the one it returns (R and Python)
- Type: r-bug (Python replicates it)
- Severity: medium
- Confidence: verified
- Where: R `R/ARMA_optim.R:246-253` scores `(lin path + nonlin path)/2`. The final `predicted`, `errors` and `results` come from `NNS.ARMA(method="both")` (`:341`, `:472-475`), which averages lin and nonlin *inside* the recursion. Python `src/nns/arma.py:136-147` vs `:215` and `:300-321`.
- Evidence: `work/TS/both_check.py` on AirPassengers[1:132] with sf=12. The path-average objective is 0.03975, which is what is reported as `obj.fn` and used to pick "both". The objective of `NNS.ARMA(method="both")`, i.e. of the returned `errors`, is 0.04633. The two forecasts differ by up to 9.4 passengers.
  - Because `weight.SSE` is computed on the re-forecast (TS-08), this mismatch can also flip the equal-weights decision for multi-period "both".
- Which side is right: neither. The selection criterion and the returned model have to match.
- Proposed fix: either score `NNS.ARMA(method="both")` in the search, or return the path average. Changes public results.

### TS-08: NNS.ARMA.optim's equal-weights candidate is never evaluated (R and Python)
- Type: r-bug (Python replicates it)
- Severity: medium
- Confidence: verified
- Where: R `R/ARMA_optim.R:341-346` (`predicted <- NNS.ARMA(..., weights = NULL)`, then `weight.SSE <- eval(obj.fn)` *before* any equal-weight forecast is made). Same at `:381-386`. Python `src/nns/arma.py:215-233`.
- Evidence: `work/TS/eqw.py`. For lin/nonlin, `weight.SSE` is identical to `nns.SSE` (vix: 4.5753 vs 4.5753; ret: 2.8252 vs 2.8252), so `weight.SSE < nns.SSE` can never hold and `$weights` is always NULL. The vignette says the optimizer "compare[s] different weights".
- Which side is right: neither, by the documented intent.
- Proposed fix: compute `predicted` with `weights = rep(1/k, k)` first, then compare its objective with `nns.SSE`. Changes results where equal weights are better.

### TS-09: The `nns.regress` decision is reported but never applied to `$results` (R and Python)
- Type: r-bug (Python replicates it)
- Severity: medium
- Confidence: verified
- Where: R `R/ARMA_optim.R:450` sets `variable <- regressed_variable`, but `model.results` at `:472` and `:475` is computed from `OV`, the raw series. The regress test also always uses `shrink=TRUE` (`:444`), whatever `nns.shrink` is. Python `src/nns/arma.py:288-292` (`del final_predicted, values`) and `:300-321`.
- Evidence: `ap_max` config. `nns.regress=TRUE`, yet `results` equal `NNS.ARMA(raw)` exactly (`np.allclose` True). The smoothed-series forecast is 442 446 449 vs the returned 580 593 483.
- Which side is right: neither. Either apply it, or document `nns.regress` as advisory only.
- Proposed fix: use `variable` (and `shrink=TRUE` when regress was chosen) in the final `NNS.ARMA` calls. Changes public results.

### TS-10: ARMA prediction intervals have poor coverage and are built in a questionable way (R and Python)
- Type: r-bug (methodology; Python replicates it), plus a docs issue
- Severity: medium
- Confidence: verified
- Where:
  - `NNS.ARMA(pred.int)` (`R/ARMA.R:279-291`, Python `arma.py:674-713`) bootstraps the *forecast path* itself with NNS.MC across rho in [-1,1]. The band at step k therefore reflects how the h forecasts vary over time, not forecast uncertainty, and it does not widen with horizon.
  - `NNS.ARMA.optim` (`R/ARMA_optim.R:464-480`, Python `arma.py:296`) uses `UPM.VaR((1-p)/2, 0, errors)` for *both* sides. That is the upper-tail quantile, so the band is symmetric even when the error distribution is skewed, and `|bias|` is added on top after the bias has already been subtracted.
  - The Rd says the optim intervals are "Constructed using the maximum entropy bootstrap"; they are deterministic.
- Evidence: `work/TS/pi_cov.py`, 11 rolling origins, h=12, nominal 95% coverage:

  | Series | NNS.ARMA pred_int | NNS.ARMA.optim |
  |---|---|---|
  | AirPassengers | 0.97 | 0.65 |
  | VIX monthly | 0.62 | 0.82 |
  | SPY returns | 0.48 | 0.90 |
- Proposed fix: base the bands on the validation errors by horizon, with LPM.VaR for the lower side, and widen them with h. Fix the Rd. Changes public results.

### TS-11: `seasonal.factor = TRUE` (the default) is turned into FALSE in R, which produces all-zero forecasts whenever the top period's weight is 0; Python keeps TRUE semantics (parity mismatch, Python is better)
- Type: r-bug and parity-mismatch
- Severity: medium
- Confidence: verified
- Where:
  - R `R/ARMA.R:83` (`if(!is.null(best.periods) && !is.numeric(seasonal.factor)) seasonal.factor <- FALSE`; `best.periods` defaults to 1).
  - Then `ARMA_seas_weighting(FALSE, ...)` in `src/internal_functions.cpp:313-323` computes the weight as `(VarCV - CV)/sqrt(lag)`, normalised. That is `0/0`, giving weight 0, when the top row is a no-seasonality sentinel or a modulo-added period with CV = VarCV.
  - Python `arma.py:759-760` uses the TRUE path (weight 1).
- Evidence: `work/TS/zero.py` / `zero.R` and `repro_core.R`. White noise (mean 10) with `NNS.ARMA(x, h=4, modulo=7)` gives **R: 0 0 0 0**; Python gives 8.73 10.75 8.58 10.07. The same holds for `modulo=5`.
- Which side is right: Python, since a forecast of 0 for a series around 10 is never right. R should keep TRUE semantics, or treat a zero denominator as equal weights.
- Proposed fix (R): in `ARMA_seas_weighting`, when `denom == 0`, return equal weights instead of zeros.

### TS-12: `seasonal.factor = TRUE, best.periods = NULL` uses lag 1 in R, not the best period (parity mismatch, Python is right)
- Type: r-bug and parity-mismatch
- Severity: medium
- Confidence: verified
- Where: R `R/ARMA.R:134-135` passes `M$all.periods`, a data.frame with no element called `all.periods`, to `ARMA_seas_weighting(TRUE, ...)`. That function (`internal_functions.cpp:285-297`) then falls through to `lag = 1`. Python `arma.py:759-760` uses `periods[0]`.
- Evidence: `arma_edge.R` / `arma_edge.py` and `repro_core.R`. R gives 433.657 435.429 437.263, identical to `seasonal.factor = 1`, while the best period is 50. Python gives 317.5 392.25 422.25.
- Which side is right: Python. The docs say TRUE "Automatically selects the best seasonal lag".
- Proposed fix (R): pass `list(all.periods = M)` or use `M$Period[1]`.

### TS-13: `dynamic = TRUE` ignores `best.periods` (and `modulo`) in R (parity mismatch, Python is right)
- Type: r-bug and parity-mismatch
- Severity: medium
- Confidence: verified
- Where:
  - R `R/ARMA.R:195-208`: `M$all.periods$Period` is read from a data.frame, so it is NULL and its length is 0. `best.periods` becomes 0, and `seas.matrix$all.periods[1:0, ]` is row 1 only. With `best.periods = NULL` it reads `length(NULL)` as well.
  - R's re-seasoning step also drops `modulo` / `mod.only`.
  - Python `arma.py:449-457` honours both.
- Evidence: `work/TS/dyn.R`. A trace of `ARMA.seas.weighting` shows the lags `50 37` (static), then `50`, `50`, `51` for each dynamic step. Grid configs 22, 45 and 68 (`best_periods=2, dynamic=True`) differ: R gives 333 421 403..., Python gives 367 409 401....
- Which side is right: Python.
- Proposed fix (R): use `seas.matrix$all.periods$Period` and pass `modulo` and `mod.only` through.

### TS-14: NNS.ARMA.optim's lin stage ignores `negative.values` in R (parity mismatch)
- Type: r-bug and parity-mismatch
- Severity: low-medium
- Confidence: verified
- Where: R `R/ARMA_optim.R:207-214, 220-227`. `NNS.ARMA(...)` in the lin search does not pass `negative.values`, so lin forecasts are clipped at 0 during the search. The nonlin, both and final stages do pass it. Python `_evaluate_arma_optim` passes it.
- Evidence: `work/TS/neg_optim.R` versus the Python one-liner. A positive series trending toward 0 with `negative.values=TRUE` gives R `[12] lin 0.1574` and Python `[12] both 0.1659`, and the final forecasts differ (R -2.43 -2.43 -1.20 ..., Python -0.63 -0.63 -0.02 ...).
- Which side is right: Python, which is consistent with the user's flag.
- Proposed fix (R): add `negative.values = negative.values` to both lin calls.

### TS-15: Python `nns_arma(method="lin", seasonal_factor=<single int>, pred_int=...)` raises TypeError, while R returns intervals
- Type: python-bug and parity-mismatch
- Severity: medium
- Confidence: verified
- Where: `src/nns/arma.py:423-425` (`if pred_int is not None: raise TypeError("non-numeric argument to binary operator")`).
- Evidence: `pi.R` (R prints a 12x3 Estimates/Lower/Upper table) versus `pi_py.py` and `repro_core.py` (Python raises TypeError). This is the vignette's headline configuration (lin with sf=12) plus intervals.
- Which side is right: R works. The Python error string imitates an old R failure.
- Proposed fix: in the vectorised lin branch, collect the per-component forecasts as `lin_regression_estimates` and call `_with_prediction_intervals` as the recursive branch does.

### TS-16: Python nns_meboot / nns_mc crash on vector `target_drift` / `target_drift_scale`, which R vectorises and documents
- Type: python-bug and api
- Severity: medium
- Confidence: verified
- Where: `src/nns/meboot.py:76-119` vectorises only `rho`. At `:146-158`, a list `target_drift` reaches `baseline = intercept + recon_slope * time`. `mc.py:30-42` passes it through unchanged.
- Evidence: `meb_api.py` and `repro_core.py`. `nns_meboot(ap, reps=5, rho=0, target_drift=[1,7])` raises `ValueError: operands could not be broadcast`, and the `target_drift_scale` list raises TypeError. R (`meb_api.R`) returns a 12x2 list-matrix. This is the documented example "Vectorized drift with a single rho" (`NNS_meboot.R:79-82`) and the Sampling vignette.
- Proposed fix: vectorise over `target_drift` and `target_drift_scale`, as R's `Vectorize` does with rho, returning a list of results.

### TS-17: NNS.seas strict cap uses integer `n/2` in R but float `n/2` in Python, so odd-length series differ
- Type: parity-mismatch
- Severity: low-medium
- Confidence: verified
- Where: R `src/NNS_seas.cpp:221` (`Period[i] < n/2`, int division). Python `src/nns/seasonality.py:334` (`periods < (n / 2.0)`).
- Evidence: `seas_r.R` / `seas_py.py`, case `odd13` (n=13). R gives periods `[3, 1]`; Python gives `[3, 6, 1]`. This affects every odd-length series whose period floor(n/2) passes, e.g. n=201 and period 100.
- Which side is right: R is the reference. Period floor(n/2) leaves only 2 points per subsample, so excluding it is also the more defensible choice.
- Proposed fix: `keep = periods < n // 2`.

### TS-18: NNS.seas tie ordering differs between Python and R, which can change `best.period`
- Type: parity-mismatch
- Severity: low
- Confidence: verified
- Where: Python `seasonality.py:261` (`np.lexsort((periods, coef))` breaks ties by ascending period). R `NNS_seas.cpp:157, 195, 211, 228` uses `std::sort` on CV only; for 16 elements or fewer that is insertion sort, which keeps insertion order.
- Evidence: `seas_tie.py` and `repro_core.py`. The zero-mean alternating series `c(-1,1,...)` with `modulo=c(7,12), mod.only=FALSE` gives best period R 2 and Python 1. The `s12` series has 35 periods tied at CV = VarCV, and their order differs.
- Which side is right: R's order is implementation-defined for more than 16 elements, so exact parity is impossible there. Use a stable sort on CV (`np.argsort(coef, kind="stable")`), which matches R exactly for 16 elements or fewer and is deterministic.
- Proposed fix: as above. Ties only.

### TS-19: Constant series forecasts 0 (auto seasonality) or NaN (numeric `seasonal.factor`) (R and Python)
- Type: r-bug (Python replicates it, and special-cases it at `arma.py:407-413`)
- Severity: low-medium
- Confidence: verified
- Where:
  - Auto path: all CVs are 0, so `VarCV - CV = 0` and the weights are 0 (`internal_functions.cpp:323`).
  - Numeric path: `Relative.seasonal = 0/0` (`R/ARMA.R:116-123`).
- Evidence: `repro_core.R`. `NNS.ARMA(rep(5,60), h=3)` gives `0 0 0`, and with `seasonal.factor=12` gives `NaN NaN NaN`. Grid configs 33/34, 56/57 and 79/80 match on both sides.
- Which side is right: neither. The forecast of a constant 5 is 5.
- Proposed fix: return `rep(last value, h)` when `sd == 0`, or use equal weights when the denominator is 0.

### TS-20: meboot `rho` targets the correlation of detrended residuals, not of the series; MC "rho = -1" replicates correlate about +0.81 with a trending series (R and Python)
- Type: docs / r-bug (Python replicates it)
- Severity: low-medium
- Confidence: verified
- Where: R `R/NNS_meboot.R:196-243` (targets `orig_res`; the trend baseline is added back at `:252`). Python `meboot.py:283-347`. The Rd says "every replicate is mixed so that its dependence on the original series matches rho". The NNS.MC Rd says it samples "from the full [-1,1] correlation space".
- Evidence: `meb_cmp.txt` and `mc.R`. On AirPassengers, mean Spearman(replicate, x) is 0.807 at rho=-1, 0.929 at rho=0 and 0.999 at rho=1, identically in R and Python. On VIX, which has no trend, the targets are met (-0.998, 0.001, 0.9996).
- Proposed fix: document that rho applies to detrended residuals, or calibrate on levels when `drift=TRUE`. Docs fix at minimum.

### TS-21: NNS.VAR / NNS.ARMA.optim runtime explodes on return-like data because of seasonal-candidate inflation; Python's lin recursion is a hot Python loop
- Type: perf
- Severity: low-medium. Python is already faster than R everywhere it was timed.
- Confidence: verified
- Where: `R/NNS_VAR.R:336-346` / `var.py:381-394` feed every `NNS.seas(..., modulo=min(tau), mod.only=FALSE)` period into `NNS.ARMA.optim`. That is 129-198 candidates per daily-return column (TS-04), so the greedy search is O(K^2 * h_eval * lags). In Python the cost is `arma.py:476-494`: per-step and per-lag `_fast_lm`, called 2.7M times.
- Timings (4 shared cores, ncores=1):

  | Workload | R | Python |
  |---|---|---|
  | NNS.VAR, 5 daily return series, n=500, h=5, tau=2 | 14.3 s | 8.0 s |
  | NNS.VAR, same, n=1000 | 47.0 s | 27.0 s |
  | NNS.ARMA.optim, VIX daily n=1000, sf=1:24 | 1.14 s | 0.30 s |
  | NNS.ARMA.optim, VIX daily n=3000, sf=1:24 | 11.5 s | 4.4 s |
  | NNS.ARMA nonlin h=50, n=3000 | 0.51 s | 0.36 s |
  | NNS.VAR monthly, k=3, NNS.dep | 2.35 s | 4.06 s (includes first-call lazy imports) |

  Profile of NNS.VAR at n=1000 (`work/TS/prof_var.out`): `_fast_lm` 11.2 s cumulative, `_generate_vectors` 8.8 s, nns_arma self time 5.1 s.
  - Profile of optim at n=3000 (`prof_optim.out`): 50% of the time is in native `_copula`, called from `nns_reg`'s default `order` dependence. That is 2422 calls at about 1.8 ms average, with component series of up to 2400 points.
- Proposed fix:
  - Cap the candidates passed from VAR, e.g. periods <= n/5, or only top-k by CV.
  - In the lin recursion, cache per-lag OLS sums. Only one component series gains a point at each step, so the other lags' fits can be reused.
  - Results unchanged.

### TS-22: Python NNS.seas is about 10x slower than R at n = 1e4 (pure-Python accumulation loops) and carries dead goal-seeking code
- Type: perf / dead-code
- Severity: low
- Confidence: verified
- Where: `src/nns/seasonality.py:227-249`. `_mean`, `_mean_exact` and `_sample_sd_from_mean` are per-element Python loops, and `_mean` and `_mean_exact` are identical. The `exact`/`exact_cv` flag and the "recompute if within 1e-12 of var_cov" branch (`:196-203`, `:218-221`) recompute the same value. `_sample_sd` (`:160`) is unused, and `_cv_or_fallback` duplicates `_cv_stat`.
- Evidence: `seas_time.R` / `seas_time.py`:

  | n | R | Python |
  |---|---|---|
  | 1e3 | 0.002 s | 0.005 s |
  | 1e4 | 0.006 s | 0.059 s |
  | 1e5 | 0.33 s | 0.55 s |
- Proposed fix: use `math.fsum` or numpy-sequential cumsum (the order-preserving `np.cumsum(...)[-1]` reproduces sequential accumulation), and delete the dead branches. Results unchanged.

### TS-23: Python ARMA argument validation gaps (silent wrong behaviour or raw IndexError)
- Type: api
- Severity: low
- Confidence: verified (`arma_edge.py`, `repro_core.py`)
- Where: `src/nns/arma.py:400-402, 723-746, 941-967`.
- Evidence:
  - `training_set=200 > n=144`: Python silently uses the whole series; R errors.
  - `training_set=0`, `seasonal_factor=200 > n`, or n=3 with the default: `IndexError: index -1 is out of bounds`. R gives `[y] must contain at least two observations`.
  - `seasonal_factor=None`: `TypeError: int() argument ... 'NoneType'`. `_is_numeric_seasonal` treats anything that is not a bool as numeric.
  - `weights` length mismatch: silently broadcast (TS-05).
  - `h=1, pred_int=.95`: `ValueError: incorrect number of dimensions`, which copies an R crash.
  - `nns_arma_optim(pred_int=None)`: `TypeError: non-numeric argument to mathematical function`, which copies an R crash.
  - `bias.shift` is returned as `-0.0`.
- Proposed fix: validate `1 <= training_set <= n`, `1 <= lag < training_set/2`, `len(weights) == len(lags)` and non-bool `seasonal_factor` types. Let h=1 and `pred_int=None` work, skipping the intervals in the latter case. Return `0.0` instead of `-0.0`.

### TS-24: Python meboot disables `expand_sd` whenever `xmin`/`xmax` is given; R does so only when `force.clt` is TRUE
- Type: parity-mismatch
- Severity: low
- Confidence: verified by code reading (the stochastic effect is small)
- Where: Python `src/nns/meboot.py:171-173` (unconditional). R `R/NNS_meboot.R:158-159` (`if (isTRUE(force.clt)) { expand.sd <- FALSE; force.clt <- FALSE }`).
- Evidence: `nns_meboot(x, xmin=0, force_clt=False, expand_sd=True)` expands the sd in R but not in Python.
- Proposed fix: `if (xmin_arg is not None or xmax_arg is not None) and force_clt: force_clt = expand_sd = False`.

### TS-25: meboot `drift=FALSE` / `target_drift=0` inflates residual amplitude to the full sd(x), trend variance included (R and Python)
- Type: r-bug (Python replicates it)
- Severity: low
- Confidence: verified
- Where: R `R/NNS_meboot.R:260-262` (`expand.sd(x = x, ...)` on levels after the trend is removed). Python `meboot.py:198-199`.
- Evidence: `meb_cmp.txt`, AirPassengers with `drift=FALSE`. The replicate sd is 123, about sd(x)=120, against a residual sd of about 45. Replicates go to -124 passengers, and the replicate slope is still 0.93 against an original of 3.1.
- Proposed fix: when the reconstruction slope differs from the original, expand against `sd(baseline + orig_res)` rather than `sd(x)`.

### TS-26: R doc vs code mismatches
- Type: docs
- Severity: low
- Confidence: verified
- Where and evidence:
  - `man/NNS.ARMA.Rd:38`: `best.periods` is documented as "[2] (default)", but the code default is 1.
  - `man/NNS.ARMA.optim.Rd:44, 46`: `obj.fn` is documented as Spearman/SSE, but the code uses MSE / co-movement. `objective` is documented as "max" (default), but the code default is "min". The title says "Minimum sum of squared errors". `pred.int` is said to use meboot; it uses VaR on the errors.
  - `man/NNS.MC.Rd:28, 30`: `lower_rho`/`upper_rho` are documented as ".01 default", but the code uses -1/1.
  - `man/NNS.VAR.Rd`: `multivariate` is documented as `NNS.stack(method = c(1, 2))`, but the code uses `method = 1`.
  - `man/NNS.seas.Rd:19`: says it returns "No Seasonality Detected". It actually returns Period 1 with CV = VarCV.
  - The forecasting vignette describes `mod.only` backwards ("ensure empirical patterns are kept"). Its hard-coded `NNS.ARMA.optim` output (obj 18.1809, bias 0) is stale: the current R and Python results are 18.0768 and bias -3.173, and they agree with each other.
  - NNS.ARMA with `h=0` returns 1 value (`for (j in 1:h)` with h=0). Python raises.
- Proposed fix: regenerate the Rd from corrected roxygen. Use `seq_len(h)`, and error when h < 1.

### TS-27: Python docs are stale about ARMA multi-lag parity
- Type: docs
- Severity: low
- Confidence: verified
- Where: `NNS-python/docs/conventions.md` ("For explicit numeric multi-lag seasonal factors ... Installed R NNS instead computes ... reverse steps 1:length(seasonal.factor) ... covered by a strict xfail") and the matching sentence in `docs/api_status.md:56`.
- Evidence: R 13.3 `R/ARMA.R:111` now uses `seasonal.factor[i]`. Grid configs 3, 7 and 11 (`[12,24,36]`) and `test_sunspots_arma_example_matches_installed_r` ([132,276]) match R exactly. No such xfail exists.
- Proposed fix: delete the divergence paragraph.

---

## Coverage (checked and fine)

- **NNS.seas**:
  - Periods and CVs are identical to R on AirPassengers, VIX monthly, SPY, five synthetic series (period 12/7 sine, trend+sine, zero-mean sine, random walk, white noise), n=10/11, and the mean-zero acf fallback.
  - `modulo=12` with mod.only TRUE, and `modulo=c(7,12)` with mod.only FALSE, are identical up to ties (TS-18).
  - Input validation (NaN, Inf, n<5 sentinel) and pandas/int input are fine.
- **NNS.ARMA**:
  - 88 of 92 grid configs match R to 1e-7 (`arma_cfg.py`, `arma_r.R`, `arma_py.py`). The grid covers:
    - all four methods;
    - sf TRUE, FALSE (best.periods 2/3), 12 and c(12,24,36);
    - training.set and h in {1, 5, 12, 24, 45};
    - shrink (single and multi-lag, h not a multiple of the lag);
    - weights "equal", c(.5,.5) and c(1,1);
    - modulo;
    - negative.values (explicit and auto);
    - VIX, returns, white noise, trend+season and constant series;
    - sf=1 and sf=12.5.
  - The four mismatches are the R dynamic bug (TS-13).
  - The prediction-interval structure (column names, h>=2) works for nonlin and multi-lag lin.
  - `random_seed` reproducibility works.
- **NNS.ARMA.optim**: AirPassengers (doc examples, 1:24, lin.only, objective="max", the vignette RMSE objective, a custom callable obj_fn), VIX monthly (1:24 and a fixed training.set), SPY monthly price and SPY monthly returns all match R in periods, method, obj.fn, shrink, nns.regress, bias.shift, results and PIs to 1e-6. The only exception is linear.approximation=FALSE (TS-06). The Python optimizer is 1.5-4x faster than R.
- **NNS.VAR**:
  - 3 and 5 monthly ETF return series with h=12/6/4/3, tau=1/2/3 and dim.red cor/NNS.dep/all: univariate, multivariate, ensemble and relevant_variables are identical (~1e-16).
  - Interior plus trailing NaN interpolation is identical, as is `naive_weights=FALSE`.
  - DataFrame column names are kept.
- **NNS.meboot**:
  - The deterministic parts (xx, z, dv, dvtrim, xmin, xmax, desintxb, ordxx) match exactly on AirPassengers, VIX and 1000 daily SPY returns.
  - The replicate statistics agree with R to within sampling error: Spearman/Pearson to x, column means and sds (force.clt gives identical colmean sd), slope, ensemble sd and quantiles.
  - These were compared at rho in {-1, -0.5, 0, 0.5, 1}, drift FALSE, target_drift 0, target_drift_scale 2, xmin, pearson and nnsdep.
  - Runtimes are similar.
  - n=1, reps=1, int and pandas input work.
  - Constant series, n=2 and exact-linear series error identically in R and Python ("function cannot be evaluated at initial parameters"). This is unfriendly but consistent.
- **NNS.MC**: the rho grid, the `exp` transform and the replicate names ("rho = 0.707106781186548") match. The default 201x30 run takes 4.1 s in R and 3.5 s in Python; the ensemble mean and sd and the per-rho Spearman profiles match.
- **Out-of-sample sanity** (`oos_arma.txt`, `oos_var.txt`):
  - With an explicit sf=12, NNS.ARMA on AirPassengers beats seasonal naive and ARIMA but loses to Holt-Winters.
  - On VIX, only "lin sf=12" is competitive with ARIMA(1,0,1); the mean forecast is best.
  - On SPY price, the default NNS.ARMA narrowly beats naive.
  - NNS.ARMA.optim is worse than a fixed sf=12 on all three, which suggests it overfits a single validation window.
  - On monthly returns, nothing in NNS beats the historical mean (VAR ensemble RMSE 0.0486 vs 0.0427).
  - Verdict: sane only with a user-chosen seasonal factor. The defaults are not sane (TS-04, TS-03).
