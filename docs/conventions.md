# Conventions

## Build

NNS Python is packaged as the `NNS` distribution and imported with `import nns`.
It includes the `nns._nnscore` native extension backed by the vendored C++ core in
`extern/NNS-core`, while preserving Python fallbacks for the public APIs that route
through the native backend. CI parity is cache-backed and does not require
`Rscript`; `Rscript` is only needed for local cache regeneration.

## Degree-Zero Boundary

At degree zero, `LPM` uses `x <= T` and `UPM` uses `x > T`.
Equality is counted by `LPM` only.

For any non-empty finite input, `LPM + UPM = 1` at degree zero.

## Empty Input Divergence From R

R NNS returns `NaN` for empty input.
NNS Python raises `ValueError`.

Rationale: empty arrays in Python are upstream bugs, and NumPy convention is to warn or fail on empty reductions rather than silently produce a meaningful statistic.

## Co-Moment Length Mismatch Divergence From R

R NNS warns when `x` and `y` lengths differ, computes over the shorter length, and divides by the longer length.
NNS Python raises `ValueError`.

Rationale: mismatched co-moment inputs lose observations silently in R. Python callers should fix alignment before computing a bivariate statistic.

## PM Matrix Target Defaults

R `PM.matrix` uses column means when `target` is `NULL` or any non-numeric value.
NNS Python accepts `None` and `"mean"` for this behavior. NNS Python also broadcasts a
scalar numeric target across all variables; R requires callers to pass an
explicit vector such as `rep(0, ncol(variable))`. Target vectors whose length
does not match the number of variables raise `ValueError`.

## Classical Moment Normalization

`mean_pm`, `var_pm`, `skew_pm`, and `kurt_pm` use population normalization by
default, matching NumPy defaults and `NNS.moments(population = TRUE)`. `var_pm`
accepts `ddof` for NumPy-style variance scaling. `skew_pm` and `kurt_pm` do not
apply SciPy's optional finite-sample bias correction.

`nns_moments` is the public `NNS.moments` wrapper and returns R's dictionary
shape with `mean`, `variance`, `skewness`, and `kurtosis`. `nns_gravity` exposes
R's public `NNS.gravity` central-tendency helper. `fsd_uni`, `ssd_uni`, and
`tsd_uni` are the unidirectional stochastic-dominance wrappers behind R's
`.uni` exports. `co_lpm_nd`, `co_upm_nd`, and `dpm_nd` expose the public
n-dimensional partial-moment wrappers.

`nns_ss` maps to R's `NNS.SS` stochastic-superiority function, not to the
stochastic-dominance tests. It returns `p_gt = P(X > Y)`, `p_tie = P(X = Y)`,
and `p_star = p_gt + 0.5 * p_tie`. `NaN` values are omitted independently from
`x` and `y`, matching R's `na.omit` preprocessing. With
`confidence_interval=True`, intervals are computed through `nns_meboot`,
`lpm_var`, and `upm_var`; exact bootstrap parity with R is not expected because
the RNG streams differ. `random_seed` is a NNS Python-only reproducibility
convenience for that stochastic path.

`nns_sd_cluster` maps to R's `NNS.SD.cluster` default path. It iteratively
peels `sd_efficient_set` results and returns a dictionary of `Cluster_1`,
`Cluster_2`, ... memberships. The output contains variable names, not numeric
cluster labels; when names are omitted, NNS Python uses R-style `X_1`, `X_2`, ...
names. `type="continuous"` is supported for first-degree efficient sets.
`dendrogram=True` returns a plain dictionary mirroring R's `hclust` fields:
`merge`, `height`, `order`, `labels`, `method`, `call`, and `dist.method`.
NNS Python does not plot the dendrogram; it only returns the object data.

The stochastic-dominance implementation is deliberately pure NumPy. It mirrors
R's C++ SD core mathematically by sorting each column once, storing prefix sums,
and evaluating dominance on each pair's merged threshold grid rather than on one
global all-column grid. The full prefix-pair dominance matrix remains available
internally for verification and fallback. Large degree-1 discrete calls use an
exact order-statistic dominance matrix: with equal-length empirical samples,
one sample first-order stochastically dominates another exactly when every
sorted order statistic is at least as large and at least one is strictly larger.
Large degree-1 continuous and degree 2/3 calls use a lazy kept-only prefix scan.
Columns are visited in R's LPM-at-global-maximum order with original-index tie
breaks, and only already-kept candidates are tested against the current column.
Each prefix pair check applies min/mean/identical guards before evaluating
curves, then exits as soon as dominance is disproved.

These choices preserve exact R-style dominance semantics: no tolerances,
approximate equality, output reordering, or diagonal/identical-column behavior
changes are introduced. Polars is intentionally not used in this SD kernel
because the hot path is dense pairwise threshold evaluation rather than
data-frame grouping or filtering. R remains faster on some large finance
fixtures because its C++ path walks merged sorted thresholds in tight parallel
loops with minimal temporaries; NNS Python instead uses NumPy order-statistic blocks,
`searchsorted`, contiguous column storage, and early-exit scans to stay
dependency-light and pure Python for alpha.

`nns_cdf` maps to R's `NNS.CDF` deterministic non-plotting paths. It is a
partial-moment distribution wrapper rather than a textbook ECDF: `degree = 0`
uses R's lower-partial-moment frequency convention, and positive degrees use
`LPM.ratio` deformation. Univariate output columns follow installed R (`x` plus
`CDF`, `S(x)`, `h(x)`, or `H(x)`), while multivariate output keeps the final
column named `CDF` for all types, including survival, hazard, and cumulative
hazard. Plotting is ignored. The univariate `NA`/`Inf` comparison quirks are
handled inside `nns_cdf` without loosening the global partial-moment APIs.

## Dependence

`nns_dep` follows R's `NNS.dep` bivariate path, including `NNS.gravity` handling
for zero-range inputs and non-positive or non-finite bin widths. NNS Python also caps
the internal gravity bin count at `4 * len(input)` to prevent pathological
allocations on inputs where R's C++ `int` conversion effectively collapses an
absurd bin count. `abs(Correlation) <= Dependence` is not guaranteed by
`NNS.dep`; both R and NNS Python can return signed correlation magnitudes above the
dependence component for near-binary inputs.

## Copula

`nns_copula(x, y)` is the bivariate scalar form of R's `NNS.copula(cbind(x, y))`.
When targets are omitted, NNS Python uses column means, matching R's `target = NULL`.
The `target_x` and `target_y` arguments map to R's two-element target vector.

## Causation

`nns_causation(x, y)` maps to R's `NNS.caus(x, y, tau = 0, p.value = FALSE)`
numeric-vector path. It returns the two directional components and the named
signed net log-ratio key selected by R, either `C(x--->y)` or `C(y--->x)`.
`causal_matrix` maps to R's `NNS.caus.matrix` antisymmetric matrix convention.
`tau='ts'` uses `nns_seas(... )["periods"]` exactly like installed R: the first
period not exceeding `sqrt(length(x))` is selected per variable, including
harmonics when R selects them. Inputs with no eligible selected period follow
R's failure convention and raise. Numeric `tau` lag values remain fully
supported.

## Partition

`nns_part` maps to R's `NNS.part` but returns plain NumPy arrays instead of
`data.table` objects: `"dt"` and `"regression.points"` are dictionaries of
arrays. Installed R 12.1 only distinguishes `type = NULL` from any non-null
`type`: `None` uses XY quadrant splits, while every non-`None` value uses
X-only splits. This differs from documentation that implies separate `"X"`,
`"Y"`, and `"XONLY"` modes. NNS Python matches the installed binary.
`order="max"` is rejected with `TypeError`; installed R coerces it to `NA` and
returns a useless zero-order map. All five `noise_reduction` modes are
supported: `"off"`, `"mean"`, `"median"`, `"mode"`, and `"mode_class"`.

## Regression

`nns_reg` maps to R's univariate numeric `NNS.reg` path with
`factor.2.dummy = FALSE` and plotting disabled. Return keys match R's list names, but data.table outputs are plain
dictionaries of NumPy arrays. `multivariate_call=True` returns R's internal
two-column regression-point structure as `{"x": ..., "y": ...}` for
`nns_m_reg`, including after dimension-reduction projection. Matrix `x` without
dimension reduction dispatches to `nns_m_reg`.
Classification is supported for numeric/logical/factor-like class-code targets.
`smooth=True` follows installed R's ordinary piecewise fallback for univariate
inputs with fewer than four observations and for univariate `order="max"`; R
does not call `smooth.spline` there. Spline-eligible inputs use a private
fixed-`spar` cubic smoothing-spline adapter matching the `stats::smooth.spline`
subset used by `NNS.reg`: `spar = (dependence + 0.5) / 2`, R-style knots, and
R's interior-band trace ratio for lambda.
Factor predictor expansion is supported through the public `nns_reg` path.
When combined with dimension reduction, factor predictors are expanded with
R's full-rank dummy convention before synthetic `x.star` coefficients are
computed. For callers that want direct multivariate regression, use
`prepare_factor_predictors(...)` first and pass the returned numeric design
matrix into `nns_m_reg(...)`:

```python
from nns import nns_m_reg, prepare_factor_predictors

design = prepare_factor_predictors(
    x,
    point_est=point_est,
    factor_levels=(["low", "mid", "high"], None),
    names=("rating", "score"),
)
fit = nns_m_reg(design.x, y, point_est=design.point_est)
```

`prepare_factor_predictors(...)` uses the same full-rank dummy expansion as
`nns_reg(..., factor_2_dummy=True)`, combines training `x` and `point_est`
before expansion, and returns deterministic feature names.

Numeric dimension reduction is supported for `"cor"`, `"NNS.dep"`,
`"NNS.caus"`, `"all"`, `"equal"`, and numeric coefficient vectors. The
synthetic `x.star` projection follows R's min-max normalization and denominator
conventions, including joint normalization for `point_est`. In this dim-red
regression path, `tau="ts"` follows R's direct `Uni.caus` call and maps to a
fixed lag of `3`; public `nns_causation(..., tau="ts")` still uses the
`NNS.seas`-derived lag path. The `"NNS.caus"` branch uses the ported `Uni.caus`
internals and may differ from installed R at small asymmetric dependence
granularity.

`order="max"` follows installed R's univariate convention: fitted values are the
observed `y` values and `regression.points` is the sorted observed `(x, y)` map.
The derivative table still comes from R's pre-reset regression-point construction,
which NNS Python matches rather than recomputing adjacent slopes from all observations.

The `"mode"` and `"mode_class"` noise-reduction modes are accepted in the
univariate path and use the shared `nns_part`/`nns_mode` implementation. The
`"mode_class"` default-order path can produce segment `standard.errors` values
that differ from R at floating grouping granularity: installed R groups the
`gradient` column through data.table's numeric radix grouping, while NumPy keeps
near-identical binary floating values as separate groups. Regression points,
coefficients, fitted values, and point estimates still match R on that path.

Regression confidence intervals are deterministic and use R's `LPM.VaR` /
`UPM.VaR` logic, not `nns_mc` / `nns_meboot`. In the univariate fitted table,
both `conf.int.pos` and `conf.int.neg` use `UPM.VaR(..., degree = 1)` on
segment residuals, matching installed R even though the lower side might look
like an `LPM` candidate. Univariate `point_est` prediction intervals use
`UPM.VaR(..., degree = 0)` for the upper column and `LPM.VaR(..., degree = 0)`
for the lower column. Below-range univariate point estimates follow R's
`findInterval`/data.table behavior: index `0` rows are dropped, so `pred.int`
can have fewer rows than `Point.est`. For class mode, fitted confidence columns
remain raw numeric values, while univariate `pred.int` columns are rounded with
R's `x %% 1 < 0.5` rule. Spline-eligible `smooth=True` interval tables use the
same deterministic residual VaR logic after smoothing, matching installed R.

## Multivariate Regression

`nns_m_reg` maps to installed R's numeric `NNS.M.reg` path with
`factor.2.dummy = FALSE` and plotting disabled.
Outputs use R's keys (`R2`, `rhs.partitions`, `RPM`, `Point.est`, `pred.int`,
and `Fitted.xy`) with data.table objects represented as dictionaries of NumPy
arrays. Numeric and class confidence intervals are deterministic and use the
global residual `UPM.VaR(..., degree = 1)` offset from installed R. In class
mode, fitted predictions and point estimates are rounded/clamped to class codes,
but `pred.int` lower/upper bounds and fitted confidence columns remain raw
numeric values. Classification mode (`type="class"`) is supported for
numeric/logical/factor-like targets and returns numeric class codes. Direct
`nns_m_reg(..., factor_2_dummy=True)` remains rejected for raw factor
predictors because installed R errors on that path. This is an intentional API
boundary rather than a mathematical gap: `nns_m_reg` is the numeric
multivariate engine, while `prepare_factor_predictors(...)` performs the
R-compatible categorical design-matrix preparation. Public `nns_reg` factor
predictor expansion is also supported with `factor_2_dummy=True` and explicit
`factor_levels=` metadata; it combines training `x` and `point_est` before
full-rank dummy expansion, matching installed R's `factor_2_dummy_FR` path.

Point estimates match installed R, including the one-row outsider behavior in
the multi-point path where R drops matrix dimensions before extrapolating.
`order="max"` follows R's convention of using the original regressor matrix as
the regression-point matrix and defaulting `n.best` to 1.

## Stack

`nns_stack` maps to R's numeric and deterministic classification `NNS.stack`
paths using the real `nns_reg` dimension-reduction and multivariate-regression
internals. `type="class"` is supported for numeric/logical/factor-like targets
and returns numeric class codes, not labels. Use `class_levels=` to reproduce R
factor level ordering. Raw string labels remain rejected unless explicit levels
are supplied. `balance=True` is supported for classification and follows R's
`downSample` + `upSample` structure: each non-empty class is downsampled to the
minority count without replacement, each class is upsampled to the majority
count with replacement, and the downsampled rows are concatenated before the
upsampled rows. Exact sampled-row parity with R is not expected because NNS Python
uses NumPy's RNG; `random_seed` is a NNS Python-only reproducibility convenience.
Numeric and class prediction intervals are supported and are combined by
installed R's weighted data.table arithmetic. For class stacks, single-method
`method=1` and `method=2` return the delegated interval table unchanged; when
`method=(1,2)`, the weighted final interval table is rounded with R's
`x %% 1 < 0.5` rule.
`ts_test` is supported and follows installed R's split exactly: CV training uses
the tail `ts_test` rows, while CV testing uses the earlier rows
`1:(n - ts_test)`. This is intentionally not changed even though it is
counterintuitive. R's `CV.size = NULL` samples a random value between 0.2 and
1/3; NNS Python uses a deterministic default of `0.25`. Pass `cv_size` explicitly for
exact R parity.

The installed-R 12.1 Iris classification vignette with `folds=1` is a documented
stack disparity rather than a NNS Python correctness target. On the `141:150` holdout,
the true labels are all class code `3`. Installed R 12.1 returns stack class code
`2` for every row because its learned class-rounding threshold is about `0.60`;
NNS Python returns class code `3` for every row because its learned threshold is about
`0.29`. Both implementations have the same high-level shape in that case
(`reg = 2`, `dim.red = 3`, raw combined stack near `2.5`), but the final
threshold rounding differs. Since R default `folds=5` also returns class code
`3`, NNS Python keeps the behavior that matches the practical classification result
instead of forcing installed-R-12.1 `folds=1` parity.

Factor predictor expansion is supported for `nns_stack(method=1)` and
`nns_stack(method=2)` with explicit `factor_levels=` metadata. NNS Python expands
training and test predictors together using the same full-rank dummy convention
as installed R's aligned train/test builder. Pure factor-predictor `method=2`
and `method=(1,2)` match installed R's fallback to method 1. Mixed
factor/numeric `method=2` uses the expanded numeric design directly. Mixed
factor/numeric `method=(1,2)` is supported for parity-covered cases that use
explicit `factor_levels` expansion.

## Boost

`nns_boost` maps to R's numeric and deterministic classification `NNS.boost`
paths and uses the real `nns_reg` and `nns_stack` implementations. The
small-feature path (`n_features <= 10`, where R evaluates all feature
combinations) is supported. For `n_features > 10`, NNS Python follows R's stochastic
epoch structure: it samples learner-trial feature sets, builds a weighted
survivor feature pool, then samples epoch feature counts and survivor features
from that pool. Exact sampled-feature parity with R is not expected because
NNS Python uses NumPy's RNG, and `random_seed` is NNS Python-only. Installed R errors for
`threshold=` on this path because the threshold short-circuit leaves
`test.features` undefined, so NNS Python keeps that guard. `ts_test` is supported on
the stochastic path and follows R's separate epoch holdout split: initial
learner trials test rows `1:(n - ts_test)`, while epochs test the final
`2 * ts_test + 1` rows. `type="class"` returns numeric class codes, not labels; use
`class_levels=` to reproduce R factor level ordering. Raw string labels remain
rejected unless explicit levels are supplied. `balance=True` is supported for
classification and uses the same R-style `downSample` + `upSample` structure as
`nns_stack`; exact sampled-row parity with R is not expected.
Explicit-level factor predictors are supported through `factor_levels=`. NNS Python
integer-codes those columns before deterministic feature selection, matching
installed R's `data.matrix` conversion under NNS Python' positional-column
convention. Pass `None` for numeric columns in mixed predictor matrices, for
example `factor_levels=(["low", "mid", "high"], None)`. Multiple explicit-level
factor predictor columns use positional `X1`, `X2`, ... semantics; installed R
data frames with semantic column names sort columns alphabetically before
fitting, so callers should order NNS Python columns explicitly when reproducing those
named-data-frame cases. Numeric `pred_int` is supported and
delegates to `nns_stack(pred_int=...)`, matching installed R; it is deterministic
and does not use MC/meboot. `features_only=True` returns before the final stack
fit and ignores `pred_int`, matching R. Classification `pred_int` is supported
and delegates to final stack `method=1`, so interval bounds remain raw numeric
values. `ts_test` is supported for deterministic and stochastic boost paths. R
requires usable column names for matrix inputs; NNS Python uses positional numeric columns. As with `nns_stack`, R
samples a random CV size when `CV.size = NULL`; NNS Python uses deterministic
`cv_size=0.25` unless specified. For classification boost, final predictions,
feature weights, and feature frequencies are parity-tested against installed R
when balance is disabled and structurally tested when balance sampling is
enabled. The public `n.best` value is structural-only because R's final internal
`NNS.stack` call samples its own `CV.size = NULL` split, while NNS Python keeps the
deterministic stack default.

The installed-R 12.1 Iris boost vignette remains a true parity gap, but not a
quality target for exact output matching. On the same all-class-`3` holdout,
installed R 12.1 balanced boost returns class code `1` for every row, while NNS Python
balanced boost returns class code `2` for every row; both are wrong for that
example. Installed R 12.1 also does not accept the `folds` argument shown in the
rendered upstream overview for `NNS.boost`, so this example is tracked as
R-version/upstream-example drift plus a boost parity gap rather than evidence
that NNS Python should copy the installed-R balanced output.

## Seasonality

`nns_seas` maps to installed R's non-plotting `NNS.seas` path and ignores
`plot`, consistent with other NNS Python ports. Inputs shorter than five observations
return R's sentinel period `0`. For mean-zero data, R falls back from coefficient
of variation to `abs(acf1) ** -1`; NNS Python follows the same fallback and
non-finite handling. Installed R can report harmonics rather than the visually
obvious period, so NNS Python matches R's candidate-period screening instead of a
textbook seasonality heuristic. Results are cached by input content and modulo
arguments with defensive copies on return; this preserves R semantics while
avoiding repeated reverse-step scans for identical series.

## ARMA

`nns_arma` maps to R's installed `NNS.ARMA` forecast path. Without prediction
intervals it returns a NumPy forecast vector of length `h`; with `pred_int` it
returns a dict keyed like R's data.table columns (`Estimates`,
`Lower <percent>% pred.int`, `Upper <percent>% pred.int`). Forecasts are
recursive: each estimate is appended before the next horizon step. Plot
arguments are ignored. Prediction intervals use `nns_mc` / `nns_meboot`; exact
stochastic parity with R is not expected because RNG streams differ.
`random_seed` is a NNS Python-only convenience for reproducible interval tests.
No-`pred_int` deterministic forecasts are parity-tested except where NNS Python
intentionally uses a more direct seasonal-lag weighting convention.
`seasonal_factor=True` uses only the first detected period from `nns_seas`,
matching `ARMA.seas.weighting(TRUE, ...)`; `seasonal_factor=False` uses the
selected `best_periods` rows. `dynamic=True` with numeric seasonal factors
raises with R's static-seasonality error. Constant-series behavior follows
installed R, including zero forecasts for automatic seasonality paths and `NaN`
forecasts for some explicit numeric-lag paths. Character `weights` with numeric
multi-lag seasonal factors is rejected because installed R errors during numeric
multiplication on that path.

For explicit numeric multi-lag seasonal factors such as
`seasonal_factor=[132, 276]`, NNS Python intentionally weights each candidate lag by
the coefficient of variation of that actual lag's reverse component series.
Installed R NNS instead computes the coefficient-of-variation term with reverse
steps `1:length(seasonal.factor)` while still applying the observation penalty
to the actual lag values. NNS Python keeps the actual-lag weighting because it better
matches the documented idea that each supplied seasonal factor is weighted by
its own seasonality strength and observation count. The R-compatible difference
is covered by a strict xfail practical test rather than hidden.

`nns_arma_optim` is supported for the installed-R optimizer path. It greedily
selects seasonal factors, evaluates the default co-moment-normalized objective,
then applies the same equal-weight, bias-shift, shrink, and smooth-regressed
variable checks as R. The optimizer's prediction intervals are deterministic
VaR bands around the in-sample optimizer errors; they are separate from
`nns_arma(pred_int=...)`, which uses the Monte Carlo path. Custom Python
`obj_fn` callables may be supplied, but R expression objects are not part of the
Python API. `nns_var` is implemented for numeric matrix-like inputs with
`dim_red_method="cor"`, `dim_red_method="NNS.dep"`,
`dim_red_method="NNS.caus"`, and `dim_red_method="all"` and returns
R-compatible public output keys. VAR's internal multivariate stack stage uses
`ceil(0.2 * n)` for the time-series validation window when that term exceeds
`2 * h`, matching installed R's effective trailing holdout size and preserving
the documented `ts.test` idea as a count of held-out observations. The `h == 0`
path is normalized to a Python dictionary containing `interpolated_and_extrapolated`
and `names` rather than R's bare data-frame return. The first-stage interpolation/extrapolation helper
`_var_interpolate_and_extrapolate` is implemented to match R's
missing-value handling and per-variable `NNS.ARMA.optim` forecasts. The private
multivariate stage `_var_multivariate_stack_stage` is implemented with
`lag.mtx` reconstruction, `NNS.stack(method=(1,2), ts.test, dim.red.method)`
logic, and R-style relevance extraction. The function returns `multivariate`
and `relevant_variables` in the same shape/naming pattern expected by
`NNS.VAR`.

`nns_nowcast_panel` is the deterministic nowcast core for user-supplied monthly
numeric panels. It accepts array-like panels or ordered mappings of column names
to numeric series, delegates numeric forecasting to `nns_var`, and returns VAR
fields plus `dates` and `metadata` dictionaries. Date labels are metadata rather
than array indices. Without dates, forecast rows are labeled `t+1`, `t+2`, ...
With dates, inputs are normalized to `YYYY-MM`, must be sorted and unique, and
forecast labels advance monthly. R NNS 12.1 beta removed `NNS.nowcast`, so NNS Python
does not export a public `nns_nowcast` wrapper. `CsvNowcastProvider` remains an
explicit payload builder whose `fetch(series, start_date)` method returns
`{"series": ..., "dates": ..., "metadata": ...}` for callers to pass to
`nns_nowcast_panel`. `CsvNowcastProvider` is offline and local-file only.
Library code does not read `.env` files. NNS Python does not ship an implicit
FRED/Yahoo provider.

## Meboot

`nns_meboot` maps to R's `NNS.meboot` maximum-entropy bootstrap algorithm and
returns plain Python dictionaries instead of R's vectorized list-matrix wrapper.
Scalar `rho` returns one result dictionary; vector `rho` returns a list of result
dictionaries in R's vectorized order. `rho=None` follows installed R's empty
output behavior, and length-one input returns only `{"x": x}`.

Exact replicate parity with R is not expected because NNS Python uses NumPy's random
number generator and SciPy's optimizer while R uses its global RNG and
`optim()`. Deterministic diagnostics (`xx`, `z`, `dv`, `dvtrim`, `xmin`,
`xmax`, `desintxb`, `ordxx`, and `kappa`) are parity-tested exactly. Stochastic
outputs are tested structurally and statistically. `random_seed` is a NNS Python-only
convenience for reproducible bootstrap draws.

## Monte Carlo

`nns_mc` maps to R's `NNS.MC` wrapper around `NNS.meboot`. The rho grid and
exponential rho transformation are parity-tested exactly against installed R.
As with `nns_meboot`, exact stochastic replicate parity is not expected because
R and NNS Python use different RNG streams and optimizer implementations.
`random_seed` is a NNS Python-only convenience passed through to `nns_meboot`.

NNS Python returns `{"ensemble": array, "replicates": dict}`. The `replicates`
mapping preserves R's names, such as `"rho = 1"` and `"rho = -0.5"`, with each
value containing that rho block's replicate matrix. Sampling-vignette examples
are covered as smoke tests, but installed R behavior remains the parity source.

## Normalization

`nns_norm(x, linear=False)` maps to R's numeric matrix `NNS.norm` path with
plotting disabled. NNS Python accepts finite 2D arrays. `linear=True` uses R's
mean-ratio scaling, while `linear=False` additionally weights scaling by
absolute correlation for fewer than 10 columns and NNS dependence for 10 or
more columns.

## Distance

`nns_distance` and `nns_distance_bulk` map to R's regression-point-matrix
helpers. NNS Python accepts `rpm` as a finite 2D numeric array with R's `y.hat`
column in the final position. `nns_distance` applies R's per-target min-max
rescaling before computing weighted nearest-neighbor predictions. `nns_distance_bulk`
matches R's compiled bulk helper, including its raw-feature distance convention.
For `nns_distance` with `k > 1`, NNS Python matches the installed R 12.1 binary:
the exponential rank-weight family uses the R C API's `Rf_dexp` scale argument
as `1 / k`. This differs from the nearby source-code comment that describes it
as a rate.

Classification distance mode returns numeric class codes, not original labels.
For single-target `nns_distance(..., class_=...)`, installed R uses weighted
mode with integer replication counts `ceil(100 * weight)`. NNS Python follows that
behavior. For equal-distance nearest-neighbor ties, NNS Python preserves RPM row order
to match installed R's first-row tie behavior. Installed R's
`NNS.distance.bulk(..., class=...)` currently ignores
the class flag in its compiled bulk helper and returns the same inverse-distance
numeric weighted average as non-class bulk distance; NNS Python matches the installed
binary rather than the higher-level classification intent.

## Classification

R classification paths work with numeric class codes. R factors become
1-indexed numeric codes in factor-level order and predictions are returned as
codes rather than decoded labels. NNS Python provides `factor_2_dummy`,
`factor_2_dummy_fr`, `encode_factor_codes`, and `prepare_factor_predictors`;
pass explicit `levels=` / `factor_levels=` to reproduce R factor level order
because NumPy arrays do not carry factor metadata.

`nns_reg(..., type="class")`, `nns_m_reg(..., type="class")`, and
`nns_stack(..., type="class")` are supported for numeric, logical, and
factor-like targets. Use `class_levels=` when passing string/object labels so
NNS Python can reproduce R factor codes explicitly. Raw string classification remains
rejected where installed R errors or produces unusable `NA` conversions.
Predictions and point estimates are numeric class codes, not original labels,
matching installed R. Class confidence intervals are supported in `nns_reg` and
`nns_m_reg`; stack/boost class `pred_int` is supported through those regression
interval tables.

## Differentiation

`nns_diff` maps to R's scalar callable `NNS.diff` path with plotting and trace
output disabled. It returns a dictionary keyed by R's matrix row names and
rounds results to `digits`, matching R's default output convention.
`dy_dx(..., eval_point="overall")` maps to R's `dy.dx(..., eval.point =
"overall")` path and returns the mean fitted gradient from unsmoothed
`nns_reg`. Numeric `dy_dx` evaluation points use R's finite-difference grid
around smooth `nns_reg` point estimates and return a table-like dictionary with
`eval.point`, `first.derivative`, and `second.derivative`. Boundary-point
quirks follow installed R where covered by parity tests.

NNS Python derivative parity is defined at the public input/output level, while
preserving R's cumulative finite-difference perturbation pattern for `dy_d`.
`dy_d` scalar `wrt` has enforced R parity for `eval_points="mean"`, `"median"`,
`"last"`, `"obs"`, and `"apd"`. Vectorized `wrt` returns one row per eval point
and one column per requested regressor for `First`, `Second`, and `Mixed` when
mixed derivatives are defined. Treat `dy_d` as an NNS finite-difference
sensitivity estimate around `nns_reg` point estimates, not as an exact analytic
calculus derivative.

Mixed derivatives require a two-regressor input. Numeric two-value evaluation
points and single-row point modes match installed R on focused fixtures. For
multi-row matrix evaluation points, including `eval_points="obs"`, NNS Python uses a
pointwise mixed finite-difference construction. Installed R's vectorized
list-matrix path packs multi-row mixed derivative points in an order-dependent
way, so NNS Python does not copy that packing quirk.

For scalar `dy_d`, R mutates lower and upper finite-difference points
cumulatively across rounded bandwidths. If rounded bandwidths repeat, R writes
the later cumulative result back to the first matching result slot and drops
the empty slots during final weighted averaging; NNS Python mirrors that behavior.
The `obs` and `apd` paths also rely on smooth dimensional-reduction
`nns_reg(..., point_est=..., dim_red_method="equal", smooth=True)` estimates.
For out-of-range smooth point estimates, R derives extrapolation slopes from
the smoothed regression points before clamping returned regression-point `y`
values, then anchors the extrapolation at the first `which.min` / `which.max`
boundary row. NNS Python mirrors those boundary quirks for parity.

## ANOVA

`nns_anova` maps to R's non-plotting `NNS.ANOVA` paths. Binary comparisons
return a dictionary keyed like R's list output, aggregate multi-group
comparisons return `{"Certainty": value}`, and `pairwise=True` returns R's
symmetric certainty matrix. Confidence interval bootstrapping is structurally
identical to R but uses NumPy RNG instead of R's `sample()`, so exact per-call
parity is not achievable; numeric values converge to the same population CI.
Pass `random_seed` for reproducible NNS Python results. Degenerate zero-variance
groups preserve R's `NaN` CDF/certainty convention.
