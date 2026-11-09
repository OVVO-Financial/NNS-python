# Conventions

## Build

NNS Python is a Python-native NumPy/SciPy port with optional private native
acceleration through `nns._nnscore` where available. Source builds use
`scikit-build-core` and `nanobind` to compile the extension; published wheels
should be preferred when available. Public APIs keep Python implementations and
explicit fallback behavior, so native code remains a deliberate,
benchmark-backed implementation detail rather than a public API. Setting the
environment variable `NNS_DISABLE_NATIVE=1` before importing `nns` forces the
NumPy fallbacks, as when the extension is not built; CI runs the invariant and
cached parity tests that way too.

## Threads

The native kernels (vectorised partial moments, the PM matrix, the n-dimensional
co-moments and the copula) can use several threads. Inputs below a fixed amount of
work always run serially, and results never depend on the thread count.
`nns.set_num_threads(n)` sets the default; `n=None` restores the automatic choice,
which is the `NNS_NUM_THREADS` environment variable when it is a positive integer
and the number of hardware threads otherwise. `nns.get_num_threads()` returns the
current default. The `ncores` argument of `nns_reg`, `nns_m_reg`, `nns_stack`,
`nns_boost`, `nns_arma_optim` and `nns_var` caps the native threads for that call:
`None` keeps the default, `1` runs serially, and anything other than `None` or a
positive integer raises `ValueError`, as in R's `NNS.M.reg`. The cap applies per
Python thread. R also uses `ncores` for its parallel `NNS.ARMA.optim` workers;
NNS Python has no process-level parallelism. The heavy kernels release the GIL,
so Python threads can run them concurrently.

## Plotting

Compute functions never draw unless asked. Where R's default is `plot = TRUE`
(`NNS.reg`, `NNS.FSD`/`SSD`/`TSD`, `NNS.ANOVA`, ...), NNS Python defaults to
`plot=False` and returns the same values either way; `plot=True` additionally
draws the figure through the matching `nns.plotting.plot_*` function and never
calls `plt.show()`. The flag exists on `nns_reg`, `nns_m_reg`, `nns_cdf`,
`nns_seas`, `nns_arma`, `nns_arma_optim`, `nns_part` (R's `Voronoi`),
`fsd`, `ssd`, `tsd`, `nns_anova`, `nns_norm` (R's `chart.type = "l"`),
`nns_diff` and `nns_copula`. `nns_sd_cluster(dendrogram=True)` draws the
dendrogram, as R does, because the flag is also what returns the `hclust` data.
A requested figure that cannot be drawn raises a `UserWarning` instead of being
skipped silently: multivariate `nns_reg(plot=True)` with more than two
regressors (R draws its 3-D scatter for exactly two; `residual_plot=True` draws
the fitted-values plot for any number), `nns_copula(plot=True)` with more than
three variables, and `nns_norm(plot=True)` on unequal-length lists.

## Degree-Zero Boundary

At degree zero, `LPM` uses `x <= T` and `UPM` uses `x > T`.
Equality is counted by `LPM` only.

For any non-empty finite input, `LPM + UPM = 1` at degree zero.

## Empty Input Divergence From R

R NNS returns `NaN` for empty input.
NNS Python raises `ValueError`.

Rationale: empty arrays in Python are upstream bugs, and NumPy convention is to warn or fail on empty reductions rather than silently produce a meaningful statistic.

## Co-Moment Lengths

`co_lpm`, `co_upm`, `d_lpm` and `d_upm` raise `ValueError` unless `x` and `y` are
non-empty and the same length, and `target_x` and `target_y` are non-empty and the
same length or one of them length 1. R's `Co.LPM`, `Co.UPM`, `D.LPM` and `D.UPM`
check the same rules.

## PM Matrix Target Defaults

R `PM.matrix` uses column means when `target` is `NULL` or any non-numeric value.
NNS Python accepts `None` and `"mean"` for this behavior. NNS Python also broadcasts a
scalar numeric target across all variables; R requires callers to pass an
explicit vector such as `rep(0, ncol(variable))`. Target vectors whose length
does not match the number of variables raise `ValueError`.

## Mode Default Divergence From R

`nns_mode` defaults to `multi=False` and returns a single float, while R's
`NNS.mode` defaults to `multi = TRUE` and returns every detected mode (e.g. two
values for a bimodal sample). Pass `multi=True` for R's default output; explicit
`multi` values match R. The Python default is kept because the multi-mode peak
finder also reports spurious secondary modes on unimodal data (in R as well),
and the internal callers all ask for a single mode.

## Classical Moment Normalization

`mean_pm`, `var_pm`, `skew_pm`, and `kurt_pm` use population normalization by
default, matching NumPy defaults and `NNS.moments(population = TRUE)`. `var_pm`
accepts `ddof` for NumPy-style variance scaling. `skew_pm` and `kurt_pm` do not
apply SciPy's optional finite-sample bias correction.

`nns_moments` is the public `NNS.moments` wrapper and returns R's dictionary
shape with `mean`, `variance`, `skewness`, and `kurtosis`. As in R, undefined
moments are `NaN` rather than errors (constant input; `population=False` with
n <= 3). `nns_gravity` exposes
R's public `NNS.gravity` central-tendency helper. `fsd_uni`, `ssd_uni`, and
`tsd_uni` are the unidirectional stochastic-dominance wrappers behind R's
`.uni` exports. `co_lpm_nd`, `co_upm_nd`, and `dpm_nd` expose the public
n-dimensional partial-moment wrappers.

`nns_ss` maps to R's `NNS.SS` stochastic-superiority function, not to the
stochastic-dominance tests. It returns `p_gt = P(X > Y)`, `p_tie = P(X = Y)`,
and `p_star = p_gt + 0.5 * p_tie`. `NaN` values are omitted independently from
`x` and `y`, matching R's `na.omit` preprocessing. With
`confidence_interval=True`, intervals are computed through `nns_meboot`,
`lpm_var`, and `upm_var`: with `random_seed = s` the bounds and `boot_vals` are
R's after `set.seed(s)`, the replicates of `x` and then of `y` drawn from one
stream as R draws them.

`nns_sd_cluster` maps to R's `NNS.SD.cluster` default path. It iteratively
peels `sd_efficient_set` results and returns a dictionary of `Cluster_1`,
`Cluster_2`, ... memberships. The output contains variable names, not numeric
cluster labels; when names are omitted, NNS Python uses R-style `X_1`, `X_2`, ...
names, or the column names of a DataFrame input. `type="continuous"` is
supported for first-degree efficient sets, `fsd`, and `fsd_uni`; as in R, an
unrecognised `type` warns (and `fsd` then runs its continuous branch, while
`fsd_uni` and the efficient-set routines run the discrete one).
`sd_efficient_set` returns 0-based column indices in R's output order (R
returns the column names), and a vector is one column, as with R's `as.matrix`.
Inputs are checked in R's order and with R's messages: missing values raise
"You have some missing values, please address.", empty samples "Each sample
needs at least one observation.". Like R, the `.uni` tests, `sd_efficient_set`
and `nns_sd_cluster` accept infinite values and compare them as R's C++ test
does, while `fsd`, `ssd` and `tsd` reject them (R first evaluates the plotted
partial-moment curves, which check `x` and then the threshold grid).
`nns_sd_cluster` validates its data in the first efficient-set search, as R
does, so a matrix with no more than `min_cluster` columns comes back as one
cluster unchecked, and `min_cluster` is compared as a real number (`1.5` merges
a trailing single-member cluster into the previous one).
`dendrogram=True` returns a plain dictionary mirroring R's `hclust` fields:
`merge`, `height`, `order`, `labels`, `method`, `call`, and `dist.method`.
With fewer than two clustered variables it warns, as R does, and returns the
clusters only. The dendrogram is drawn as well (see Plotting).

The stochastic-dominance implementation is deliberately pure NumPy. It mirrors
R's C++ SD core mathematically by sorting each column once, storing prefix sums,
and evaluating dominance on each pair's merged threshold grid rather than on one
global all-column grid. Pairwise tests (`fsd`, `ssd`, `tsd`, and the `*_uni`
wrappers) accept samples of unequal length: each sample's curve is evaluated on
the merged grid exactly as R's `NNS.FSD`/`NNS.SSD`/`NNS.TSD` compute
`LPM(degree, sort(c(x, y)), sample)` per sample, and as the patched R `.uni`
walkers do (each sample walked with its own length). Every pairwise verdict
comes from one port of R's native pairwise test: for TSD and continuous FSD it
also checks the closed-form extremum of the curve difference strictly between
merged sample points (LPM2 is piecewise quadratic and the continuous ratio
linear-fractional, so the difference can peak there), as the patched R does. The
matrix-based efficient-set and cluster routines operate on data columns and
therefore remain equal-length by construction. Every efficient-set and cluster
call, whatever the column count, tests each pair on that pair's own grid, so a
pair's verdict never depends on which other columns are present. Degree-1
discrete calls use an exact order-statistic dominance matrix: with equal-length
empirical samples, one sample first-order stochastically dominates another
exactly when every sorted order statistic is at least as large and at least one
is strictly larger. Degree-1 continuous and degree 2/3 calls use a lazy
kept-only prefix scan.
Columns are visited in R's LPM-at-global-maximum order with original-index tie
breaks, and only already-kept candidates are tested against the current column.
Each prefix pair check applies min/mean/identical guards before evaluating
curves.

Curve differences up to `64 * eps * S^d` (`S` the largest absolute value of
the two samples; `d` = 1 for SSD and the mean condition, 2 for TSD, a ratio
analogue for continuous FSD) are treated as ties, as in the patched R, so
rounding noise does not decide dominance; discrete FSD compares exact
empirical CDF fractions. No output reordering or diagonal/identical-column
behavior changes are introduced. Polars is intentionally not used in this SD kernel
because the hot path is dense pairwise threshold evaluation rather than
data-frame grouping or filtering. R remains faster on some large finance
fixtures because its C++ path walks merged sorted thresholds in tight parallel
loops with minimal temporaries; NNS Python instead uses NumPy order-statistic blocks,
`searchsorted`, contiguous column storage, and early-exit scans to stay
dependency-light and pure Python.

`nns_cdf` maps to R's `NNS.CDF` deterministic non-plotting paths. It is a
partial-moment distribution wrapper rather than a textbook ECDF: `degree = 0`
uses R's lower-partial-moment frequency convention, and positive degrees use
`LPM.ratio` deformation. Univariate output columns follow installed R (`x` plus
`CDF`, `S(x)`, `h(x)`, or `H(x)`), while multivariate output keeps the final
column named `CDF` for all types, including survival, hazard, and cumulative
hazard. Input column names that repeat each other or `CDF` get R
`make.unique` suffixes (`CDF.1`, `a.1`), since a dict cannot hold the duplicate
names R returns. Plotting is ignored. As in the patched R (see [parity](parity.md)), missing
values are dropped first (complete rows for a matrix), so the CDF reaches 1, and
infinite values raise `ValueError`. For `d >= 3` columns the survival, hazard and
cumulative hazard use the joint survival `P(X_1 > t_1, ..., X_d > t_d)` by
inclusion-exclusion over all column subsets.

## Dependence

`nns_dep(x, y)` follows R's `NNS.dep` bivariate path. With `asym=True` the partition
is on `x` and the result measures `y` as a function of `x`. `nns_dep(X)` on a 2-D
array or DataFrame returns R's `Correlation` and `Dependence` matrices; with
`asym=True` entry `[i, j]` is `nns_dep(X[:, i], X[:, j], asym=True)` (row variable
-> column variable, which is what R returns although its manual states the
reverse). `p_value=True` runs R's permutation test on the bivariate form: the
measures of `x` against 100 shuffles of `y` give a two-sided `Correlation p.value`,
an upper-tail `Dependence p.value` and the shuffled values' 95% partial-moment
intervals; with `random_seed = s` the shuffles are R's after `set.seed(s)`.
The bivariate path includes `NNS.gravity` handling
for zero-range inputs and non-positive or non-finite bin widths. The gravity
histogram (bin width `IQR / sqrt(n)`) is built from the occupied bins only, so a
single outlier that makes R allocate millions of empty bins costs nothing and gives
R's value. The compiled kernel (NNS-core `nns::gravity`) and the NumPy fallback
compute the same value bit for bit. They differ from R 13.3 only when the bin count
exceeds the C `int` range: R's cast overflows there and the histogram collapses to
one bin, while NNS Python (like the patched R) keeps the real histogram.
`abs(Correlation) <= Dependence` is not guaranteed by
`NNS.dep`; both R and NNS Python can return signed correlation magnitudes above the
dependence component for near-binary inputs.

## Copula

`nns_copula(x, y)` is the bivariate form of R's `NNS.copula(cbind(x, y))`, and
`nns_copula(X)` takes a 2-D matrix with any number of columns (at least two), as R
does; `target` gives the per-column targets of the matrix form. When targets are
omitted NNS Python uses the column means computed like R's `colMeans` (long-double
sequential sum), matching R's `target = NULL` to the last bit. The `target_x` and
`target_y` arguments map to R's two-element target vector. Empty or non-finite
input raises `ValueError` (R stops on missing values).

## Causation

`nns_causation(x, y, tau)` maps to R 14's `NNS.caus(x, y)`: the two directional
indices `C(x--->y)` and `C(y--->x)` in `[0, 1]`, which sum to one and are 0.5 when
neither direction dominates. They measure directional lagged association, not
causal identification. `causal_matrix(X)` (also `nns_causation(X)` with `y=None`,
as in R) is R's `NNS.caus.matrix`: entry `[i, j]` is `C(column i ---> column j)`,
so `[i, j] + [j, i] = 1` off the diagonal, which is 0. A DataFrame input returns
DataFrames labelled with its columns.

`p_value=True` adds R's permutation test, with R's `nperm`, `permute`, `seed` and
`conf_int` (`conf.int`). For a pair, the signed index `C(x--->y) - C(y--->x)` is
compared with its null under `permute="auto"` (a circular shift of `y` at least
`max(tau + 1, n / 10)` steps from alignment for a positive lag, a shuffle of `y`
otherwise) or an explicit `"y"`, `"x"`, `"both"` or `"shift"`; the result is
`{"causation": ..., "p.value": {...}}` with R's fields. For a matrix,
`"auto"` shifts the second column of each pair for time series and shuffles every
column otherwise, and the result holds R's `causality`, `lower_CI`, `upper_CI`
and `p.value_matrix`. Every permutation is scored at the lag resolved on the
observed series, and with `seed = s` the draws are R's after `set.seed(s)`.
`tau='ts'` uses `nns_seas(... )["periods"]` exactly like installed R: the first
period not exceeding `sqrt(length(x))` is selected per variable, including
harmonics when R selects them. Inputs with no eligible selected period follow
R's failure convention and raise. Numeric `tau` lag values remain fully
supported.

## Partition

`nns_part` maps to R's `NNS.part` but returns plain NumPy arrays instead of
`data.table` objects: `"dt"` and `"regression.points"` are dictionaries of
arrays. As in R 13.3, `type` must be `None` (XY quadrant splits) or `"XONLY"`
(any case); other values raise `ValueError`. `order` is `None`, a positive
integer (NumPy integers and integral floats are accepted), or `"max"`, which
returns R's explicit maximum partition (one point per observation, or per unique
`x` for `XONLY`). `obs_req` must be a non-negative integer. The returned `order`
is the depth actually reached; it never exceeds `floor(log2(n))` and can be smaller
than requested. Pairs with `NaN` in either variable are dropped and infinities are
kept, as in R. All five `noise_reduction` modes are supported: `"off"`, `"mean"`,
`"median"`, `"mode"`, and `"mode_class"`.

## Regression

`nns_reg` maps to R's `NNS.reg` (plotting is a separate side effect). Return
keys match R's list names; data.table outputs are plain dictionaries of NumPy
arrays. `multivariate_call=True` returns R's internal two-column
regression-point structure as `{"x": ..., "y": ...}`. More than one encoded
predictor without dimension reduction runs the `NNS.M.reg` branch.

Inputs follow R's data-frame rules (`.nns_reg_encode_predictors`):

- pandas column names are kept as they are (R's `check.names = FALSE`; only
  duplicates are renamed, by `make.unique`) as keys in `RPM`, `Fitted.xy` and
  `equation`; factor dummy columns are `<column>_<make.names(level)>`. And `point_est` columns are matched to the training
  columns by name: the same set of names is reordered into training order,
  names disjoint from the training names (or an unnamed array) are taken
  positionally, and a partial overlap raises `ValueError`. A named row such as
  `df.iloc[0]` is matched the same way. A single predictor is positional.
- A column is a factor when it is a pandas categorical, when `factor_levels=`
  declares its levels, or when it holds labels: strings (including
  numeric-looking strings such as `"1"`, `"10"`, as in R) or an object column
  with any non-number. An object column of numbers (`df.to_numpy()` of a mixed
  frame) is numeric. `bool` columns (and object columns of booleans) are
  factors with levels `FALSE`/`TRUE`, as R treats logicals: dummies
  `<column>_FALSE.` / `<column>_TRUE.` (R's `make.names`), so even a single
  bool predictor runs the `NNS.M.reg` branch.
- Factor levels are the declared levels in their order (pandas categories, or
  `factor_levels=`: a mapping from column name to levels, one entry per column
  with `None` for numeric columns, or a flat level list for a single
  predictor). Undeclared label columns use sorted levels. Divergence: R uses
  first-appearance order for character predictors, which makes its fit depend
  on the row order; sorted is R's `factor()` order. With the default
  `factor_2_dummy=True` the order only permutes the one-hot columns (one per
  level); with `factor_2_dummy=False` the level order is the integer code.
- Missing values in a factor column (training or `point_est`) raise
  `ValueError`, as in R; `point_est` levels unseen in training raise too.
- The response is a factor when it is a pandas categorical or `class_levels=`
  is given (codes `1..K` in that order, `class.levels` lists them, and
  classification is implied), a set of labels when it holds strings (sorted
  levels, R's `factor(y)`), and numeric otherwise; an object array of numbers
  is a numeric response, not classification. Missing labels raise.
  Classification returns numeric class codes with `class.levels` as the label
  map.

`dist=None` is the default for `nns_reg`, `nns_m_reg`, `nns_stack`, and `nns_boost` and selects the native blended NNS distance, with `dist="NNS"` as the explicit alias. Continuous predictors use range-normalized coordinate differences and sum `abs(z) + z**2` across non-zero-range columns; explicit `dist="L1"`, `dist="L2"`, and `dist="FACTOR"` keep their legacy behavior.
Multivariate predictions follow R's C++ kernel to the last bit, in the native
extension and the NumPy fallback alike: the ensemble weights use R's own nmath
`dt` and `dnorm` densities and libm `exp`, summed in R's order. Classification
probabilities follow the R-level weights R's classifier uses instead (long-double
`sum`/`var`, `rowSums` distances). `nns_m_reg(categorical=...)` is the argument
of R's internal `NNS.M.reg` (default `False`): exact joint cells, maximum order
and one neighbour; `nns_reg` sets it when every predictor is categorical, as R's
`NNS.reg` does.
`smooth=True` follows installed R's ordinary piecewise fallback for univariate
inputs with fewer than four observations and for univariate `order="max"`; R
does not call `smooth.spline` there. Spline-eligible inputs use a private
fixed-`spar` cubic smoothing-spline adapter matching the `stats::smooth.spline`
subset used by `NNS.reg`: `spar = (dependence + 0.5) / 2`, R's
`1e-6 * IQR(x)` tie grouping, `.nknots.smspl` knot thinning above 49 unique x,
the Fortran `sgram` penalty (with its `.3330` constant), R's interior-band
trace ratio for lambda, and a banded Cholesky solve.
With dimension reduction, factor predictors are encoded first (one dummy column
per level with `factor_2_dummy=True`) and the synthetic `x.star` coefficients
are computed on the encoded columns, as in R. For callers that want an explicit
numeric design matrix, `prepare_factor_predictors(...)` returns the same
full-rank dummy expansion:

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

`prepare_factor_predictors(...)` combines training `x` and `point_est` before
expansion and returns deterministic feature names.

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

`noise_reduction` accepts R's `"mean"`, `"median"`, `"mode"` and `"off"`;
classification always reduces by the class mode.

Regression confidence intervals are deterministic residual bands: the
`alpha/2` and `1 - alpha/2` type-8 quantiles of the training residuals are
added to the fitted values (`conf.int.neg` / `conf.int.pos` in `Fitted.xy`)
and to each point estimate (`pred.int.neg` / `pred.int.pos`, one row per
`point_est` row, including points outside the training range). In class mode
`pred.int` bounds are snapped to the observed class codes; the fitted
confidence columns stay raw.

## Multivariate Regression

`nns_m_reg` maps to R 13.3's `NNS.M.reg`, including its defaults
(`factor_2_dummy=True`). Predictors and the response are prepared exactly as in
`nns_reg` (see Regression): raw string and categorical columns are encoded by
the function itself (one-hot dummies, or integer codes with
`factor_2_dummy=False`), pandas column names are kept as the `RPM`,
`rhs.partitions` and `Fitted.xy` keys (`V1..Vp` for an unnamed matrix), and
`point_est` columns are matched by name. A string, bool or categorical
response, or one given `class_levels=`, is a classification target with
`class.levels` set; `type="class"` forces classification of a numeric
response. At least two encoded predictor columns are required. Outputs use
R's keys (`R2`, `rhs.partitions`, `RPM`, `Point.est`, `pred.int`,
`Fitted.xy`, `n.best`, `dist`, `class.levels`) with data.table objects as
dictionaries of NumPy arrays; intervals are the residual-quantile bands
described under Regression. `prepare_factor_predictors(...)` is a separate
legacy helper that builds a full-rank dummy design; it is not needed for
either function.

Point estimates match installed R, including the one-row outsider behavior in
the multi-point path where R drops matrix dimensions before extrapolating.
`order="max"` follows R's convention of using the original regressor matrix as
the regression-point matrix and defaulting `n.best` to 1.

## Stack

`nns_stack` ports R 13.3's `NNS.stack`: out-of-fold selection of `n.best`
(Method 1) and of the dimension-reduction count (Method 2), classification
thresholds and ensemble weights, using the real `nns_reg` internals.

Randomness: every split and balancing draw reproduces R's Mersenne-Twister
`sample()` stream (`set.seed(seed)`, default `seed=123`), so a seeded run
matches R bit for bit. `seed=None` draws a seed from OS entropy (R continues
its global stream). `random_seed=` is an alias for `seed`.

Splits follow R: `cv_size=None` (R's `CV.size = NULL`) with the default
`folds=5` is stratified (classification) or plain k-fold cross-validation;
`folds=1` with `cv_size=None` draws one holdout size from `runif(0.2, 1/3)`;
an explicit `cv_size` gives `folds` repeated holdouts of that size. `ts_test`
uses rolling chronological folds: each validation block of `ts_test` rows is
predicted from all earlier rows. `balance=True` (classification) down- and
up-samples each fitting split as R's `.balance_indices` does.

Inputs follow R's data-frame rules. pandas column names are kept (an unnamed
matrix is `V1..Vp`, a vector `X1`), and `ivs_test` columns are matched to the
training columns by name (same set reordered, disjoint names positional,
partial overlap an error). Factor predictors are pandas categoricals (all
declared levels, as `levels()`), columns given `factor_levels=` (same forms as
`nns_reg`), or label columns (strings, including numeric-looking strings),
which use sorted levels; each becomes one dummy per level, fitted on the full
training levels, named `<column>__<make.names(level)>`.

Classification (`type="class"`, or automatic for string, bool, categorical
and two-valued numeric responses) returns numeric class codes `1..K`, with
`class.levels` as the label map. Codes follow declared order for a pandas
categorical or `class_levels=` (unused levels dropped, as R's `droplevels`),
sorted order for strings, `False < True` for bool, and sorted values for a
numeric response. The sorted order is R's factor order, which R's `NNS.stack`
and `NNS.boost` use for character responses and predictors.

`NNS.dim.red.variables` lists the names of the predictors carried by the
final dimension reduction, ranked by absolute coefficient, and
`NNS.dim.red.coefficients` is a dict from encoded column name to coefficient
(R's named vector). `NNS.reg.n.best` is `NaN` when the all-points candidate
wins with a single (encoded) predictor, as R's `NA`, and the RPM row count in
the multivariate case.

`nns_stack_fit` / `nns_stack_predict` run the cross-validated selection once and
reuse it for new test sets.

The installed-R 13.0 Iris classification vignette with `folds=1` is a documented
stack disparity rather than a NNS Python correctness target. On the `141:150` holdout,
the true labels are all class code `3`. Installed R 13.0 returns stack class code
`2` for every row because its learned class-rounding threshold is about `0.60`;
NNS Python returns class code `3` for every row because its learned threshold is about
`0.29`. Both implementations have the same high-level shape in that case
(`reg = 2`, `dim.red = 3`, raw combined stack near `2.5`), but the final
threshold rounding differs. Since R default `folds=5` also returns class code
`3`, NNS Python keeps the behavior that matches the practical classification result
instead of forcing installed-R-13.0 `folds=1` parity.

## Boost

`nns_boost` ports R 13.3's `NNS.boost`. Learner trials fit multivariate
`nns_reg` models on feature subsets (all `2^p - 1` subsets when that is at most
`learner_trials`, otherwise distinct random subsets) against fresh holdouts;
`threshold` is the probability passed to `LPM.VaR` over the learner-trial
objective distribution (`0.8` / `0.2` by default, `extreme=True` uses the best
trial); `epochs` (default `2 * n`) re-test the surviving subsets; the final
estimate replicates the keeper features by their relative frequencies into one
`nns_stack(method=1, stack=False)`. `depth` is passed as `order`. All random
draws reproduce R's Mersenne-Twister stream (default `seed=123`), so seeded
runs match R; `cv_size=None` draws the holdout size from `runif(0.2, 1/3)` as R
does.

Inputs are prepared as in `nns_stack`: pandas column names are kept and name
`feature.weights` / `feature.frequency` (`V1..Vp` for an unnamed matrix), test
columns are matched by name, and factor predictors (categoricals,
`factor_levels=`, or label columns with sorted levels) stay factors in every
learner fit; declared levels absent from the training data are dropped
(R's `droplevels`). A string, bool or categorical response is classified and
returns class codes with `class.levels` (declared order, `class_levels=`, or
sorted labels); a numeric response with `type="class"` returns its original
class values.

The installed-R 13.0 Iris boost vignette remains a true parity gap, but not a
quality target for exact output matching. On the same all-class-`3` holdout,
installed R 13.0 balanced boost returns class code `1` for every row, while NNS Python
balanced boost returns class code `2` for every row; both are wrong for that
example.

## Seasonality

`nns_seas` maps to R 14's `NNS.seas`; `plot=True` draws the figure as a side
effect (see Plotting). A period's score is the length-weighted Gini mean
difference of its component series (relative to the component means when the
data have one sign), and periods scoring below the whole series are candidates,
sorted by score. The linear trend is removed first (`detrend=True`), the scan
covers periods up to `min(n / 2, n / 5)` (or `max_period`), `modulo` adds
multiples as in R, and the reported period is the fundamental one: the smallest
plausible divisor of the strongest period within two paired standard errors of
it. Inputs shorter than five observations return R's sentinel period `0`. The
scores follow R's arithmetic and agree with R to the last bit. `p_value=True`
adds R's permutation p-value of each period (`NaN` for period 1): its dispersion
ratio against the best ratio of each of `nperm` shuffles of the detrended series;
`periods` then lists the periods with p-value at most 0.05 (else 1). With
`random_seed = s` the shuffles are R's after `set.seed(s)`.

## ARMA

`nns_arma` maps to R's installed `NNS.ARMA` forecast path. Without prediction
intervals it returns a NumPy forecast vector of length `h`; with `pred_int` it
returns a dict keyed like R's data.table columns (`Estimates`,
`Lower <percent>% pred.int`, `Upper <percent>% pred.int`). Forecasts are
recursive: each estimate is appended before the next horizon step. Plot
arguments are ignored. Prediction intervals use `nns_mc` / `nns_meboot`: with
`random_seed = s` they are R's after `set.seed(s)`, bit for bit. As in R 14, a
one-point or constant forecast path has no dispersion to resample, and its band
is the path widened by the per-lag spread alone. Forecasts, the seasonal lag
weights and `nns_seas` scores use R's arithmetic (long-double `mean`/`sum`, R's
`sd`), so they agree with R to the last bit.
As in R, `seasonal_factor=True` with a non-`None` `best_periods` is treated as
`seasonal_factor=False`: it weights the `best_periods` best periods from
`nns_seas` (the default `best_periods=1` selects the best period).
`seasonal_factor=True, best_periods=None` uses the first detected period
(weight 1), and `seasonal_factor=False, best_periods=None` weights all periods. `dynamic=True` with numeric seasonal factors
raises with R's static-seasonality error. Where the lag weights are `0/0` (a
constant series, or selected periods whose CV equals the variable CV), the
weights fall back to equal weights as in the patched R, so a constant series
forecasts its constant. Character `weights` such as `"equal"` weight every lag
equally, including with numeric multi-lag seasonal factors. Explicit numeric multi-lag seasonal factors weight
each lag by the coefficient of variation of that lag's own reverse component
series, as installed R does (`seasonal.factor[i]`).

Arguments are validated up front where R either errors or returns garbage:
`training_set` must lie in `1..n`, numeric lags in `1..n`, numeric `weights`
need one entry per lag (R recycles silently), and automatic seasonality needs at
least 5 observations. `seasonal_factor=None` behaves like R's `NULL`, i.e. like
`False`. With `h=1` and `pred_int`, where R fails, the band is the forecast
widened by the linear residual term because there is no forecast path for
`nns_mc` to bootstrap. `nns_arma_optim(pred_int=None)` returns empty bounds like
R.

`nns_arma_optim` is supported for the installed-R optimizer path. It greedily
selects seasonal factors, evaluates the default co-moment-normalized objective,
then applies the same equal-weight, shrink, smooth-regressed variable and
bias-shift checks as R, the bias last, on the returned model. The optimizer's
prediction intervals add the `lpm_var`/`upm_var` of that model's validation
residuals (actual - forecast) to the results; they are separate from
`nns_arma(pred_int=...)`, which uses the Monte Carlo path. Custom Python
`obj_fn` callables may be supplied, but R expression objects are not part of the
Python API. `nns_var` is implemented for numeric matrix-like inputs (a pandas
DataFrame keeps its column names) with `dim_red_method="cor"`,
`dim_red_method="NNS.dep"`, `dim_red_method="NNS.caus"`, and
`dim_red_method="all"` and returns R-compatible public output keys. As in R,
leading and interior missing values are filled from an index regression; when a series has
trailing missing values, every missing value comes from an `nns_stack` Method 1
fit on the index. Each variable's univariate `nns_arma_optim` forecast is
appended before the lagged design is built. For each variable the relevant
variables are found as in NNS 11.6: the dimension reduction threshold is
cross-validated on the last `max(2h, ceil(0.2 n))` lagged training rows, and
the regressors whose coefficient meets it are kept. The multivariate estimate is
`nns_stack(method=1, stack=False)` on exactly those regressors (`nns_reg` with
each its own dimension and a cross-validated `n_best`), trained on the observed
lagged rows and evaluated on the `h` forecast-period rows of the lag matrix built
from the series extended with the univariate forecasts (as in the patched R; 13.3
evaluated on the last `h` observed rows, an in-sample refit). `relevant_variables` is that
selection, so it lists exactly the regressors each multivariate estimate is
built from. `obj_fn` (a `(predicted, actual) -> float` callable) and
`objective` apply to both stages; without `obj_fn` the default MSE /
co-movement objective falls back to MSE when it is undefined for every Method 1
candidate, as in R. Series whose observed minimum is
positive have their interpolated values floored at zero. The `h == 0` path is normalized to
a Python dictionary containing `interpolated_and_extrapolated` and `names`
rather than R's bare data-frame return. `nowcast` (row labelling only in R) is
accepted for API compatibility; `ncores` caps the native threads (see Threads).

## Meboot

`nns_meboot` maps to R's `NNS.meboot` maximum-entropy bootstrap algorithm and
returns plain Python dictionaries instead of R's vectorized list-matrix wrapper.
`rho`, `target_drift` and `target_drift_scale` are vectorized like R's
`Vectorize`/`mapply`: the given arguments are recycled elementwise (not crossed)
and a list of result dictionaries is returned in that order; when all of them are
scalars a single result dictionary is returned. With none of the three given there
is nothing to map over and, as in R, the result is empty; length-one input returns
only `{"x": x}`. When the detrended
residuals are all exactly zero (a constant or exactly linear series) `rho` has no
effect, as in R: the replicates are copies of the trend with their means spread by
`force_clt`, and a constant series gives constant columns.

With `random_seed = s` the replicates are R's after `set.seed(s)`, bit for bit:
R draws only through `runif` (one per observation per replicate, then one per
shrunken column in each `expand.sd` pass), which the port reproduces from R's
Mersenne-Twister stream, and the deterministic steps follow R's arithmetic,
including `optimize()` (Brent's `fmin`, with non-finite objective values replaced
by the largest double as R does) for the `rho` target, `qnorm` for the CLT
spread and R's trimmed `mean`. Without a seed the stream is seeded from fresh
entropy. `force_clt` spreads the replicate means by `sd / sqrt(n)` with `n` the
series length, R 14's correction of meboot's division by the number of
replicates. Given only `target_drift`/`target_drift_scale` (no `rho`), the
replicates are drawn without a dependence target, as R's `Vectorize` does;
`expand_residual_sd` is R's `expand.residual.sd`.

## Monte Carlo

`nns_mc` maps to R's `NNS.MC` wrapper around `NNS.meboot`. The rho grid and
exponential rho transformation are parity-tested exactly against installed R.
With `random_seed = s` the replicates and ensemble are R's after
`set.seed(s)`. Arguments are validated with R's messages, the ensemble is R's
long-double `rowMeans`, and the replicate names format each rho as R's
`as.character` does (`"rho = 1e-04"`).

NNS Python returns `{"ensemble": array, "replicates": dict}`. The `replicates`
mapping preserves R's names, such as `"rho = 1"` and `"rho = -0.5"`, with each
value containing that rho block's replicate matrix. Sampling-vignette examples
are covered as smoke tests, but installed R behavior remains the parity source.

## Normalization

`nns_norm(x, linear=False)` maps to R's numeric matrix `NNS.norm` path
(`plot=True` draws R's `chart.type = "l"` panels). NNS Python accepts finite 2D arrays. `linear=True` uses R's
mean-ratio scaling, while `linear=False` additionally weights scaling by
absolute correlation for fewer than 10 columns and NNS dependence for 10 or
more columns.

## Distance

`nns_distance` and `nns_distance_bulk` map to R's regression-point-matrix
helpers. NNS Python accepts `rpm` as a finite 2D numeric array with R's `y.hat`
column in the final position. `nns_distance` applies R's per-target min-max
rescaling before computing weighted nearest-neighbor predictions. `nns_distance_bulk`
matches R's compiled bulk helper, including its raw-feature distance convention.
For `nns_distance` with `k > 1`, NNS Python matches the installed R 13.0 binary:
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

R classification paths work with numeric class codes: factor levels become
1-indexed codes in level order and predictions are returned as codes rather
than labels, with `class.levels` as the label map. NNS Python follows that
contract. pandas categoricals carry their level order; `class_levels=` (for
the response) and `factor_levels=` (for predictors) declare it explicitly for
plain arrays; undeclared string labels are coded in sorted order. String
labels are accepted directly by `nns_reg`, `nns_m_reg`, `nns_stack` and
`nns_boost`. The standalone helpers `factor_2_dummy`, `factor_2_dummy_fr` and
`encode_factor_codes` take `levels=` for the same purpose. Class confidence
intervals are supported in `nns_reg` and `nns_m_reg`; stack/boost class
`pred_int` is supported through those regression interval tables.

## Differentiation

`nns_diff` maps to R's scalar callable `NNS.diff` path with trace
output disabled (`plot=True` draws the function and secant panel). It returns a dictionary keyed by R's matrix row names and
rounds results to `digits`, matching R's default output convention.
`dy_dx(..., eval_point="overall")` maps to R's `dy.dx(..., eval.point =
"overall")` path and returns the mean fitted gradient from unsmoothed
`nns_reg`. Numeric `dy_dx` evaluation points use R's finite-difference grid
around smooth `nns_reg` point estimates, stepping by
`gravity(diff(sort(unique(x)))) * h` for each bandwidth `h` in R's `h_s` (the
spacing of the sorted values, as in the patched R, so row order does not matter),
and return a
table-like dictionary with `eval.point`, `first.derivative`, and
`second.derivative`. At `min(x)` or `max(x)` the first derivative is R's
one-sided slope and the second derivative is NaN. As in R, every position in
`h_s` counts once, so a bandwidth that repeats (n < 49) is averaged in each
time it appears.
`eval_point` must be given (a number, a vector, or `"overall"`).
`nns_diff` finds the inferred step like R's `uniroot(extendInt = "yes")` with
`tol = tol / 10`, and declares convergence when the bisection bracket on the
intercept `B` is narrower than `tol * max(|point|, |f(point) - B|)`, a
derivative error of about `tol * max(1, |f'(point)|)` (as in the patched R), so
iterations and convergence flags match R.

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

`nns_anova` maps to R's `NNS.ANOVA` paths (`plot=False` by default; see Plotting). Binary comparisons
return a dictionary keyed like R's list output, aggregate multi-group
comparisons return `{"Certainty": value}`, and `pairwise=True` returns R's
symmetric certainty matrix. The grand statistic is the mean (or, with
`medians=True`, the median) of all observations pooled, as in the patched R.
`NaN` values are dropped from every group, as R does with `na.omit`; `±Inf` raises a `ValueError` (R's binary path fails on it in
`LPM.VaR`). Confidence-interval bootstrapping draws from a
reproduction of R's Mersenne-Twister `sample()` stream, so
`random_seed=s` gives the same effect-size bounds as `set.seed(s)` followed by
the same `NNS.ANOVA` call in R; without a seed the stream starts from fresh
entropy. `robust=True` follows R's resampling exactly: 100 independent bootstrap
resamples of each group, each from its own full sample and size (as in the patched
R; 13.3 resampled only the first `min(n_control, n_treatment)` positions of both
groups). It returns `Robust Certainty Estimate`, the median of the finite
resampled certainties (R's `median(na.rm = TRUE)`), plus R's `Lower <level>% CI` and `Upper <level>% CI` bounds, and no bounds when
`confidence_interval=None`. R names the effect-size keys of the robust output
`Effect_Size_LB.2.5%` etc. (an `unlist()` artefact); NNS Python keeps
`Effect_Size_LB`/`Effect_Size_UB`. One-sided effect-size intervals follow the
patched R: `tails="left"` gives `(treatment_lower - control_upper, Inf)` and
`tails="right"` gives `(-Inf, treatment_upper - control_lower)` (R 13.3 failed
on them). Degenerate zero-variance groups preserve R's `NaN` CDF/certainty
convention; where every robust certainty is `NaN`, R's VaR stops while NNS Python
returns `NaN` bounds.
