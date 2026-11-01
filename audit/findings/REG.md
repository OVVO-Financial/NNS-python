# REG findings: univariate/multivariate regression, distance, differentiation

Scope: `nns_reg` / `NNS.reg`, `nns_m_reg` / `NNS.M.reg`, `nns_distance`, `dy_dx`, `dy_d`, `nns_diff`, smoothing.
Python reference: `/home/user/NNS-python/src/nns` (public `nns.nns_reg` -> `_public_reg.nns_reg` -> `regression.nns_reg` -> `_reg_engine.nns_reg_engine`).
R reference: NNS 13.3 (commit 82a92eb), run with `Rscript` in the C locale.

The A/B harness is in `/home/user/audit/work/REG/`:
- `gen_cases.py` and `gen_cases2.py` write 87 cases to `cases/<name>/`.
- `run_r.R` and `run_py.py` write outputs to `out_r/` and `out_py/`.
- `compare.py` compares them at rtol 1e-7. The full listing is in `compare_all.txt`.

Result: **72 of 87 cases match R to 1e-7** across R2, SE, Point.est, pred.int, regression.points, derivative, the whole Fitted.xy, RPM, rhs.partitions and n.best. Every mismatch is explained below. `bool_y` is a harness artefact: pandas writes True/False and R reads it as character.

---

### REG-01: `smooth=True` spline port is not R's `smooth.spline` (wrong penalty constant, ignores knot thinning)
- Type: python-bug (parity-mismatch)
- Severity: high
- Confidence: verified
- Where: `_reg_engine.py:524-606` (`_smooth_spline_predictor`). Knots at line 556 use every unique x. The penalty at lines 563-575 is the exact Gauss-Legendre integral. R side: `stats::smooth.spline` (`.nknots.smspl`, Fortran `sgram.f`).
- Evidence:
  - Every smooth case in `compare_all.txt` differs from R. In `sin_smooth`, Point.est is off by up to 1.3%. In `bigsin_smooth` (5000 obs) it is off by up to 8.8% (0.4795 R vs 0.4751 py). `fin_univ_smooth` differs by up to 43% relative on near-zero returns; `boston_lstat_smooth` by 1e-4.
  - Direct test (`smooth/ss.R`, `smooth/ss.py`, x of length 8, spar 0.3):
    - R lambda is 1.77990e-4; Python's is 1.77487e-4.
    - Predictions differ at 1e-4.
    - At n=80, R uses 59 knots and Python uses 80; predictions differ at 1e-2.
  - Root causes:
    1. R's `sgram` integrates B''·B'' with the linear-approximation formula using the constant **0.3330** instead of 1/3.
    2. For more than 49 unique x, R uses `nknots = trunc(.nknots.smspl(nx))` knots at `xbar[seq.int(1, nx, length.out = nknots)]`. Python always places a knot at every unique x.
  - Fix verified: `smooth/patched_ss.py`, installed by `run_py_patched_smooth.py`, reproduces R to 12 digits (`smooth/check_patch.py`). With it, **all 8 smooth A/B cases match R at 1e-7** (out_py_patched).
  - The parity tests hide this. `tests/parity/test_regression.py` uses `SMOOTH_SPLINE_PARITY = 5e-4` and small n, with the comment "reproduces to ~1e-4 (documented deviation)".
- Which side is right: R. The port claims to be "faithful", and R's behaviour is a stable, documented stats routine.
- Proposed fix:
  - Build Sigma with the sgram formula: `w*(y1 y1' + .5(y2 y1' + y1 y2') + .3330 y2 y2')`, where y1 is B'' at the left knot and y2 is the change in B'' across the interval.
  - Implement `.nknots.smspl` with index truncation.
  - Optionally apply R's tie tolerance `1e-6*IQR(x)`.
  - Vectorise: evaluate all basis second derivatives at the knots once with `BSpline.design_matrix(..., nu=2)`.
  - Changes public results for every `smooth=True` call and for every `dy_dx` call (see REG-02).

### REG-02: `dy_dx` numeric path uses the bandwidth's *index* instead of its *value* for the step
- Type: python-bug
- Severity: high
- Confidence: verified
- Where: `diff.py:703-705`: `indices = flatnonzero(h_s == h_value) + 1; h_step = spacing * indices`. R `dy_dx.R:67`: `h_step <- gravity(abs(diff(x))) * h_s[index]`.
- Evidence: `diff/dydx.R`, `diff/dydx.py`, `diff/dydx_patched.py`.
  - Cubic y=x^3 at (-1.5, 0, 0.5, 1.5): Python first derivative is (3.859, 3.223, 3.282, 3.900); R gives (3.708, 3.774, 3.679, 3.719).
  - Sorted cubic: Python (6.748, 0.0005, 0.750, 6.754) vs R (6.873, 0.0042, 0.754, 6.910).
  - R's own example, `sin` on `seq(0, 2*pi, pi/100)` at 1.75: Python -0.18161 vs R -0.18268.
  - Using `h_s[index]` plus the REG-01 spline fix makes the cubic case match R to 1e-8.
  - Effect: Python uses steps of 1..5 × spacing; R uses 2..sqrt(n) × spacing.
  - Existing parity tests use n=24 and atol 5e-3, which masks this.
- Which side is right: R. It is the documented algorithm, and wider bandwidths are the intent of `h_s`.
- Proposed fix: `h_step = spacing * h_s[h_s == h_value]`. Changes all numeric `dy_dx` results.

### REG-03: `point_est` column names are ignored, so permuted DataFrame columns give silently wrong predictions
- Type: python-bug (parity-mismatch)
- Severity: high
- Confidence: verified
- Where: `_reg_engine.py:231-246` (`_prepare_points`) and `_encode_predictors` (249-256). R `Regression.R:159-231` (`.nns_match_predictor_names`).
- Evidence: `api/api1.py` and `api/api1.R`, mtcars `x = [wt, hp]`.
  - R: `point.est = pe[, c("hp","wt")]` gives 21.0 and 21.2, the same as unpermuted, because it reorders by name.
  - Python with the permuted pandas frame gives **10.4 and 10.4**, with no warning.
  - R raises an error on partial name overlap; Python never checks.
- Which side is right: R. Names carry alignment information, and the documentation says point.est columns are matched by name.
- Proposed fix:
  - If both x and point_est carry column labels and the sets are equal, reorder.
  - If they are disjoint, match positionally.
  - Otherwise raise.
  - No change for numpy inputs.

### REG-04: R `dy.d_` partial derivatives are not partial derivatives (both sides)
- Type: r-bug (Python replicates it bit-for-bit)
- Severity: high
- Confidence: verified
- Where: R `dy_d_wrt.R:119-133`. The estimates come from `NNS.stack` on `Xstar = rowMeans(x)`, using `cbind(Xstar, Xstar)` for both train and test points. Python `diff.py:369-398` does the same.
- Evidence:
  - `diff/dyd2.R`: y = 2*x1 with x2 irrelevant, n=500. dy/dx1 is (0.83, 0.78, 1.05), truth 2. **dy/dx2 is (1.07, 0.83, 0.90), truth 0.**
  - `diff/dyd.R` and `diff/dyd.py`: y = x1^2 + 3*x2. dy/dx1 at x1 = (-1, 0, 1, 1.5) is (1.06, 1.58, 1.49, 1.49), truth (-2, 0, 2, 3). dy/dx2 is about 1.5, truth 3.
  - Python and R agree to 5 digits.
  - Collapsing to the equal-weight index means f(x ± h e_j) depends only on mean(x). Every regressor therefore gets ≈ the index-model slope/p, whatever the true dependence.
- Which side is right: neither. Mathematically a partial derivative requires varying x_j with the others held fixed in a model that can tell them apart.
- Proposed fix (R and Python): estimate f(x ± h e_j) with the full multivariate `NNS.reg` / `NNS.stack` on the actual design (no rowMeans). Keep the dy.dx-style adaptive step. Add an analytic regression test such as y = 2*x1 → (2, 0).

### REG-05: R `dy.dx` is not invariant to row order: the step comes from `diff()` of unsorted x (both sides)
- Type: r-bug (Python replicates it)
- Severity: high
- Confidence: verified
- Where: R `dy_dx.R:67` and Python `diff.py:701`: `spacing = gravity(abs(diff(x)))` on x as supplied.
- Evidence (`diff/dydx.R`):
  - y = x^3 with 500 unsorted uniform x: first derivative is (3.71, 3.77, 3.68, 3.72) at (-1.5, 0, 0.5, 1.5). The same data sorted gives (6.87, 0.004, 0.75, 6.91); truth (6.75, 0, 0.75, 6.75).
  - Finance (SPY on QQQ daily returns), unsorted vs sorted at (-0.02, 0, 0.02): (0.907, 0.871, 0.889) vs (0.726, 0.580, 0.805).
  - Unsorted spacing is roughly the sample's typical |Δx| (≈ σ), so 2..√n steps span the whole range.
- Which side is right: neither. A derivative estimate must not depend on row order.
- Proposed fix: compute the spacing on `diff(sort(unique(x)))`, or on the order statistics. This changes results only for unsorted inputs.

### REG-06: Multivariate `nns_reg` is 5-126x slower than R; brute-force 3-D broadcast distances, missing R short-cuts
- Type: perf
- Severity: high
- Confidence: verified
- Where:
  - `_reg_engine.py:904-926` (`_mreg_distances`) builds a `(chunk, n_rpm, p)` temporary and then two more.
  - `_reg_engine.py:953-1005`: per-row Python loops for k==1 and class mode.
  - `_reg_engine.py:1103-1131` (`_mreg_build_rpm`) calls the reducer once per cell per column, including singleton cells.
  - `_reg_engine.py:1202-1205`: no `order="max", k==1` identity shortcut. R has one at `Multivariate_Regression.R:418-431`.
  - Distance and prediction run in pure NumPy, while R uses `NNS_mreg_predict_v2_cpp`.
- Evidence (`perf/time_r.R`, `perf/time_py.py`, `perf/time_*.txt`). Best of 2-3 runs, 1 thread, on a shared 4-core box:

| case | R 13.3 | Python | ratio |
|---|---|---|---|
| univariate n=1e3 default (100 point.est) | 0.006 s | 0.004 s | 0.7x |
| univariate n=1e4 default | 0.035 s | 0.048 s | 1.4x |
| univariate n=1e5 default | 0.397 s | 0.414 s | 1.0x |
| univariate n=1e5 smooth | 0.623 s | 0.624 s | 1.0x |
| univariate n=1e4 order="max" | 0.071 s | 0.162 s | 2.3x |
| univariate n=1e5 order="max" | 0.99 s | 2.25 s | 2.3x |
| **mv n=1e4, p=5 default** (n.best=92, 9711 RPM rows) | **2.24 s** | **11.8 s** | **5.3x** |
| mv n=1e4, p=5 n.best=1 | 2.38 s | 10.7 s | 4.5x |
| **mv n=1e4, p=5 order="max"** | **0.080 s** | **10.1 s** | **126x** |
| mv n=1e4, p=5 binary class | 2.29 s | 24.1 s | 10x |
| mv n=3e4, p=2 order=2 n.best=1 | 0.108 s | 0.657 s | 6x |
| finance 5 regressors (n≈4200) default | 0.49 s | 3.41 s | 7x |
| mv dim.red cor / NNS.dep / all (n=1e4) | 0.08 / 0.14 / 0.34 s | 0.05 / 0.11 / 0.38 s | ≈1x |

  - cProfile (`perf/prof_mv.py`): `_mreg_distances` takes 8.1 s of 14 s on the default mv case. The `_reduce_value` / `nns_gravity` per-cell loop takes 2.5 s (58k calls). `_stable_topk`'s per-row loop takes 0.64 s.
  - Prototype (`perf/speedup_patch.py`, `perf/speedup_check.py`; outputs **bit-identical** on 6 configurations):
    - Accumulate distances column by column in 2-D buffers, with the same per-element arithmetic.
    - Add a singleton fast path in `_mreg_build_rpm`.
    - Result: mv 1e4 default goes from 11.6 s to 3.9 s; Boston FACTOR/L2 from 0.17 s to 0.05 s.
- Proposed fixes, all result-preserving:
  1. Column-loop distance accumulation (≈3x).
  2. Singleton-cell fast path, plus a vectorised mean/median for multi-member cells.
  3. R's `order="max" && k==1` shortcut: fitted = y, or the RPM y.hat mapped by ID. That turns 10 s into about 0.05 s.
  4. Port `NNS_mreg_predict_v2_cpp`, the fused distance + top-k + ensemble-weight kernel, into `_nnscore` with optional threads. That should beat R once `ncores` is honoured.
  5. Vectorise the class-mode weighted mode with `np.add.at` on (row, class) indices.
  6. Univariate `order="max"`: replace the `f"{v:.15g}"` per-key list and per-group reducer with `np.unique` + `reduceat` for singletons.

### REG-07: `dy_dx` boundary fallback corrupts the estimate (wrong column, wrong sign, applied after prediction)
- Type: python-bug
- Severity: medium
- Confidence: verified
- Where: `diff.py:729-736`. When a run is 0 (eval point at min(x) or max(x)), Python rewrites `deriv_points[:, 2]` to `eval - fallback`. That puts the upper point *below* the eval point, and it also does this for zero *lower* runs. It happens after `nns_reg` has already been evaluated on the original points, so rise and run no longer correspond.
- Evidence (`diff/dydx.py`, `sin_bound`): sin'(0) gives Python **-0.563** vs R 0.965 (truth 1). At 2π: Python 35.55 and second derivative -443 vs R 0.966 / NaN.
- Which side is right: R, which is approximately right; truth is 1. R's own fallback is dead code (REG-22), so R simply uses the one-sided slope.
- Proposed fix: drop the post-hoc rewrite and let `first = (rise1+rise2)/(run1+run2)` use the one-sided run, as R does. Better: shift the window inward before calling `nns_reg`.

### REG-08: `NNS.part` rounds regression-point **x** to integers for integer-valued x but leaves y, biasing `NNS.reg` (both sides)
- Type: r-bug (Python replicates it)
- Severity: medium
- Confidence: verified
- Where: R `Partition_Map.R:220-228` (`if (is.discrete(x)) RP$x <- round-half-up(RP$x)`). Python `part.py:225` (`_is_discrete_like_r`). This is the partition code, but it is visible through every `NNS.reg` with integer predictors such as counts, ages, years, Boston rad/tax/zn.
- Evidence (`repro/integer_x_bias.R`, `api` checks):
  - y = 2x on x = 1..1000 gives `max|y.hat - y| = 1`, with first regression points (1, 2, 4) / (2, 3, 8.5). The point (2, 3) lies off the line.
  - Shifting x by 0.5 gives an exact fit.
  - n=10 gives R2 0.9938 on a perfect line in both R and Python.
- Which side is right: neither. The cell's y-centroid is paired with a rounded, shifted x.
- Proposed fix: do not round RP$x. If discreteness matters, round both coordinates consistently or snap to an observed x with its own reduced y. This changes results for integer-valued predictors.

### REG-09: `dim.red.method = "NNS.dep" / "NNS.caus" / "all"` coefficients differ from R on tied/discrete predictors
- Type: parity-mismatch (root cause in `nns_dep` / `_uni_caus`; cross-ref the DEP group)
- Severity: medium
- Confidence: verified
- Where: `_reg_engine.py:731-761` calls `nns_dep` and `causation._uni_caus`. The discrepancy is inside those functions.
- Evidence (`dimred/dep.R`, `dimred/dep.py`, Boston):
  - `NNS.dep(x_j, medv, asym=TRUE)` differs only for ptratio (R 0.46074 vs py 0.38679) and black (0.30305 vs 0.26503). These are the heavily tied columns.
  - `Uni.caus(medv, x_j, "cs")` differs for chas (0.01371 vs 0), nox, rad (0.1577 vs 0.1073), tax, ptratio and black.
  - Downstream, `boston_dr_NNS.dep` Point.est is off by up to 4.5%, `boston_dr_NNS.caus` by 2.5%, and `boston_dr_all` by 15% (R2 0.109 vs 0.089).
  - `cor`, `equal`, numeric vector and threshold paths all match exactly.
  - Also `nns_dep(x, sin(x))` on R's `seq(0, 2*pi, pi/100)` gives 0.97937 vs R 0.98070 (`repro/dep_sin_grid.*`).
- Which side is right: needs the DEP analysis. For regression, parity with R is the stated goal.
- Proposed fix: fix `nns_dep` / `_uni_caus` tie handling (DEP group). No change needed in the regression engine.

### REG-10: Default order and smoothing `spar` hinge on a last-ulp `np.mean` vs R `colMeans` difference (knife-edge in `NNS.copula`)
- Type: python-bug (parity) plus r-bug (discontinuous copula)
- Severity: medium
- Confidence: verified
- Where: `_reg_engine.py:402` (`target = np.mean(m, axis=0)`) and `copula.py:70`. R `Copula.R:80` (`if (discrete_Co_pm==1 || discrete_Co_pm==0) return(1)`).
- Evidence (`repro/dep_sin_grid.R`, `repro/dep_sin_grid.py`; R's own `dy.dx` example data):
  - Python's column mean of rescaled `sin(x)` is 0.49999999999999994; R's long-double `colMeans` gives 0.5.
  - One observation sitting exactly at 0.5 changes the degree-0 co-LPM. That makes `discrete_Co_pm` exactly 1, and the early return fires.
  - Python's copula is **1.0**, R's 0.5759.
  - The regression dependence becomes 0.990 vs 0.778, the default order 10 vs 8, and the smooth `spar` 0.745 vs 0.639.
  - Using a long-double mean (`np.sum(m.astype(np.longdouble), 0)/n`) reproduces R's 0.5759122837209066 exactly.
  - On the 35 other univariate A/B cases, dependence and order match.
- Which side is right:
  - Python should reproduce R's mean.
  - The R early return is itself a bug. An exact-equality test on a 3-pair sum is not a perfect-dependence test: one tie at the mean flips the result from 0.58 to 1.
- Proposed fix:
  - Python: an R-compatible mean helper, long-double sum / n with R's second-pass refinement for `mean()`. Use it in `_dependence`, `nns_copula` targets and `_reduce_value("mean")`.
  - R: remove the early return, or use a tolerance that requires all pairs to be degenerate.

### REG-11: R `noise.reduction="mode"` multivariate results depend on the native/pure-R path and on `order`
- Type: r-bug (produces a parity mismatch)
- Severity: medium
- Confidence: verified
- Where: R `src/NNS_mreg_setup.cpp:15-22`. Reducer code 2 there is the *discrete* most-frequent value (smallest on ties). `Regression.R:347` instead uses `mode(z)`, which is `NNS.mode(discrete=FALSE)`. The pure-R path (`options(NNS.native.mreg=FALSE)`) and the `order="max"` path use the latter.
- Evidence (`api` R one-liner in the session; Boston, all 13 predictors):
  - Native path: R2 0.83225.
  - Pure-R path: R2 0.87796.
  - Python follows the pure-R reference (R2 0.877960398), so it mismatches R's default (`boston_mv_nrmode`).
  - The mean reducer agrees between the two R paths.
- Which side is right: within R, the pure-R reference, because the C++ comment says it must reproduce it. Python is on the reference side.
- Proposed fix (R): make the C++ mode reducer call the same continuous `NNS.mode` used in R. No Python change needed beyond documenting this.

### REG-12: Categorical level order: R uses first appearance for character columns and declared levels for factors; Python always sorts
- Type: parity-mismatch, plus an r-bug (R results depend on row order)
- Severity: medium
- Confidence: verified
- Where: Python `_reg_engine.py:270-271` (and `_task` 148-153 for y). R `Regression.R:251` (`levels.train <- if (is.factor(z)) levels(z) else unique(as.character(z))`).
- Evidence:
  - `mtcars_factor_chr_nodummy` (character `cyl`, `factor.2.dummy=FALSE`): R2 0.9082 R vs 0.9273 py; Point.est off by 3.4%.
  - In R, shuffling the rows changes the fit, to R2 0.9200 (`api/rowperm_r.txt`).
  - pandas `Categorical(categories=[virginica, versicolor, setosa])` as y: R `factor(levels=...)` returns Point.est 3 2 1 with that level order. Python returns 1 2 3 and `['setosa','versicolor','virginica']`.
  - A Categorical predictor with custom order is also re-sorted (`api/api1.py` #10).
  - With dummies (`factor.2.dummy=TRUE`) the results match; only the column order and names differ.
- Which side is right:
  - Sorted is defensible for character columns. R's row-order dependence is a bug: it should use `sort(unique())`.
  - For declared categories, R is right and Python should honour `Series.cat.categories`.
- Proposed fix:
  - Python: read `.cat.categories` from pandas Categorical inputs for x and y.
  - R: use sorted levels for character predictors.

### REG-13: object-dtype numeric `y` is silently treated as a classification target with string-sorted classes
- Type: python-bug
- Severity: medium
- Confidence: verified
- Where: `_reg_engine.py:132` (`categorical = y.dtype.kind in {"U","S","O","b"}`). Predictors use `_numeric_like`; y does not.
- Evidence (`api/api1.py` #4): `nns_reg(boston.lstat, boston.medv.astype(object))` returns "R2" 0.0336, which is classification accuracy, with class.levels `['10.2','10.4',...]`. This happens whenever y comes from `np.asarray(mixed_df)[:, k]` or `df.values` of a mixed frame. R treats such y as numeric.
- Proposed fix: classify an object y as categorical only if it is not numeric-like, mirroring `_numeric_like`. No change for string or bool y.

### REG-14: Missing values in categorical predictors silently become a level
- Type: python-bug
- Severity: medium
- Confidence: verified
- Where: `_reg_engine.py:267-279` (`str(v)` of None/NaN).
- Evidence (`api/api1.py` #6): a column `["a","b",None,...]` produces RPM column `g_nan` and the fit proceeds. R raises "Predictor 'g' contains missing values." (`api/api1.R`).
- Proposed fix: before encoding, reject `pd.isna` in categorical columns and in categorical point_est.

### REG-15: `class_levels` / `factor_levels` are accepted and silently ignored by `nns_reg`; `nns_m_reg(class_levels=...)` turns classification into regression
- Type: api (python-bug)
- Severity: medium
- Confidence: verified
- Where:
  - `regression.py:205-206` (`del class_levels, factor_levels`).
  - `multivariate_regression.py:101-107`, which encodes labels to 1..K floats and does not set `type`.
- Evidence (`api/api1.py` #2-3):
  - `nns_reg(..., class_levels=["virginica","versicolor","setosa"])` gives the same codes and `class.levels` as without it.
  - `nns_m_reg(X, species, class_levels=[...])` returns R2 0.987 with non-integer y.hat (1.045...) and `class.levels=None`; without class_levels it auto-classifies (accuracy 1.0).
  - `docs/conventions.md` repeatedly tells users to pass `class_levels=` / `factor_levels=` to reproduce R ordering.
- Proposed fix:
  - Implement both: map labels to codes in the given order, set `class.levels`, apply `factor_levels` per column.
  - Or remove the parameters and raise `TypeError`.
  - In `nns_m_reg`, pass `type="class"` when `class_levels` is given.

### REG-16: `nns_m_reg` diverges from R `NNS.M.reg` API: wrong default, stale guard, crashes on strings, drops names
- Type: api
- Severity: medium
- Confidence: verified
- Where: `multivariate_regression.py:61-110`.
- Evidence:
  - The default is `factor_2_dummy=False`, while R's is TRUE.
  - `factor_2_dummy=True` raises `NotImplementedError`, claiming "installed R's internal NNS.M.reg raw factor path errors". In R 13.3, `NNS:::NNS.M.reg(data.frame(g=c("a","b","c"), z=...), y)` works (R2 0.786, RPM g_a g_b g_c z).
  - Any string column crashes on `np.asarray(x, dtype=float64)` with "could not convert string to float".
  - DataFrame column names are discarded (keys V1..Vp), whereas `nns_reg` keeps them.
  - The numeric path matches R exactly: R2, Point.est, n.best, RPM rows, pred.int and class mode were checked against `NNS:::NNS.M.reg`.
- Proposed fix:
  - Pass x through unchanged to `nns_reg_engine` (which already encodes predictors).
  - Default `factor_2_dummy=True`.
  - Delete the guard and `_validate_inputs`.

### REG-17: `nns_diff` uses a 1e-14 root tolerance where R's `uniroot` uses eps^0.25, so it reports non-convergence on smooth functions
- Type: parity-mismatch (python-bug in the reported flags)
- Severity: medium
- Confidence: verified
- Where:
  - `diff.py:845-853` (`brentq(xtol=1e-14)`); R `Numerical_Differentiation.R:167` (`uniroot(..., extendInt="yes")`, tol = 1.22e-4).
  - `diff.py:64`: `np.isclose(atol=0)` vs R `all.equal`, which uses an absolute difference when the target is about 0.
- Evidence (`diff/nnsdiff.R`, `diff/nnsdiff.py`):
  - For sin@1, x²@3, x³@2, log@5, tanh@0.5 and x⁴-3x@-2, R converges in 28-33 iterations. Python hits max_iter=100 with `converged=0, termination.code=1`.
  - DERIVATIVE values differ at 1e-7 to 2e-5 relative.
  - abs(x)@-1: R returns at iteration 0 (B1≈B2≈0); Python iterates once and reports `converged=0`.
  - With R's tolerance substituted (`diff/nnsdiff_patched.py`), Python reproduces R's derivative, iteration count and flags exactly on 4 of 5 functions; tanh agrees to 2e-11.
- Which side is right: R, for parity. Python's derivatives are slightly *more* accurate, but its convergence flags are wrong.
- Proposed fix: `xtol = eps**0.25` with R's `extendInt` behaviour, and an `all.equal`-style equality check (absolute when |target| < tol).

### REG-18: Python docs and type hints describe the removed legacy engine
- Type: docs
- Severity: medium
- Confidence: verified
- Where:
  - `docs/conventions.md:153-246`.
  - `regression.py:30-102` and `multivariate_regression.py:19-49` (TypedDicts).
  - `_reg_engine.py:12` (docstring).
  - `regression.py:198-202`, `_public_reg.py:39-43`.
- Evidence: what the docs say vs what the code does.

| What the docs / types say | What the code does |
|---|---|
| `nns_reg` maps to R with `factor.2.dummy = FALSE` | default is TRUE |
| Intervals use `UPM.VaR`/`LPM.VaR` | type-8 residual quantiles |
| Below-range `pred.int` rows are dropped | one row per point (verified: 4 rows for 4 points) |
| Segment `standard.errors` exist | none |
| `"mode_class"` is accepted | `ValueError` |
| smooth matches `smooth.spline` | REG-01 |
| "full-rank" dummies | full one-hot (4 columns for a 3-level factor plus 1 numeric) |
| R errors on raw factors | false (REG-16) |
| `RegFitted` has `standard.errors` | absent |
| `MRegPredInt` keys `lower.pred.int`/`upper.pred.int` | `pred.int.neg`/`pred.int.pos` |
| `MRegResult` point_only returns only `Point.est`/`RPM` | all keys, with None values |
| `RegResult` `Point.est` is "empty" | None |
| `_reg_engine` docstring: "predictive R2 = 1 - SSE/SST" | squared correlation (Racine-Hastie) |

  - `help(nns.nns_reg)` documents no parameters or returns.
- Proposed fix: rewrite the Regression and Multivariate sections and the TypedDicts from `_reg_engine`. Add a real docstring.

### REG-19: About 1,500 lines of dead regression code, including a second divergent implementation and an unused native kernel
- Type: dead-code
- Severity: medium
- Confidence: verified (coverage run `cov_driver.py` over the 87 cases plus stack/boost/dy_d/dy_dx/arma/m_reg/distance; grep for callers)
- Where:
  - `regression.py:227-1609`: `_nns_reg_legacy` and all of `_nns_reg_univariate_core`, `_nns_reg_dimred`, `_predict_points`, `_apply_univariate_intervals`, and so on. 565 of 716 statements are never executed; there are no callers.
  - `multivariate_regression.py:160-510`: 202 of 256 statements unexecuted, including `_validate_inputs`, `_predict_points` and `_outside_prediction`.
  - `smoothing.py`: the whole module, reachable only from the legacy code.
  - `distance.nns_distance_path_single_bulk`.
  - Native `_nnscore.nns_reg_mv`, reachable only from legacy code.
  - `diff._dy_d_stack_estimates` and `_r_seq`.
- Risk: the legacy code implements different semantics (UPM.VaR intervals, a different extrapolation rule, full-rank dummies) and still ships, so readers and agents patch the wrong function.
- Proposed fix: delete it; keep `prepare_factor_predictors`, `_maybe_render_reg` and the small helpers still imported. This does not change results.

### REG-20: `noise_reduction="mean"` uses `np.mean`, so constant groups do not reproduce the tied value (NNS.ID drift)
- Type: python-bug
- Severity: low
- Confidence: verified
- Where: `_reg_engine.py:354-355`. Also the `nns_part` mean reducer.
- Evidence (`boston_mv_nrmean`): 30 rows of `Fitted.xy.NNS.ID` differ, in the indus component (Python 7 vs R 8). `np.mean(np.full(30, 19.58))` is 19.580000000000002, while R's `mean()` is exactly 19.58. The boundary therefore moves above the tied observations. Fitted values happened to coincide here, but cell membership differs.
- Proposed fix: the same R-compatible mean as REG-10.

### REG-21: Predictor typing differs from R: bool is treated as numeric, numeric strings as numbers
- Type: parity-mismatch
- Severity: low
- Confidence: verified
- Where: `_reg_engine.py:267`, `333-340` (`_numeric_like`). R `Regression.R:248` (`is.factor || is.character || is.logical`).
- Evidence (`api/api1.py` and `api/api1.R`):
  - A univariate bool x stays univariate in Python. R creates dummies `x_TRUE.` / `x_FALSE.` and runs NNS.M.reg; Point.est is 47.3/48.2 py vs 48.75/48.75 R.
  - An object column `"1","2","10"` is numeric in Python (R2 0.990) and categorical in R (R2 0.646, columns g_X1, g_X2, g_X10).
- Which side is right: either is defensible; Python's numeric-string coercion is friendlier. Document it, or offer `factor_2_dummy` handling for bool to match R.

### REG-22: R `dy.dx` zero-run fallback is dead code and buggy
- Type: r-bug
- Severity: low
- Confidence: verified
- Where: R `dy_dx.R:85-90`. `eval.point.max` is assigned twice with opposite signs. It uses the stale loop variable `index` (always 5). The modified `run_1` / `run_2` are never used, because `run_1c` / `run_2c` are recomputed from the unmodified `deriv.points` at lines 97-105.
- Evidence (`sin_bound` in `diff/dydx.R`): the second derivative is NaN at both range ends; the first derivative is one-sided.
- Proposed fix: delete the block, or shift boundary windows inward before calling `NNS.reg`.

### REG-23: `nns_diff` (R) flags "converged" while the derivative error is ~1e-5
- Type: r-bug
- Severity: low
- Confidence: verified
- Where: R `Numerical_Differentiation.R:160-190`. Convergence is `|inferred.h| < tol` on a root found only to 1.2e-4.
- Evidence (`diff/nnsdiff.py` output):
  - log'(5): R 0.1999996164 (truth 0.2).
  - tanh'(0.5): R 0.7864373 (truth 0.7864477).
  - (x⁴-3x)'(-2): R -34.99943 (truth -35).
  - All are reported `converged=1`.
- Proposed fix: pass `tol = 1e-12` (or tol/10) to `uniroot`, and base convergence on B-interval width rather than on h.

### REG-24: R documentation does not match `NNS.reg` behaviour
- Type: docs
- Severity: low
- Confidence: verified (code reading, plus `sin_xonly` == `sin_default`)
- Where: R `Regression.R:662-786` (roxygen) and `man/NNS.reg.Rd`.
- Evidence:

| Topic | R docs say | Code does |
|---|---|---|
| `n.best` default | "sqrt(# of regressors)" | `floor((1-NNS.copula(cbind(X,y)))*sqrt(nrow(X)))` |
| `dim.red.method="cor"` | "standard linear correlation" | Spearman |
| `confidence.interval` | "reports the standard error for each individual segment" | global type-8 residual quantiles; no SE output |
| Multivariate `Fitted.xy` | includes "gradient" | absent |
| `type="XONLY"` example | x-only partitioning | no effect; `.nns_reg_build_points` ignores `xonly` and always partitions XONLY |
| Note on CLASS | "base category should be 1" | auto-classification triggers only for exactly {0,1} |
| equation "Denominator" | coefficients > 0 | `abs(coef) > 0` |

  - Vignette 01 lists `NNS.diff(x, y)`; the signature is `(f, point)`.
- Proposed fix: update the roxygen and vignette text.

### REG-25: Smaller API rough edges
- Type: api
- Severity: low
- Confidence: verified (`api/api1.py`)
- Where: `_reg_engine.py:183-228` (`_make_names`), `_task` 157, `_public_reg.py`.
- Evidence:
  - Column names are passed through `make.names` (`"my col"` → `my.col`, `"2nd"` → `X2nd`), so result keys differ from R, which keeps raw names via `check.names=FALSE`.
  - Numeric class levels are formatted `'0.0','1.0'` vs R `'0','1'`.
  - `location=` (an R argument) raises `TypeError`.
  - `ncores` is silently ignored in `nns_reg`, but `nns_m_reg` warns.
  - `_r2` uses exact `np.array_equal` where R uses `all.equal`, in the zero-variance branch.
  - `_task` / `_encode_predictors` encode labels with an O(n·K) `levels.index(v)` list loop.

---

## Coverage (checked and found fine)
- **Univariate `nns_reg` vs R, exact to 1e-7**: R2, SE, Point.est, pred.int, regression.points, derivative table, and the whole Fitted.xy (y.hat, NNS.ID, gradient, residuals, conf.int.*). The following settings were checked:
  - Datasets: unsorted sin (n=400), x² (n=500), step with ties (2-decimal x), big sin (n=5000), integer x and y, finance SPY~QQQ (n=4232), Boston lstat, mtcars wt.
  - Orders: default, 1, 3, 5 and `"max"`.
  - Noise reduction: mean, median, mode.
  - Other: `type="XONLY"`, `point_only`, CI 0.90 and 0.95, and point.est outside the training range (linear extrapolation).
  - Edge cases: n=2, n=3, constant y, constant x.
- **Classification**: binary 0/1 auto-class (with CI), iris CLASS, auto-class from factor, numeric codes with and without CLASS, n.best=3, order="max", L2, noise mean, dim.red cor with CLASS, univariate class, and mtcars am auto-class. `Prediction.Accuracy` and class `pred.int` snapping match.
- **Multivariate `nns_reg`** (Boston 13 predictors, mtcars 3 and 10 predictors, finance 2 and 5 regressors): exact for these settings.
  - default, n.best 1, 5 and "all"
  - order 1 and 2, and order "max"
  - dist L1, L2 and FACTOR
  - noise mean and median (values exact; only the IDs differ, REG-20)
  - CI, point_only, and point.est outside the training range
  - factor predictors with dummies (character and factor)
- **dim.red**: `"cor"`, `"equal"`, numeric vector, threshold 0.4, finance `"cor"`, `"NNS.dep"` and `"NNS.caus"` with tau="ts" (2 regressors) are exact. The x.star/equation normalisation of point.est uses training min/max on both sides.
- **`nns_m_reg` direct** vs `NNS:::NNS.M.reg`: numeric default, order=2 with n.best=3 and CI 0.8, and iris CLASS are exact.
- **`nns_distance`** vs `NNS.distance`: 16 combinations of point × k ∈ {1, 2, 5, "all"}, regression and class, all exact (1e-12).
- **`dy_dx(eval_point="overall")`** matches R (5.6289e-06).
- **`dy_d`** matches R `dy.d_` to 5 digits (deterministic). REG-04 is about the algorithm itself.
- **`nns_diff`** with immediate exit (linear, x²@0, exp@0) matches R exactly.
- **Validation messages** mirror R: order, n.best, dist, noise.reduction, confidence.interval range, y length < 2, NaN in x/y, XONLY with more than one predictor, tau.
- **R `NNS.distance.bulk`** ignores `class` (`(void)is_class`) and uses unscaled distances; Python mirrors this. It is not exported in R.
- **`order="max"` in R and Python** uses `split` and as.character(15-digit) grouping; Python replicates it, and the step_ties_omax case matches.
- **Locale**: R runs in the C locale here, so string-ID ordering (`order(ids)` vs the native `std::sort`) agrees. In non-C locales R's pure-R path collates IDs differently from its native path. Not testable here (only C locales installed), so speculative.
