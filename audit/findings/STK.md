# STK findings: NNS.stack, NNS.boost, categorical helpers, R RNG emulation

Scope: `src/nns/stack.py`, `boost.py`, `categorical.py`, `_rrng.py` against R 13.3 `R/Stack.R`,
`R/Boost.R`, `src/NNS_stack_fast.cpp`, `src/table_ops.cpp`, the man pages and vignettes 01/07/08.

All scripts are under `/home/user/audit/work/STK/`:
- `ab/compare.py` and `ab/boostcmp.py` are generic A/B drivers. They write CSV and JSON, run
  `ab/r_stack.R` or `ab/r_boost.R`, run Python, and print the diffs.
- `ab/cases/<name>/` holds the inputs, `r.json` and the spec for every case.

Overall result: on continuous data, the default 5-fold path, `CV.size` holdouts, `ts.test`, pred.int,
custom `obj.fn`/`objective`, numeric `dim.red.method`, `order`/`dist` options and NNS.boost all
match R to about 1e-14. The RNG streams are reproduced exactly. The remaining divergences come from
(a) last-ulp differences that NNS's discontinuous partitioning amplifies, (b) `nns_dep` on tied data
(DEP-08/09), and (c) label and level ordering. R itself has several real bugs on these paths.

---

### STK-01: Root cause of the 2 failing Boston `nns_stack` parity tests: the Python call uses `cv_size=0.25` but the R reference uses `CV.size=NULL` (5-fold CV)
- Type: python-bug (test fixture)
- Severity: high (the suite's only red tests, and the default CV path has no parity coverage)
- Confidence: verified
- Where:
  - `tests/parity/test_practical_examples.py:114-123` and `:157-165` pass `cv_size=0.25`.
  - The R scripts at `:424-430` and `:478-484` call `NNS.stack(...)` with no `CV.size`.
- Evidence: `work/STK/boston/r_boston.R` and `py_boston.py` (same split, same data):

  | call | R stack RMSE | Python stack RMSE | weights (R / Py) |
  |---|---|---|---|
  | default (5-fold) | 3.739482377561860 | 3.7394823775618624 | (1,0) / (1,0) |
  | CV.size = 0.25 | 3.71468893125048 | 3.7146889312504823 | (.57,.43) / (.57,.43) |

  - The test's "Python 3.7147 vs R 3.7395" is cv=0.25 Python against default R.
  - Aligned, every output agrees to 6e-14: reg, dim.red, stack, n.best, threshold and the stack head.
  - No R 13.3 Stack.R change is involved. `git diff 497ce80 b93a272 -- R/Stack.R` only adds
    `NNS.dim.red.variables/coefficients`.
  - The test also overwrites `params.threshold` with R's value (`:135`, `:174`) "because Python
    keeps the equivalent threshold". Python's `NNS.dim.red.threshold` is actually identical to R
    (0.6588617024605853), so this is goal-seeking that masks the assertion.
  - Every parity test in `tests/parity/test_stack.py` pins `cv_size=0.25` (lines 26-785), so the
    default `CV.size=NULL` k-fold path (what users get) is not parity-tested at all.
- Which side is right: neither side is wrong. The test is inconsistent.
- Proposed fix:
  - Drop `cv_size=0.25` from both Boston tests, or add `CV.size = 0.25` to the R scripts.
  - Remove the threshold override.
  - Add default-path (no `cv_size`) cases to `test_stack.py`.
  - The factor-path test also passes `factor_levels=` and string "0"/"1" chas, which Python
    silently ignores and treats as numeric (STK-07). It passes for the wrong reason.

### STK-02: NNS.boost on a two-valued numeric response (without type="CLASS") returns internal class codes 1/2 instead of the response values (R and Python)
- Type: r-bug (Python replicates it)
- Severity: high (silent wrong numbers on a plausible regression call)
- Confidence: verified
- Where:
  - R `R/Boost.R:389-395`: auto-class only for factor, character or logical.
  - R `R/Boost.R:900-927`: the final `NNS.stack(DV.train = y, type = type)` auto-classifies a
    2-unique numeric y and returns codes.
  - Python `boost.py:189-191` and `:592-620` have the same logic.
- Evidence: `work/STK/api/r_boost_bin.R`. Iris rows 1:140, target `ifelse(setosa, 5, -5)`; the
  truth for rows 141:150 is all -5.
  - R `NNS.boost(...)$results` = `1 1 1 1 1 1 1 1 1 1`.
  - Python `nns_boost(...)['results']` = `[1. ... 1.]`.
  - With a 0/1 target the results are also 1s (a code, not the value 0).
- Which side is right: neither. Boost treats y as regression, its learner trials score SSE on the
  raw values, but the final stack silently switches to classification and its codes are passed
  through as regression estimates.
- Proposed fix:
  - In boost, pass `type = "CLASS"` to the inner stack only when `is_class`. Otherwise pass an
    explicit regression flag, or map `final.stack$class.levels[results]` back.
  - Better still, apply the stack auto-class rule in boost too and decode through `class_values`.
  - This changes public results for binary-numeric boost regressions.

### STK-03: R NNS.stack "ALL" candidate: Method 1 scores a constant training-mean predictor out-of-fold but deploys `NNS.reg(n.best = "all")`, a different, non-constant estimator
- Type: r-bug (Python replicates it)
- Severity: medium-high (model selection is not choosing the model it deploys, and OBJfn.reg is misreported)
- Confidence: verified
- Where:
  - R `Stack.R:1274-1285` (univariate) and `:1352-1366` (multivariate): the `all_raw` OOF prediction
    is `rep(mean(train_y_fit))`.
  - R `Stack.R:1672-1712` and `:1745-1784`: the final fit is `NNS.reg(..., n.best = "all")`.
  - Python `stack.py:960-965` and `:1128-1173` do the same.
- Evidence: `work/STK/ab/run_noise.py` (pure-noise y, n=160). "ALL" wins in every case.
  - Univariate: `OBJfn.reg` = 143.18, the SSE of a constant mean. The returned `reg` has
    sd = 0.227 and is not constant (R and Python agree).
  - Multivariate `method=1`: sd = 0.105, and `NNS.reg.n.best` is reported as 159 (the RPM rows).
- Which side is right: neither. The OOF score must evaluate the estimator that is fitted.
- Proposed fix (both): either score `NNS.reg(n.best = nRPM)` OOF through the same path
  (`NNS_mreg_predict_path_cpp` with `kmax = nRPM`, or point.est on the fold), or make the final
  "all" prediction the constant `mean(y)`. Return `n.best = "all"`/`NA` consistently (see STK-15).

### STK-04: Python class coding ignores factor/Categorical level order and `class_levels`, giving a different (non-ordinal) code map and much worse classifiers than R's factor path
- Type: parity-mismatch / api
- Severity: high
- Confidence: verified
- Where:
  - `stack.py:246-268`: string or object `y` is coded by first appearance.
  - `stack.py:155`: `del class_levels, factor_levels` silently discards the user's level order.
  - `np.asarray(pd.Categorical)` loses the categories.
- Evidence: `work/STK/ab/run_class.py`, `run_class2.py` and `api/api_checks.py`, with iris shuffled
  so that virginica appears first.
  - R factor: levels setosa/versicolor/virginica, OOF accuracy reg/dim = 0.985/0.985.
  - Python: levels virginica/setosa/versicolor, OOF accuracy 0.837/0.815. Test predictions differ
    (`iris_shuf`).
  - Random permutation (seed 4), `CV.size=.3`: R test accuracy 1.00, Python 0.90. Also seen with
    `ts.test=20`.
  - `class_levels=["setosa","versicolor","virginica"]` gives identical output to no levels
    ("class_levels ignored").
  - `pd.Categorical(..., categories=reversed)` gives the first-appearance order.
  - Inconsistency: `nns_boost` sorts string labels (`boost.py:211-212`) and honours `class_levels`.
  - `docs/conventions.md:258-260` says "Use `class_levels=` to reproduce R factor level ordering.
    Raw string labels remain rejected". Both statements are false.
- Which side is right: R's factor path. Python's behaviour equals R's *character* path, which is
  itself an R bug (STK-05).
- Proposed fix:
  - Honour `class_levels`, and take `pd.Categorical`/`Series.cat.categories` order.
  - Otherwise sort labels (as boost and R `factor()` do).
  - This changes results when labels are not in sorted first-appearance order.

### STK-05: R NNS.stack codes a character response by first appearance, so results depend on row order and differ from the factor path and from NNS.boost
- Type: r-bug
- Severity: medium
- Confidence: verified
- Where:
  - R `Stack.R:405-407` (`class_values <- unique(DV.train)`), against `:401-404` for factors and
    `Boost.R:405-410` (`factor(DV.train)`, sorted).
  - The same mismatch exists for character *predictors*: `Stack.R:212` uses `unique(tr_chr)`,
    while `Boost.R:207-210` uses `sort(unique)`.
- Evidence: `work/STK/ab/r_chr.R`, on the same shuffled iris.
  - `factor(y)`: OOF accuracy 0.985, correct labels.
  - `as.character(y)`: levels virginica/setosa/versicolor, OOF 0.815, test predictions include
    "setosa" for virginica rows.
  - `NNS.boost` on the same character y returns sorted levels.
- Which side is right: neither is principled. The coding should not depend on row order, because
  NNS treats codes as ordinal in a regression. Sorted (factor) order at least matches the
  documented "base category of 1" contract.
- Proposed fix: `class_values <- levels(factor(DV.train))` in Stack.R, and `sort(unique(tr_chr))`
  for character predictors.

### STK-06: Last-ulp differences in Spearman `cor` and in the X* matrix product change NNS partitions, so OOF objectives, weights and n.best differ from R
- Type: parity-mismatch (both are fragile; root cause identified and fix verified)
- Severity: medium
- Confidence: verified
- Where:
  - `stack.py:93-100`: `_spearman_signed` uses float64 two-pass Pearson. R `stats:::C_cor` uses
    long-double accumulators.
  - `stack.py:683-686`: `train_design @ coef` (OpenBLAS). R `%*%` is reference-BLAS dgemv,
    column-sequential.
  - `stack.py:799-803`: Method 2 uses the same `@`. R Method 2 uses `NNS_xstar_path_cpp`, which
    accumulates in |coef|-descending order.
- Evidence:
  - `ab/r_fold.R`, `ab/r_fold2.R`, `ab/dump_fold.py` (Boston default call):
    - Fold-3/4 X* differ from R by 5e-17 to 1e-16.
    - R's RPM has 125 rows on R's X* but 124 on Python's X*. R on Python's X* also gives 124,
      so the kernels are identical.
    - The validation-path predictions differ by up to 16.7.
    - Result: `OBJfn.reg` 5.20938 vs R 5.21243.
  - `ab/py_cor.py`: 6 of 13 Spearman coefficients differ by 1 ulp from R.
  - `ab/rcor_emul.py` (long-double emulation of R's `cov.c`) is bit-exact on 13/13.
  - User-visible effects:
    - Boston `dim.red.method="all"`: R n.best 6, Python 7; `reg` predictions differ by up to
      0.748 (RMSE-scale).
    - `NNS.dep`: weights .76/.24 vs .75/.25, stack differs by up to 0.09.
    - Simulated continuous n=500, p=10 (`timing/`): weights .80 vs .79, stack differs by up to 3e-3.
    - Iris `optimize.threshold=FALSE`: n.best 3 vs 1, weights .98 vs .50.
  - Fix validated in `ab/stack_patched.py` and `ab/patched_rerun.py`. Using the emulated cor,
    column-order accumulation for Method 1 folds and |coef|-order accumulation for Method 2/final
    makes every `cor` case bit-identical to R: Boston default/folds3/seed7/L1, iris nothr, the
    factor case and fac3_m12.
- Which side is right: neither, numerically. But R is the reference, and the discontinuity also
  makes R results depend on BLAS.
- Proposed fix:
  - Python parity: port `C_cor` (long-double sums, 2-pass mean) for `_spearman_signed`.
    `scipy.stats.rankdata` gives identical ranks.
  - Replace `@` with column-sequential accumulation (Method 1 folds and `project_xstar`) and with
    |coef|-ordered cumulative sums (Method 2 path and the final projection).
  - Both sides, robustness: round X* (for example `signif(., 12)`) before partitioning, so results
    stop depending on summation order and BLAS.

### STK-07: Factor predictors: numeric-looking strings are silently treated as numbers, `factor_levels` is ignored, and level order is first appearance (R: one-hot dummies in levels() order)
- Type: parity-mismatch / api
- Severity: medium
- Confidence: verified
- Where:
  - `stack.py:453` (`_numeric_column` check) and `:456-459` (first-appearance levels).
  - `stack.py:155`: `factor_levels` deleted.
- Evidence: `ab/run_factor.py`.
  - chas given as "0"/"1" strings: Python encodes 13 columns. R `factor(chas)` encodes 14
    (`chas__X0`, `chas__X1`).
  - "river"/"inland" strings: both have 14 columns, but in a different order (sorted vs appearance).
  - `ab/run_fac2.py`: a 3-level factor {zeta, alpha, mid}. `NNS.dim.red.coefficients` order
    differs: R `[x1, g__alpha, g__mid, g__zeta, x2]`, Python `[x1, zeta, alpha, mid, x2]`.
  - In `fac3_m12` the column order plus STK-06 shifted the weights (.79 vs .78). With the STK-06
    patch they match.
- Which side is right: R. A user who passes a categorical column expects a factor.
- Proposed fix:
  - Accept `factor_levels`/pandas categorical dtypes to force factor treatment and level order.
  - Sort levels for string columns (as `factor()` does).
  - Return coefficient names.
  - Cross-ref REG-12 and REG-21 (the same issue in `nns_reg`).

### STK-08: pandas column names are discarded: test columns are matched by position (silently wrong), and boost and stack outputs carry index labels instead of names
- Type: api
- Severity: medium-high
- Confidence: verified
- Where:
  - `stack.py:212` and `:292` (`np.asarray`).
  - `stack.py:861-863`: `NNS.dim.red.variables` is a list of 0-based ints.
  - `boost.py:554`: `feature_names = x1..xp`.
- Evidence: `work/STK/api` (inline run). mtcars `[wt, hp, disp]` trained, test passed as
  `[disp, hp, wt]`:
  - R realigns by name and returns identical predictions (22.45 29.63 ...).
  - Python silently returns 2714.95, 2095.4, 8579.71, ... mpg.
  - Boost: R `feature.frequency` names are `lstat, crim, rm`; Python gives `x13, x1, x6`
    (`ab/run_boost.py`).
- Which side is right: R. Its docs promise name matching.
- Proposed fix: when inputs are DataFrames, keep the columns, reorder the test frame by name
  (error on a partial overlap, like R), and return names in `NNS.dim.red.variables`,
  `NNS.dim.red.coefficients` and boost `feature.*`. Cross-ref REG-03.

### STK-09: R `.balance_indices` / `.balance_training` use `sample(g, k)`; when a class has a single member in the fitting split, R samples from `1:g` and drops that class
- Type: r-bug (Python is right)
- Severity: medium (it hits exactly the rare-class case that `balance=TRUE` exists for)
- Confidence: verified
- Where:
  - R `Stack.R:596-602` and `Boost.R:564-567`.
  - Python `stack.py:437-440` and `boost.py:340-341` correctly sample within the group.
- Evidence:
  - `ab/r_bal_proof.R`: groups `{1,2,3}/{4,5,6}/{7}`. The down-sample "class 3" pick is row 1
    (class 1); the up-sample picks are rows 3, 6, 2 (classes 1, 2, 1). Class 3 vanishes from the
    "balanced" set.
  - `ab/run_bal1.py`: 3 classes (29/29/2), `balance=TRUE`. R OOF 0.85, n.best 2, threshold .29;
    Python 0.817, 1, .86. The streams diverge after the first such draw.
- Which side is right: Python.
- Proposed fix (R): use `g[sample.int(length(g), k, replace = ...)]`.

### STK-10: Tie-sensitive `nns_dep` mismatch (DEP-08/09) propagates into stack and boost: balanced or discrete classification, `dim.red.method="NNS.dep"/"all"`, and boost learner scores
- Type: parity-mismatch (downstream of DEP)
- Severity: medium
- Confidence: verified
- Where:
  - Univariate `nns_reg` default order: `order=NULL` is derived from `nns_dep`.
  - `stack.py:619-626` (dep coefficients).
- Evidence:
  - Iris `balance=TRUE` (the vignette 08 call): R `OBJfn.dim.red` 0.9643, weights .85/.15 (as
    printed in the vignette); Python 0.9571, .88/.12.
  - `ab/cmp_dbg.py iris_class_bal`: splits, balance draws and X* inputs are bit-identical, but
    the fold-1 m=3 univariate fit differs by 0.745.
  - `ab/uni/`: on the dumped fold data R `NNS.reg` gives 62 regression points, Python 33.
    `NNS.dep` is 0.4463 vs 0.3305 (216 rows, 107 unique x, 3-valued y). Fixed orders 1..5 and
    "max" agree.
  - Boston boost `threshold=1.5` (`ab/run_boost2.py`, `r_thr.R`): 3 of 15 learner SSEs differ.
    The one-feature `ptratio` learner (43 unique values) has `NNS.dep` 0.5011 in R vs 0.4425 in
    Python, so the feature frequencies differ (crim 9 vs x1 10, ...).
  - Binary iris and mtcars `am`: weights .44 vs .26, threshold .83 vs .78.
- Which side is right: see DEP-08/09 (summation-order and exact-compare fragility). The fix
  belongs in `nns_dep`.
- Proposed fix: fix it in DEP. Add stack and boost regression tests on tied data (iris, ptratio).

### STK-11: NNS.boost is about 3.5-4.5x slower than R; most of the time goes to the pure-Python RPM build and a 3-D broadcast distance tensor
- Type: perf
- Severity: medium
- Confidence: verified
- Where:
  - `_reg_engine.py:1103-1131` (`_mreg_build_rpm`: per-group, per-column `_reduce_value` →
    `nns_gravity` → `np.median`, 377k calls).
  - `_reg_engine.py:904-926` (`_mreg_distances` builds an (m, n, p) temporary).
  - `_reg_engine.py:929-948` (per-row `_stable_topk` loop).
  - `_reg_engine.py:1008-1050` (per-row k=1 loop and ensemble weights recomputed for every k).
- Evidence: `work/STK/timing/` (single thread; R `ncores=1`; Python warm):

  | task | R | Python |
  |---|---|---|
  | stack n=500 p=5 | 0.23 s | 0.31 s |
  | stack n=500 p=10 | 0.38 s | 0.34 s |
  | stack n=2000 p=5 | 0.50 s | 0.65 s |
  | stack n=2000 p=10 | 0.86 s | 0.85 s |
  | stack n=10000 p=5 | 2.68 s | 2.66 s |
  | stack n=10000 p=10 | 2.99 s | 3.56 s |
  | boost 50/50 n=500 p=5 | 1.87 s | 3.01 s |
  | boost 50/50 n=500 p=10 | 4.26 s | 3.65 s |
  | boost 50/50 n=2000 p=5 | 5.03 s | 18.1-19.6 s |
  | boost 50/50 n=2000 p=10 | 10.7 s | 46.9 s |
  | boost 20/20 n=10000 p=5 | 31.9 s | 135.2 s (results equal to 1.8e-15) |

  - The first call in a fresh process adds about 1 s: 0.3 s import plus 0.7 s of lazy scipy.
  - Profile (`timing/prof.py 2000 5 boost`, 32.6 s total): `_mreg_build_rpm` 15.7 s cumulative,
    `_mreg_distances` 10.1 s, `_stable_topk` 2.1 s, `nns_dep`/`nns_part`/`_copula` 2.4 s.
  - `timing/speedup_test.py` monkeypatches two changes and gets **19.6 s → 8.0 s with
    bit-identical boost results**:
    - Column-accumulated distances (`acc += |z_j| + z_j²` per column). On a 400x3000x10 case this
      alone is 14x faster (1.24 s → 0.087 s, max diff 0).
    - A singleton-group shortcut in `_mreg_build_rpm` (gravity of one value is that value).
- Proposed fix (concrete, in order):
  1. Column-accumulated distances, or bind R's `NNS_mreg_predict_path_v2_cpp` / `NNS_mreg_setup_cpp`
     (the vendored core has no mreg kernel; REG-06).
  2. In `_mreg_build_rpm`, only reduce groups of size > 1, and use the native `_nnscore.gravity`
     for them.
  3. `_mreg_predict_path`: vectorise the k=1 tie gravity and the k-loop of
     `_mreg_ensemble_weights`. It is currently O(kmax²·m) with scipy `t/expon/lognorm` pdf calls
     per k; precompute the rank-only terms once.
  4. `_rank_average` (`stack.py:61`) is a Python while-loop. `scipy.stats.rankdata` is identical
     and 2-6x faster (0.6 s of 3.6 s in stack n=10000).
  5. Method 2 refits identical projections. R dedupes via `representative` in
     `NNS_xstar_path_cpp`. Reuse the fit when the cumulative coefficient adds zero (numeric
     `dim.red.method` with zeros, zero-cor dummies).

  None of these change results except (1) if a pairwise-sum order changes. The column order
  reproduces R's sequential sum.

### STK-12: Python docs and docstrings describe a previous design (NumPy RNG, deterministic cv_size=0.25 default, rejected string labels, working `class_levels`/`factor_levels`)
- Type: docs
- Severity: medium
- Confidence: verified
- Where:
  - `stack.py:1-16`: "random fold assignment ... use a local numpy.random.Generator ... match R
    distributionally, not bit-for-bit". The code uses `RRNG` and matches bit-for-bit.
  - `docs/conventions.md:254-297` and `:299-347`:
    - "NNS Python uses NumPy's RNG".
    - "CV.size = NULL ... NNS Python uses a deterministic default of 0.25". False: the default is
      5-fold like R.
    - "ts_test ... CV training uses the tail rows". False: it is rolling chronological.
    - "Raw string labels remain rejected". False: they are accepted.
    - "Use class_levels= to reproduce R factor level ordering". Ignored by stack.
    - "factor_levels= ... supported". Ignored.
    - Boost "n_features>10 ... NotImplementedError guard". No longer exists.
  - `docs/api_status.md:53` ("exact stochastic sample parity is not expected") and `:70-71`.
  - `StackResult` doc (`stack.py:45-54`): 0-based indices instead of R's names. This is
    documented, but it diverges from R.
- Proposed fix: rewrite those sections to the current behaviour. Make `class_levels` and
  `factor_levels` work, or raise when they are passed.

### STK-13: R documentation does not match code (NNS.stack / NNS.boost / vignette 08)
- Type: docs (R)
- Severity: low-medium
- Confidence: verified
- Where and evidence:
  - `man/NNS.stack.Rd:61` and `Stack.R:20`: "`dim.red.method = "NNS.dep"` (default)". The usage
    and code default is `"cor"`. "cor uses standard linear correlation", but the code uses
    Spearman (`Stack.R:815`).
  - `Stack.R:13` CV.size: "Defaults to a random value between 0.2 and 0.33". With
    `CV.size=NULL` and `folds>1` it is stratified k-fold CV. The random size applies only when
    `folds=1` (`Stack.R:708-711`).
  - `Stack.R:55-59` note about "object 'RP' not found ... reduce CV.size" is stale (that code no
    longer exists).
  - `man/NNS.boost.Rd:57` and `Boost.R:18`: `objective` "max" (default). The code default is
    "min". `threshold` is documented as [0,1] but not validated: 1.5 or -0.2 are silently
    clamped by LPM.VaR.
  - `Boost.R:21` `feature.importance` "(default) TRUE draws a plot": it plots even inside
    scripts and is the reason R boost opens a device.
  - Vignette 08 line 99 says "For brevity, we set `folds = 1`", but the code uses `folds = 5`.
    The printed vignette numbers (OBJfn.dim.red 0.9642857, weights .85/.15) are R 13.3 output;
    Python differs (STK-10).
- Proposed fix: correct the docs. Validate `threshold` in [0,1].

### STK-14: R dead or duplicated code on the stack path
- Type: dead-code (R)
- Severity: low
- Confidence: verified
- Where:
  - `Stack.R:1686-1692`: the univariate "all" branch fits `.fit_univariate_raw(...)` and
    immediately overwrites the result with a second `NNS.reg` fit (a wasted full fit).
  - `Stack.R:1777-1784`: re-runs `.nns_mreg_prepare_model` on the full data only to count RPM rows.
  - `Stack.R:1579`: `if (is_class) 0.5 else 0.5`.
  - `src/table_ops.cpp` `NNS_mreg_reduce_cpp` is exported but never called from R. Its reducer
    codes (0 gravity, 1 mean, 2 median, 3 mode) contradict `.nns_mreg_prepare_model`'s
    `reducer.code` (mean=0, median=1, mode=2, off=3) used for `NNS_mreg_setup_cpp`. Wiring it in
    would silently swap reducers.
- Proposed fix: delete these, or align the codes.

### STK-15: `NNS.reg.n.best` when "ALL" wins: R returns NA (univariate) or nrow(RPM) (multivariate); Python returns n (univariate)
- Type: parity-mismatch / api
- Severity: low
- Confidence: verified
- Where: R `Stack.R:1535-1537` (`reg_best_k <- NA` and never set in the univariate branch) and
  `:1781`. Python `stack.py:1148-1149`.
- Evidence: `ab/run_noise.py` `noise_uni`: R NA, Python 160. Multivariate: both 159 or 33.
- Which side is right: neither is informative. Return `"all"` (or NA plus a flag) in both branches.
- Proposed fix: add a `NNS.reg.n.best.label`, or return "all". Make Python match R's choice.

### STK-16: `_rrng.sample_int_prob_noreplace` does not reproduce R (no descending revsort); it is dead code
- Type: python-bug / dead-code
- Severity: low (currently unused)
- Confidence: verified
- Where: `_rrng.py:126-157`. R's `ProbSampleNoReplace` first `revsort`s the probabilities.
- Evidence: `work/STK/rng/rng_check.sh`, `set.seed(42); sample.int(5,3,prob=c(.1,.5,.2,.15,.05))`:
  R gives `1 4 2`, Python `4 3 2`. Already-sorted probabilities agree (`4 3 1`).
- Proposed fix: revsort (stable descending, permuting ids) and use `totalmass=1` after
  normalising. Otherwise delete it. Also note `RRNG` accepts seeds > 2^31-1, which R rejects.

### STK-17: Dead code in `categorical.py` and `stack.py`
- Type: dead-code
- Severity: low
- Confidence: verified
- Where:
  - `categorical.py:80-147`: `_down_sample_rows`, `_up_sample_rows`, `_balance_class_training`
    and `_dense_factor_codes` are used only by `tests/invariants/test_sampling.py`. They use the
    NumPy RNG and an R `downSample`/`upSample` layout that stack and boost no longer use.
  - `stack.py:84-90`: `_spearman_scores`, "kept for nns.var imports", but nothing imports it.
  - Stack and boost duplicate `_scalar_integer` and `_scalar_logical`.
- Proposed fix: delete them, or move them to a shared helper.

### STK-18: Small API rough edges
- Type: api
- Severity: low
- Confidence: verified (`work/STK/api/api_checks.py`)
- Where and evidence:
  - An `obj_fn` that returns a vector raises a cryptic
    `TypeError: only 0-dimensional arrays can be converted`. R gives "[obj.fn] must return one
    numeric scalar." (`stack.py:355`, `boost.py:270`)
  - `random_seed=` silently overrides an explicit `seed=123` (`stack.py:156`, `boost.py:101`):
    `seed=123, random_seed=5` equals `seed=5`.
  - `feature_importance` (boost) and `ncores` are accepted and silently ignored
    (`boost.py:99`, `stack.py:155`). They are not documented as no-ops.
  - `status=True` prints to stdout via `print`, and the messages differ from R's.
  - A two-valued numeric `y` auto-switches `nns_stack` to classification and returns codes 1/2
    (R does the same; documented only indirectly). Combined with STK-02 this surprises
    regression users.
  - With `seed=None` Python draws OS entropy, while R continues the global stream and boost
    passes `seed=NULL` to the inner stack. This is fine, but Python boost passes the drawn seed
    to the inner stack, so the stream structure differs from R.

---

## Coverage (checked and found matching R, max abs diff about 1e-14 unless noted)

- **Boston (mlbench split, 13 numeric):**
  - method 1, method 2, (1,2) with stack and without.
  - pred.int=0.9.
  - dim.red.method "equal", "NNS.caus" (final outputs), and numeric coefficients.
  - folds 1/3/5, seed 7.
  - order 1 and "max"; dist L1.
  - CV.size 0.25; factor chas.
  - With the STK-06 patch, every `cor` case is bit-exact including OBJfn.
- **R RNG emulation:**
  - `set.seed`, `runif`, `sample`, `sample.int` with and without replacement, n up to 1e6, and
    seed 2^31-1 match R exactly.
  - Stack fold ids, stratified folds, repeated holdouts, `folds=1` random CV size and balance draws
    are identical to R (verified fold by fold in `ab/cmp_dbg.py`).
  - Boost holdouts, subset sampling and epoch schedules match.
- **Simulated nonlinear data** (`ab/run_misc.py`, n=300, p=4): default, ts.test=30, ts.test=50
  with folds=2, folds=1 with CV.size=.3, objective="max" with a custom negative SSE, MAE obj.fn,
  numeric dim.red.method, method 2 with pred.int, method 1 with pred.int, other seeds, and a
  single predictor (method 2 dropped with a warning).
- **Finance** (`ab/run_fin.py`: 8 lagged ETF features, last 1000 days, SPY next-day sign):
  - Auto-class 0/1, with ts.test=100 and folds=3, and the return regression with MAE all match
    exactly.
  - Hit rate 0.47 equals the base rate. The optimized threshold 0.20 means "always up" (both
    sides), so there is no predictive skill, as expected.
- **Classification:**
  - Iris CLASS (rows 1:140): default, pred.int, CLASS levels and decoding.
  - mtcars `am` (outputs equal; OOF weights differ per STK-10).
  - Classification `pred.int` decoding.
- **NNS.boost** (`ab/run_boost.py`, `run_boost2.py`, `run_mt.py`; results equal to 5e-14 and
  feature frequencies equal up to naming):
  - iris balance and no-balance (the vignette 08 call matches R output exactly).
  - Boston regression with pred.int, CV.size, ts.test, depth="max", dist L1, epochs=0, extreme,
    threshold=-0.2, seed 9.
  - mtcars regression.
  - `features_only` (structure).
- **Timing:** stack is at parity with R once warm. Boost is not (STK-11).
- **Fixture special-casing:** no fixture-specific branches in stack.py, boost.py or _rrng.py. The
  only fixture issues are in the tests (STK-01): the cv_size mismatch, the threshold override,
  and a weak iris stack test that asserts only that the output is constant.
- **`nns_stack_fit`/`nns_stack_predict`:** identical to a one-shot `nns_stack` for regression and
  for classification with and without balance.
- **Input validation messages** (NaN in X or y, bad method/folds/objective, unseen test levels,
  1-row test vector, constant column) match R semantics.
