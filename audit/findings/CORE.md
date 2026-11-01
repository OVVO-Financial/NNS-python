# CORE group findings: C++ layer (R 13.3 src, NNS-core, vendored core, nanobind bindings)

Scope: `/home/user/NNS-core` (src, original_src, sync, scripts, tests, .github), the vendored
`/home/user/NNS-python/extern/NNS-core`, `/home/user/NNS-python/src/nns/{_nnscore_bindings.cpp,_native.py}`
plus `CMakeLists.txt`, and the R 13.3 sources in `/home/user/ovvo-financial/nns/src` (commit 82a92eb).
All scripts and outputs are in `/home/user/audit/work/core/`.

Where another group reported the same thing at the Python/R API level, this file says so (PM-13,
PM-18, PM-19, REG-06, REG-11). The entries here add the C++ root cause and extra evidence.

## Task answers (short)

**T1. Build, smoke test and fidelity.** NNS-core builds cleanly into `work/core/build`
(Release, `-DNNSCORE_BUILD_TESTS=ON -DNNSCORE_BUILD_FIDELITY=ON`). `ctest` reports 1/1 passed.
`smoke_partial_moments` is only 3 checks, all on `lpm`/`upm` at degree 1. I generated truth from the
installed R 13.3 (`generate_truth.R` -> `work/core/truth.json`) and ran `run_fidelity.py`:
**28/28 passed** (`work/core/fidelity_report.md`). The suite only covers scalar LPM/UPM (degrees 0-2,
which use the direct path) and Co/D partial moments at degree 0/1 with equal degrees. Extra checks I
added:
- `ab_native.py`/`.R`: 261 checks of the bound kernels against R 13.3. They cover unequal and fractional
  co-moment degrees, more than 32 targets, nD at degrees 0/1/2/0.5 with ties, `clpm_nd_batch`,
  `PM.matrix` in 6 configurations, `fast_lm`, gravity, mode, `generate.*vectors`, stochastic
  superiority and `is.discrete`. Everything matches except `fast_lm` on offset data (CORE-14). The
  apparent `generate_lin_vectors` mismatch was a harness artefact (row/column flattening); I checked it
  directly and it matches.
- `core_fid/`: NNS-core `partition` (4 noise modes x XY/XONLY), `seasonality`, `fsd/ssd/tsd_uni` and
  `rescale` agree exactly with the R 13.3 kernels on returns, discrete and AirPassengers data.
- The exception is `nns::dep_matrix`, which is stale (CORE-04).
- `asan/fuzz.cpp` ran an ASan+UBSan sweep. It found 2 memory/UB defects (CORE-07, CORE-12) and 2
  crash or OOB paths in the vector generators (CORE-13).

**T2. R 13.0 snapshot (`original_src`) vs R 13.3 `src`.** After normalising CRLF line endings, the
following files are unchanged: `fast_lm.cpp`, `partial_moments.cpp`, `SD.cpp`, `NNS_seas.cpp` and
`stoch_sup.cpp`. Changed files:

| R file | 13.0 -> 13.3 change | NNS-core status | Python matters? |
|---|---|---|---|
| NNS_dep.cpp | discordant independence anchor 0.75 -> 0.5 (13.1) | stale (0.75) | no: Python binds its own `nns_dep_pair`, which already uses 0.5 |
| NNS_distance.cpp | `compute_distances` fixed from row-major to column-major indexing (a 13.0 R bug in `NNS_distance_path_cpp` and `NNS_distance_bulk_cpp`) | NNS-core was already correct | no (Python distance is pure NumPy) |
| NNS_part.cpp | `R_finite` -> `!ISNAN`, so +-Inf are now treated as values | stale; the binding `nns_part_off` is stale too | yes, parity on Inf input (CORE-11) |
| central_tendencies.cpp | gravity refactored into a shared `gravity_value()`; no numeric change | same numbers | no |
| internal_functions.cpp | `generate.vectors(len=)` buffer reuse for ARMA | missing | no (Python ARMA does not use it) |

Files that are new in 13.3, or present in 13.0 but missing from the snapshot:

| R file | content | NNS-core / NNS-python | Python matters? |
|---|---|---|---|
| NNS_mreg_predict.cpp (13.1-13.2) | M.reg prediction (single k, all k up to k_max, threaded, partial selection, 1-D fast path) | absent; Python reimplements it in NumPy (`_reg_engine._mreg_predict*`) | **yes, perf** (CORE-02): 14-25x slower kernel, `nns_stack` method 1 is 7.6x slower |
| NNS_mreg_setup.cpp (13.1) | findInterval IDs + grouped reduce; duplicate-column map | absent; NumPy `_mreg_partition_matrix`/`_mreg_build_rpm` | **yes**: perf (about 40% of `nns_stack` time) and parity (its `mode` reducer differs, CORE-03) |
| NNS_stack_fast.cpp (13.1) | X* projection path | absent; NumPy `project_xstar` | minor |
| nns_reg_points.cpp (in 13.0, not in snapshot) | NNS.reg regression-point builder | dead in R (never called); the binding `nns_reg_mv` is a separate port of the Python code | no |
| table_ops.cpp (in 13.0) | `NNS_mreg_reduce_cpp` | dead in R | no |
| partial_moments.h / partial_moments_rcpp.{cpp,h} / central_tendencies.h (in 13.0) | prefix backend, wrappers (default target = mean), workers | prefix backend ported; `_rcpp` wrappers are on the sync `ignore` list | yes (CORE-01, CORE-05) |

**T3. NNS-core vs vendored copy.** `extern/NNS-core` is byte-identical to NNS-core at `cfc25a3`, the
`core_commit` recorded in `NNS-python/sync/nns_source.json`. NNS-core HEAD differs only by the
`// ---- NNS-CORE PROVENANCE ...` comment block that `sync_adapter.py --init` inserted into
`src/partial_moments.cpp` and `include/nns/partial_moments.hpp` (commit 400b8fc). It also adds a
trailing newline and the `NNSCORE_BUILD_FIDELITY` CMake option. There is no functional difference.

**T5. Native vs pure Python.**
- Python calls these natives: `lpm`/`upm`/`lpm_ratio_v`/`upm_ratio_v` (`core.py`), `co_*_v`/`d_*_v`
  (`co_moments.py`), `pm_matrix`, `copula_nd`, `nns_dep_pair`, `gravity_exact` (dependence), `gravity`,
  `mode`, `nns_part_off`, `nns_reg_mv`, `fast_lm` (ARMA, meboot), `stochastic_superiority`,
  `clpm_nd_batch` (cdf).
- Bound but never called: scalar `co_lpm`/`co_upm`/`d_lpm`/`d_upm`, `clpm_nd`/`cupm_nd`/`dpm_nd`,
  `fast_lm_mult`, `is_discrete`, `vec_sd`, `col_sd`, `factor_2_dummy(_fr)`, `generate_vectors`,
  `generate_lin_vectors`.
- NNS-core kernels with no binding (Python reimplements them in NumPy): dependence, distance*,
  partition, seasonality, SD (fsd/ssd/tsd/efficient set), rescale, meboot_*, up/down sample and ARMA
  weighting.
- Timings show these NumPy versions are acceptable (SD efficient set 17-22 ms vs R 7-8 ms; `nns_part`
  2-4x R; gravity and mode faster than R). The large gap is M.reg/stack, where the R kernels do not
  exist in NNS-core at all (CORE-02).

---

## Findings (ordered by severity)

### CORE-01: Vectorised LPM/UPM prefix-power backend (>32 targets) suffers catastrophic cancellation: negative and wildly wrong moments in R, NNS-core and Python
- Type: r-bug (Python inherits it through the native kernel). Same issue as PM-13; this entry adds the C++ root cause and the native-vs-fallback split.
- Severity: high
- Confidence: verified
- Where:
  - R `src/partial_moments.h:120-190` (`PrefixPartialMomentBackend::lpm/upm`; `suffix = total_power - prefix_power[k]` at :166 and :183; the binomial expansion).
  - NNS-core `src/partial_moments.cpp:129-177`, with the switch at :255 (`n_targets > 32`).
  - Python `core.py:52`.
- Evidence (`work/core/prefix_stability.R`, `prefix_stability.py`, `fallback_ab.py`):
  - SPY prices, degree 8, 100 targets: max relative error 2.9e3, and one negative UPM. Python
    `nns.upm(8, tz, p)` min = -4,724,413. At t=772.36 the result is 6,182,957 vs the true 2,122.0
    (the true value is returned when the same target is passed with 32 or fewer targets).
  - Degree 4: relative error 1.5e-5.
  - Degree 2 with targets just below the sample maximum: R returns -1.35e-17 (negative).
  - Daily returns at degree 12: relative error 2e-7.
  - The Python pure-NumPy fallback (used when `nns._nnscore` is missing) gives the correct value, so
    native and fallback disagree by 2e7.
- Which side is right: a partial moment is a mean of non-negative terms. The direct sum (scalar path,
  or 32 or fewer targets) is right. The result must not depend on how many targets are requested.
- Proposed fix:
  - Use the prefix backend only for degree <= 2.
  - Build suffix sums from the top end (a separate reverse cumulative sum) instead of `total - prefix`.
  - Clamp the result at >= 0.
  - For degree >= 3, or when `|t - shift|` exceeds the data spread, use the sorted direct loop (sum
    only over x <= t, or x > t).
  - This changes public results only where they are currently wrong.

### CORE-02: The M.reg / NNS.stack native kernels from R 13.1-13.3 are missing from NNS-core; the Python NumPy port is 14-25x slower and `nns_stack` is 7.6x slower than R
- Type: perf. Root cause of part of REG-06 and STK-11.
- Severity: high
- Confidence: verified
- Where:
  - R `src/NNS_mreg_predict.cpp` (`NNS_mreg_predict_v2_cpp`, `NNS_mreg_predict_path_v2_cpp`) and `src/NNS_mreg_setup.cpp`.
  - Python `_reg_engine.py:953-1050` (`_mreg_predict`, `_mreg_predict_path`, which runs a scipy.stats ensemble per k) and `:1063-1130` (partition/RPM build).
- Evidence:
  - `mreg_path_ab.py`/`.R` (n_RPM=1500, p=3, 600 queries, k_max=30): Python path 1.077 s vs R kernel
    0.044 s; single k=12: 0.208 s vs 0.015 s. Results agree to 4e-15.
  - `stack_time.py`/`.R` (Boston, 400 train / 106 test, `folds=1`, method 1): Python 1.30 s (0.74 s
    warm) vs R 0.17 s.
  - `stack_prof.py`: `_mreg_partition_matrix` + `_mreg_build_rpm` take 0.55 s of 0.74 s, mostly 6,333
    per-group `nns_gravity`/`np.median` calls. R does this in `NNS_mreg_setup_cpp`.
- Which side is right: both give the same numbers; Python is just slow.
- Proposed fix:
  - Port `NNS_mreg_setup_cpp` (with a consistent mode reducer, see CORE-03) and
    `NNS_mreg_predict(_path)_v2` into NNS-core as pure C++, using `std::lgamma`-based t, normal and
    lognormal densities instead of Rmath.
  - Bind them, and call them from `_mreg_predict`, `_mreg_predict_path` and `_mreg_build_rpm`.
  - Public results are unchanged up to floating-point rounding.

### CORE-03: The native `NNS_mreg_setup_cpp` "mode" reducer is "most frequent exact value", but R's own fallback and Python use `NNS.mode`; NNS.reg results change with `options(NNS.native.mreg)`
- Type: r-bug + parity-mismatch. Same symptom as REG-11; this is the C++ root cause.
- Severity: medium
- Confidence: verified
- Where:
  - R `src/NNS_mreg_setup.cpp:14-22` (`reduce_numeric`, `reducer==2`).
  - R fallback `R/Regression.R:347` (`mode = mode(z)`, which is `NNS.mode(multi=FALSE)`, `Central_tendencies.R:24`).
  - Python `_reg_engine.py:358`.
- Evidence (`mreg_mode.R`, `mreg_mode2.R`): `NNS.reg(X, y, noise.reduction="mode")` on rounded 2-D
  data.
  - R native: R2 0.4974. R with `NNS.native.mreg=FALSE`: R2 0.4349. Max |y.hat diff| = 5.59.
  - Python matches the R fallback to 3.6e-15.
  - On `z=c(0.1,0.1,0.2,0.25,...)`, NNS.mode gives 0.2485 while the native reducer gives 0.1.
- Which side is right: the package defines "mode" as `NNS.mode` everywhere else. The C++ shortcut
  silently changes the estimator, so Python and the R fallback are right.
- Proposed fix: in `reduce_numeric`, call the same continuous mode as `NNS_mode_cpp`
  (`multi=false`). Keep the exact-frequency mode only for `is_class`. This changes R's public
  results for `noise.reduction="mode"`.

### CORE-04: NNS-core is frozen at R 13.0 and the R -> NNS-core -> NNS-python sync chain is broken in both directions
- Type: docs / dead-code (process)
- Severity: medium
- Confidence: verified (workflow and code reading, plus a numerical check)
- Where:
  - `NNS-core/sync/nns_r_source.json` (`r_commit 905b8bb`, `r_version 13.0`; that commit is not in the R clone's history).
  - `NNS-core/src/dependence.cpp:89,125` (`indep_D = 0.75`).
  - R `.github/workflows/downstream-sync-dispatch.yml` dispatches only `nns-r-package-updated` to NNS-python. Nothing sends `nns-r-src-updated` to NNS-core, which contradicts R `.github/downstream-sync.md`.
  - NNS-core `notify-python.yml` dispatches `nns-core-updated` to NNS-python's `sync-from-nns-core.yml`. That file does not exist, and `workflow-baseline.yml` deletes any workflow outside its list of 5 from same-repo PRs.
- Evidence:
  - `core_fid/depm.cpp` vs `NNS:::NNS_dep_matrix_cpp` on 3 independent N(0,1) columns (n=1000):
    NNS-core dependence 0.46-0.47 vs R 13.3 0.30-0.35. On SPY/TLT/GLD: 0.466/0.477/0.495 vs
    0.406/0.380/0.351.
  - The 13.2 NNS_part Inf change (CORE-11) is not ported either.
  - `NNS-python/sync/nns_source.json` records `r_commit 82a92eb / r_version 13.3`, but the vendored
    native core comes from 13.0. The notes admit this, yet the fields read as if the native layer
    were 13.3.
- Which side is right: R 13.3 is authoritative.
- Proposed fix:
  - Add an `nns-r-src-updated` dispatch for `src/**` in the R repo.
  - Add a `sync-from-nns-core.yml` receiver to the NNS-python baseline list.
  - Port the 13.1-13.3 changes to NNS-core (dep 0.5 anchor, part `!ISNAN`).
  - Add a separate `native_r_commit` field to `nns_source.json`.

### CORE-05: The sync "auto" path marks partial-moment changes ready after only a provenance-comment rewrite and 28 scalar fidelity cases
- Type: docs (process / CI)
- Severity: medium
- Confidence: verified (code reading of `scripts/sync_adapter.py:168-189`, `sync/r_src_map.json`, `tests/fidelity/generate_truth.R`)
- Where:
  - `sync/r_src_map.json` marks `src/partial_moments.cpp` and `src/partial_moments.h` as `auto: true, fidelity_covered: true`.
  - `src/partial_moments_rcpp.cpp` is on the `ignore` list as "generated Rcpp glue". It is not generated: it holds `LPM.ratio`/`UPM.ratio`/`D.LPM`/`D.UPM`/`PMMatrix_RCPP` wrapper logic, including the target=mean default (see PM-01).
- Evidence:
  - For an "auto" file the adapter only rewrites the `// NNS-R-SOURCE:` line. It never ports code.
  - The fidelity truth exercises only scalar LPM/UPM (direct path) and Co/D at equal degrees 0/1.
  - So any change to the prefix backend, the nD kernels, the PM matrix or fractional-degree branches
    would yield a non-draft "ready" PR claiming sync while the C++ logic is unchanged. Changes to
    `partial_moments_rcpp.cpp` never request a port.
- Proposed fix:
  - Set `auto: false` until the fidelity suite covers more than 32 targets, the nD functions, the PM
    matrix, fractional and unequal degrees, and ties.
  - Remove `partial_moments_rcpp.*` from `ignore`.
  - Have the fidelity gate compare a hash of the ported files with the R files, so a semantic R change
    with an unchanged core is refused.

### CORE-06: `nns::parallel_for` spawns fresh `std::thread`s on every call and the bindings always use all cores: small native calls are 10-19x slower than serial, and `ncores` is ignored
- Type: perf. Same as PM-19; this entry adds measurements.
- Severity: medium
- Confidence: verified
- Where:
  - NNS-core `include/nns/parallel.hpp:21-66`, which creates and joins threads per call (no pool).
  - Every `n_threads` default is `-1`. `_nnscore_bindings.cpp` never passes `n_threads` and never releases the GIL.
- Evidence (`core_fid/bench_threads.cpp`, `thread_overhead.py`, 4-core host):
  - `pm_matrix`+`clpm_nd` at n=200, d=3: 19.4 us serial vs 372 us default.
  - n=400, d=14: 159 us vs 490 us.
  - n=5000, d=14: 2058 us vs 1920 us (no real gain even there).
  - Python `co_lpm_v` with 2 targets, n=200: 47.5 us vs 2.8 us for two scalar calls.
  - `clpm_nd` (3 thread fan-outs): 407 us per call.
- Proposed fix:
  - Run serially below a work threshold (e.g. n*d*targets < 2e5).
  - Use a persistent pool, or OpenMP with a fixed team.
  - Expose `n_threads` in the bindings and honour Python `ncores` (default 1).
  - Add `nb::call_guard<nb::gil_scoped_release>()` on the heavy kernels.
  - Public results are unchanged.

### CORE-07: `NNS.mode(discrete=TRUE)` returns -2147483648 for values beyond the int range (R, NNS-core and Python)
- Type: r-bug (Python inherits it)
- Severity: medium
- Confidence: verified
- Where:
  - R `src/central_tendencies.cpp:287-288` (`std::unordered_map<int,int> freq; ++freq[nearest_int_half_up(v)]`).
  - NNS-core `src/central_tendencies.cpp:289-291`.
  - Python `central_tendencies.py:63`.
- Evidence (`ct_edge.R`, `ct_edge.py`, ASan/UBSan `asan/fuzz.cpp`):
  - `NNS.mode(c(3e9,3e9,3e9,1,2), discrete=TRUE)` gives -2147483648. `c(-5e9,...)` gives the same.
  - UBSan reports "3e+09 is outside the range of representable values of type 'int'".
  - Typical exposure: volumes, IDs, nanosecond timestamps.
- Proposed fix: key the map on `long long`, or on the rounded double, and return doubles. This
  corrects public results.

### CORE-08: Gravity and mode binning has no cap on the number of bins: a tight IQR plus one outlier takes seconds and GBs (R, NNS-core, Python), and the `int` bin count can overflow (UB)
- Type: r-bug + perf
- Severity: medium
- Confidence: verified
- Where:
  - R `src/central_tendencies.cpp:92-98` (`simple_bin_counts`: `nbins = (int)floor((xmax-origin)/width)+1`, then `resize(nbins)`).
  - NNS-core `src/central_tendencies.cpp:83-89`.
  - The binding's `gravity_exact` caps at 1e6 bins (`_nnscore_bindings.cpp:323`), so the two Python gravity implementations can disagree on such data.
- Evidence (`ct_edge.R`/`.py`):
  - `x <- c(5 + 1e-7*rnorm(1000), 6)`: `NNS.gravity` takes 12.8 s and `NNS.mode` 13.3 s for n=1001
    (about 2.3e8 bins, more than 2.7 GB of transient vectors).
  - Python `nns_gravity` on the 1e-6 variant: 0.21 s and 309 MB peak RSS vs 0.007 s for the capped
    `gravity_exact`.
  - When (range/width) exceeds INT_MAX the cast is UB: x86 gives INT_MIN, so all mass ends up in
    bin 0.
- Which side is right: no histogram should need more bins than about n (or a few times n).
- Proposed fix:
  - Clamp `width >= range / (4*n)` (or cap nbins at max(128, 4n)) and compute nbins in `double` or
    `int64` before any allocation.
  - Use one cap in both gravity implementations.
  - Results change only for pathological inputs.

### CORE-09: Degree-0 tie conventions are inconsistent: nD CUPM counts ties, DPM counts ties as discordant, and the bivariate copula double-counts; the moments do not partition the sample, giving a dependence floor on discrete data
- Type: r-bug (Python copies it)
- Severity: medium
- Confidence: verified
- Where:
  - R `src/partial_moments.cpp:146` (`CoUPM_CountWorker` drops a row only if `< target`, so ties count) and `:190,213` (`DpmCountWorker`: ties are discordant).
  - R `src/NNS_dep.cpp:67-70` (`d0_clpm` uses `<=`, but the dpm count treats `dx==0` as discordant).
  - NNS-core `partial_moments.cpp:596-629`.
  - Bindings `_nnscore_bindings.cpp:614-623, 842-880`.
- Evidence (`dep_ties.R`; inline in this file):
  - x=c(1,2,3,3,4,5), y=c(3,3,1,5,3,2), t=3: `Co.UPM(0)`=0 but `Co.UPM_nD(0)`=0.333.
    `D.LPM+D.UPM`=0.5 but `DPM_nD(0)`=1. CLPM_nD+CUPM_nD+DPM_nD = 1.833 (should be 1).
  - Independent discrete data whose mean lies on the support ({-1,0,1}, n=12000):
    d0_Co + dpm_d0 = 1.344. `NNS.dep` = 0.247 and `NNS.copula` = 0.362 for independent variables.
- Which side is right: at degree 0 the four (or 2^d) orthant frequencies must sum to 1. Use the
  bivariate convention (x <= t is lower, x > t is upper) everywhere, and define discordant as "not
  all-lower and not all-upper" under that same split.
- Proposed fix:
  - `CUPM_nD`: count only rows with all `x > t`.
  - `DPM`: discordant = not (all `x <= t`) and not (all `x > t`).
  - `copula_signed`: use the same split.
  - This changes public results on tied or discrete data.

### CORE-10: `NNS.copula` uses the bivariate early return `discrete_Co_pm == 1 || == 0 -> 1` for any dimension, so it returns perfect dependence for about 6% of small 3-D integer samples
- Type: r-bug (Python copies it in both the binding and the fallback)
- Severity: medium
- Confidence: verified
- Where:
  - R `R/Copula.R:80`.
  - Binding `_nnscore_bindings.cpp:894`.
  - Python `copula.py` fallback (`discrete_co_pm == 1.0`).
- Evidence (`copula_eq1.R`, `copula_noearly.R`):
  - In 1,140 of 20,000 random 10x3 integer matrices, the sum over pairs is exactly 1. The d=3
    independence level is 1.5.
  - `NNS.copula(X)` returns 1, while the formula without the shortcut gives 0.588. Python
    `nns_copula` also returns 1.0.
  - This feeds M.reg's default `n.best = floor((1-dep)*sqrt(n))`, which then becomes 1.
- Which side is right: "all points concordant" is `sum == C(d,2)`, not 1. For d >= 3 a sum of 1 is
  an interior value.
- Proposed fix: return 1 only when `discrete_Co_pm == choose(d,2)` (or `== 0` with d == 2). This
  changes public results for d >= 3.

### CORE-11: `NNS.part` handling of +-Inf: R 13.3 now treats Inf as a value (`!ISNAN`), but NNS-core and the Python binding still use `isfinite` gating, so quadrants differ
- Type: parity-mismatch
- Severity: low
- Confidence: verified
- Where:
  - R `src/NNS_part.cpp` (13.3 `complete_case_value`).
  - Binding `_nnscore_bindings.cpp:447-452`.
  - NNS-core `src/partition.cpp`.
- Evidence (`part_inf.sh`): x=c(1:20, Inf, -Inf), y=c(20:1, 5, 7), order=2. R gives quadrant
  counts q1 2, q2 9, q3 10, q4 1, and the -Inf row is `q4`. Python gives q1 2, q2 9, q3 11, and the
  -Inf row is `q3`.
- Which side is right: -Inf is below any finite centre, so R 13.3 is right.
- Proposed fix: replace `std::isfinite(x[i]) && std::isfinite(cx)` with `!std::isnan(...)` in
  `partition_labels` and in NNS-core `partition`.

### CORE-12: Heap buffer over-read in `smooth_counts_tri7` (continuous mode) when there are 3 or fewer bins, e.g. `NNS.mode(c(0,0,1,1))`
- Type: r-bug (R, NNS-core and Python all run the same code)
- Severity: low (UB; the values read are not used because the peak loop is empty for 6 or fewer bins)
- Confidence: verified (ASan)
- Where:
  - R `src/central_tendencies.cpp:250-254`.
  - NNS-core `src/central_tendencies.cpp:106-110`. The mirror `counts[-idx]` / `counts[2n-2-idx]` assumes n >= 4.
- Evidence: `asan/asan_driver.cpp mode` reports a heap-buffer-overflow READ of size 4, 0 bytes past
  a 12-byte region (`central_tendencies.cpp:107` and `:108`). Inputs with 4-5 observations often
  produce 3 bins.
- Proposed fix: clamp the reflected index (`std::clamp(idx, 0, n-1)`) or skip smoothing when
  `n < 7`. Public results are unchanged.

### CORE-13: `generate.vectors` reads out of bounds when lag > n; `generate.lin.vectors(x, 0)` kills the R session (SIGFPE); the NNS-core port hits size_t underflow instead
- Type: r-bug
- Severity: low
- Confidence: verified
- Where:
  - R `src/internal_functions.cpp:193-202` (`start = n%lag+1 > n`, `m = 1`, then `x[n]` is read) and `:235/:254` (`% l` with l=0).
  - NNS-core `src/internal_functions.cpp:201-203,226` (`n - start` in size_t underflows, giving a huge allocation or a `max_size` exception).
- Evidence:
  - `NNS:::generate.vectors(c(1,2,3), 7L)` returns 4.9e-311, 2.8e-313, ... and changes on every run.
  - `NNS.ARMA(x6, h=2, seasonal.factor=9, method="lin")` silently returns `NA NA`, while Python
    raises ValueError.
  - `NNS:::generate.lin.vectors(1:4, 0L)` gives "Floating point exception" and exit code 136.
- Proposed fix: validate `1 <= lag <= n` in the kernels (and in `NNS.ARMA` for `seasonal.factor`),
  and compute in signed 64-bit arithmetic.

### CORE-14: NNS-core `fast_lm` uses a naive double mean while R uses Rcpp's long-double mean; both compute fitted values as `a + b*x` and lose precision with large offsets
- Type: parity-mismatch (minor)
- Severity: low
- Confidence: verified
- Where:
  - NNS-core `src/fast_lm.cpp:38-43,141-176`.
  - R `src/fast_lm.cpp:18-19`.
- Evidence (`ab_native.py` case `fast_lm_offset`; exact rational check inline): x = 1e9 + i*1e-3,
  n=500.
  - Slopes: exact -0.1261886569160136, R -0.12618865691600559, NNS-core -0.1261886569158673.
  - Fitted values differ between the two by up to 1.4e-4 (relative 4.6e-4).
  - fitted[0]: exact 0.0059631365, both R and NNS-core 0.0059631467.
  - Python only calls `fast_lm` with time index x = 1..n (ARMA, meboot), so exposure is small.
- Proposed fix: use a long-double, two-pass mean (as R does) and compute fitted values as
  `my + b*(x - mx)`.

### CORE-15: The Python C++ layer has two gravity implementations; the one used by NNS.dep/part/reg (`gravity_exact`) is not bit-identical to R, contrary to its comment
- Type: parity-mismatch (latent) / docs
- Severity: low
- Confidence: verified
- Where: `_nnscore_bindings.cpp:235-372`. The comment says the vendored `nns::gravity` "diverges from
  the R-faithful Python". `gravity_exact` is used by `nns_dep_pair`, `nns_part_off` and
  `nns_reg_mv`. Public `nns_gravity` uses `nns::gravity`.
- Evidence (`gravity_two_impls.py`, 3000 random vectors):
  - `nns::gravity == NNS.gravity` in 3000/3000 cases; `gravity_exact == NNS.gravity` in only
    2030/3000.
  - The worst relative difference is 1.5e-12. A last-ulp difference in a centre is enough to flip a
    split (compare STK-06 and REG-10).
  - No flip was found in 200k small integer samples.
- Proposed fix: make `gravity_exact` use long-double accumulation as R does, or simply call
  `nns::gravity` (with the shared bin cap from CORE-08). Fix the comment.

### CORE-16: R accepts negative degrees and silently computes the wrong thing (`repeat_multiplication` with n<0 returns 1)
- Type: r-bug. Same as PM-18; this adds the C++ root cause.
- Severity: low
- Confidence: verified
- Where: R `src/partial_moments.cpp` (`repeat_multiplication`, the `is_integer` path in LPM_C, CoLPM_C, etc.) and the identical code in NNS-core `partial_moments.cpp:32-36`.
- Evidence (`neg_degree.R`): `LPM(-1,0,c(-2,1))` = 0.5. The value under the power rule is 0.25;
  the kernel returns the degree-0 count. `Co.LPM(-1, c(-2,1), c(-2,1), 0, 0)` = 1: zero deviations
  become 0^-1 = "1", so everything is counted. Python rejects negative degree.
- Proposed fix: `stop()` on `degree < 0` in every R wrapper and kernel, and throw in NNS-core.

### CORE-17: The pure-Python fallbacks are never tested, and there is no switch to force them; results depend on whether the extension built
- Type: api / CI
- Severity: low
- Confidence: verified
- Where: `_native.py` (no environment-variable override) and `.github/workflows/native-backend-ci.yml` (always builds the native extension).
- Evidence (`fallback_ab.py`, which patches `native_fn` to return None): most paths agree to 1e-19
  or exactly, but `upm(8, 100 targets)` differs by 1.97e7 (CORE-01). An sdist install without a
  compiler would silently give different numbers. CI cannot catch fallback regressions.
- Proposed fix: add `NNS_DISABLE_NATIVE=1` to `_native.py`, plus a CI job that runs the parity
  subset with it set.

### CORE-18: CI gaps and hazards in both repos
- Type: docs (CI)
- Severity: low
- Confidence: verified by reading the workflows. The UBSan grep gap is "likely".
- Where / evidence:
  - NNS-core `ci.yml` runs only `ctest`, which is one smoke test with 3 asserts. The live-R fidelity
    suite runs only inside `sync-from-nns-r.yml`, with every step `continue-on-error`, and that
    workflow is never triggered (CORE-04). There are no sanitizer or Python-binding builds.
  - Script-injection patterns: `${{ github.event.client_payload.* }}` and `${{ inputs.* }}` are
    expanded directly inside `run:` in NNS-core `sync-from-nns-r.yml` ("Resolve payload", including
    `echo '${{ toJson(...changed_files) }}'`) and in NNS-python `inspect-r-api-update.yml:52-63`
    (validation runs after the interpolation has already executed). Exploitation needs a token that
    can dispatch, but the fix is cheap: pass values through `env:`.
  - `sync-from-nns-r.yml`: the `workflow_dispatch` path writes `changed_files.json = []`, so the
    adapter reports "Manual review: none" even when the src hash changed.
  - `notify-python.yml` diffs only `HEAD^..HEAD`, so a multi-commit push whose last commit is
    docs-only does not notify.
  - The R `ci.yml` UBSan job runs `R CMD check ... || true` and greps only the check's stderr.
    Sanitizer reports from tests and examples go to `*.Rcheck/*.Rout`, so they are likely missed.
    The job has no ASan, so CORE-12 and CORE-13 could not be caught.
  - NNS-python `native-backend-ci.yml` runs on Ubuntu only. Wheels for macOS and Windows get a
    one-line `lpm` smoke test (`pyproject [tool.cibuildwheel] test-command`).
  - `workflow-baseline.yml` (pull_request_target with a PAT) would silently delete any new receiver
    workflow such as `sync-from-nns-core.yml`.
- Proposed fix:
  - Run fidelity (with an R setup step) and an ASan/UBSan build in NNS-core `ci.yml`.
  - Route all event fields through `env:`.
  - Diff `github.event.before..github.sha`.
  - Fail the UBSan job on `runtime error` found in `*.Rcheck/**/*.Rout*`.
  - Run a parity subset on the macOS and Windows wheels.

### CORE-19: Dead or unused native code: R exports kernels that nothing calls, and Python binds kernels it never uses
- Type: dead-code
- Severity: low
- Confidence: verified (grep)
- Where / evidence:
  - R `RcppExports.R` exports with no caller in `R/*.R`: `NNS_reg_points_cpp`, `NNS_mreg_reduce_cpp`
    (`table_ops.cpp`), `NNS_dep_matrix_cpp`, `NNS_distance_path_cpp`, `fast_lm_mult`,
    `sd_dom_matrix_prefix_parallel` and `NNS_bin`.
    - `NNS_dep_matrix_cpp` uses arithmetic-mean partitions and floor(log2 n) order, unlike `NNS.dep`
      (gravity, ceiling). On 4 ETFs its dependence differs from `NNS.dep(X)` by up to 0.070
      (`dep_matrix_kernel.R`).
    - `NNS_reg_points_cpp` would index `rx[-1]` if given no points.
    - `NNS_bin` casts `(val-origin)/width` to `int` without a range check.
    - `NNS_mreg_reduce_cpp` uses reducer codes (0 = gravity, 3 = mode) that differ from
      `NNS_mreg_setup_cpp` (3 = gravity, 2 = mode).
  - Python binding `nns_reg_mv` re-ports the Python path rather than R's `NNS_reg_points_cpp`. It
    hard-codes stn = 0.95 and omits R's single-row tripling. It is only reachable in the gated regime.
  - Unused bindings are listed under T5. NNS-core modules that are compiled but never bound:
    dependence, distance, partition, seasonality, SD, rescale, meboot, sampling and ARMA weighting.
- Proposed fix: delete the dead R exports, or test them against the live paths. Drop the unused
  bindings, or wire them in where NumPy is slower. Note in `docs/` which NNS-core modules ship
  without bindings.

### CORE-20: R mreg prediction workers call Rmath and `Rcpp::stop` from RcppParallel threads (thread-safety hazard)
- Type: r-bug
- Severity: low
- Confidence: speculative (no failure reproduced; valid inputs never reach the warning or stop paths)
- Where: R `src/NNS_mreg_predict.cpp:316,329` (`::Rf_dt`, `::Rf_dnorm4` inside `PredictPathWorker` when `nthreads > 1`) and `:506` (`stop()` in `distances_for_row`, unreachable after `validate_dist_code`). `NNS_distance.cpp` deliberately avoids Rmath in its workers ("bypasses R C-API ::Rf_dt").
- Proposed fix: use the proportional closed-form densities as `NNS_distance.cpp` does, and remove
  `stop()` from the worker code.

---

## Coverage (checked, found fine)

- NNS-core builds with no warnings at default flags. Smoke test and the 28-case fidelity suite pass
  against R 13.3.
- The vendored core is byte-identical to NNS-core@cfc25a3; HEAD only adds a provenance comment and a
  CMake option.
- Native kernels vs R 13.3 (`ab_native.py`/`.R`, 261 checks) match to <= 1e-12:
  - LPM/UPM and their ratios at degrees 0/0.5/1/2/2.5/3 on returns (the prefix path is fine there).
  - Co.LPM/Co.UPM/D.LPM/D.UPM with unequal, zero and fractional degrees on returns, ties and discrete
    data.
  - Vector-target recycling.
  - Co.LPM_nD/Co.UPM_nD/DPM_nD at degrees 0/0.5/1/2 with norm on and off, including ties.
  - `Co.LPM_nD.batch` and `PM.matrix` in 6 degree/pop_adj/norm configurations (including the crossed
    DUPM/DLPM mirror).
  - Gravity and mode (continuous, discrete, multi) on 9 vector types, `generate.vectors`,
    `generate.lin.vectors`, `stoch_superiority_cpp`, `is.discrete` and `fast_lm_mult`.
- NNS-core `partition`, `seasonality`, `fsd/ssd/tsd_uni` and `rescale` match the R 13.3 kernels
  exactly on 3 datasets (`core_fid/`).
- The ASan/UBSan sweep over all NNS-core public kernels with n = 1/2/3/4/5, constant, NaN, Inf,
  large-integer, tight-IQR and random inputs found nothing beyond CORE-07, CORE-12 and CORE-13.
  NaN/Inf inputs to seasonality and SD throw cleanly.
- Thread safety of NNS-core `parallel_for` bodies: all write disjoint slots
  (dep_matrix, distance_path_parallel, pm_matrix with its upper-triangle mirror, sd dom matrix).
  R `partial_moments.h` and `NNS_dep` workers use `RVector`/`RMatrix` with no R API calls.
- The binding `nns_dep_pair` matches `NNS.dep` exactly on SPY/QQQ (0.553197105270922), in 6.2 ms vs
  R 8.0 ms.
- Python `_mreg_predict`/`_mreg_predict_path` reproduce R's NNS_mreg_predict kernels to 4e-15. Only
  speed differs.
- NumPy implementations that are adequate without native code: SD efficient set (2-3x R, under
  25 ms), `nns_part` (2-4x R, under 15 ms), `nns_gravity`/`nns_mode` at n=1e5 (faster than R), copula
  (0.6 ms vs R 0.8 ms for 400x14).
- The `NNS_xstar_path_cpp` denominator cannot be 0, because R orders by |coef| with zeros last and
  drops NA (falling back to the R path). `NNS_mreg_setup_cpp`'s `find_interval_one` matches
  `findInterval(rightmost.closed=TRUE)`.
- R's 13.0 `NNS_distance` row-major indexing bug is fixed in 13.3. NNS-core was already correct.
