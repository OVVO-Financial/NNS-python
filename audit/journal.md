# NNS implementation journal (append-only)

Rules: append only, newest entries at the bottom, UTC timestamps. Records what was examined,
what was decided, what was committed, and which branches of investigation are open or closed,
so work is not repeated after context compaction.

Key paths
- Audit brief: /home/user/audit/BRIEF.md; fix brief: /home/user/audit/FIX_BRIEF.md
- Findings: /home/user/audit/findings/{PM,DEP,REG,STK,TS,SDA,CORE,X}.md (164+ entries), parsed to report/findings.json
- Repro scripts: /home/user/audit/work/<GROUP>/
- Data: /home/user/audit/data (yfinance 2010-2026 prices/returns, Boston, iris, mtcars, airquality, AirPassengers)
- Repos: /home/user/NNS-python (branch claude/zen-cerf-jh8gs5, pushed), /home/user/NNS-core (same branch),
  /home/user/ovvo-financial/nns (R, read-only, no push access; local fix branches only)
- Python worktrees: /home/user/wt/<name> on fix/<name>; shared venv /home/user/wt/venv with PYTHONPATH=src
- R libs: ~/Rlib = unpatched NNS 13.3 reference; ~/Rlib-patched = CORE C++ fixes; ~/Rlib-rfix = R-level fixes

## 2026-10-31 22:30-23:05 UTC  Setup + audit phase
- Git identity: user.name Rasheed; email first set to rasheed.khoshnaw@gmail.com, push rejected (GitHub email
  privacy), switched to 99349713+gitRasheed@users.noreply.github.com. No commit trailers (user's global CLAUDE.md rule).
- Installed R 4.3.3 via apt, R deps via apt r-cran-* (CRAN was blocked until user opened network), built NNS 13.3
  from the R clone (commit 82a92eb) into ~/Rlib. NNS 13.3 no longer imports Rfast.
- NNS-python baseline: 2240 pass, 2 fail (Boston nns_stack parity), 4 skip.
- Ran 7 parallel audit agents (PM, DEP, REG, STK, TS, SDA, CORE). All finished. Totals: 164 findings
  (4 critical, 25 high, 68 medium, 67 low); 66 are R bugs that Python copies.
- Headline items: DEP-01 NNS.dep(x,-x) Correlation=+1; DEP-02 NNS.caus tau never lags; TS-01 NNS.VAR multivariate
  replays last h observations; TS-02 method="means" zeros; TS-03 nonlin forecasts clipped at 0; PM-01 R integer
  targets replaced by mean (NNS.CDF(1:10) = 0.5 everywhere); REG-01 smooth.spline port wrong; REG-06 multivariate
  nns_reg 5-126x slower than R; STK-01 Boston failures = test used cv_size=0.25 vs R default.

## 2026-10-31 23:05 UTC  First commits (NNS-python claude/zen-cerf-jh8gs5, pushed)
- fc9d424 tests: align Boston nns_stack parity tests with the R reference call (STK-01)
- 6a0224b nns_cdf: native kernels, drop dense n x n fallbacks (PM-02, PM-03, PM-05, PM-21 names)
- b87f75e lpm/upm: reject non-finite x like R, target=None -> mean (PM-04, PM-10)
- Full suite after these: green except the commits fixed the 2 Boston tests.

## 2026-10-31 23:15 UTC  Fix phase launched
- Python fix agents (worktrees): reg-smooth (REG-01/02/07/17, STK-06), reg-perf (REG-06, STK-11, REG-20),
  api-labels (REG-03/12-16/18, STK-04/07/08/12/15), ts (TS python-only items), sda (SDA-01/03/06/11/12 + low),
  dep-pm (DEP-07/10/15-18/22/25, REG-10, PM-09/11/12/21/23).
- NNS-core agent: CORE C++ bugs + R 13.3 kernel sync + threads + fidelity tests + CI; mirrors C++ fixes into R clone
  branch fix/cpp-core-bugs (install ~/Rlib-patched).
- R agent: R-level bug patches in worktree /home/user/wt/r-fixes (branch fix/r-bugs, install ~/Rlib-rfix);
  design questions go to /home/user/audit/R_PROPOSALS.md.
- Own finding X-01: scipy.interpolate imported eagerly (0.4 s import cost); defer until reg-smooth merges.
- Plan after agents: merge fix/* into integration branch in order (dep-pm, ts, sda, reg-smooth, api-labels, reg-perf),
  full suite, push; then Python mirror of R-patched shared bugs (needs cache regen from ~/Rlib-rfix);
  then Python bindings for new NNS-core M.reg kernels; then final report.
- Open branches not yet explored by anyone: examples/ and gists/ folders, docs site build (mkdocs), plotting module,
  NNS-python .github workflows beyond CORE-18, packaging/wheels, Python 3.11 compatibility, hypothesis property tests.

## 2026-10-31 23:35 UTC  Side checks
- examples/run_all_vignettes.py: 9/9 PASS, but 07 Clustering & Regression takes 169 s: cProfile 168 s in
  _reg_engine._mreg_distances (151 s self, 833 calls). Sent to reg-perf agent as benchmark (REG-06).
- Incident: git stash is shared across worktrees; reg-smooth and ts agents popped each other's stashes. ts recovered
  (commit 3bf3d21 clean). Told all fix agents: no git stash, use WIP commits / diff files.
- Explored and closed: examples/ vignettes (all pass). Still open: gists/, mkdocs build, plotting module, NNS-python
  workflows (docs.yml, native-backend-ci.yml, release.yml, workflow-baseline.yml, inspect-r-api-update.yml), packaging.

## 2026-10-31 23:50 UTC  Own exploration (closed branches)
- gists/: curve_fitting called nonexistent nns.reg -> fixed (commit 7ec9339). Other gists only use nns_seas/nns_arma_optim
  (timeseries gists need external data/torch; not run).
- CI: workflows install unpinned ruff/mypy -> pinned (7e31561); docs group capped mkdocs<2 (same commit). mkdocs build --strict OK.
- Plotting smoke: X-05 inconsistent plot= kwarg (deferred until module agents merge; touches their files).
- Integration branch now: fc9d424, 6a0224b, b87f75e, 7ec9339, 7e31561 (pushed). Fix branches are based on b87f75e.

## 2026-11-01 00:05 UTC  R testthat baseline + API fuzz
- R testthat (unpatched 13.3, test_dir package="NNS" load_package="installed"): 96 tests, 0 fail, 2 skip.
  Note: running with library(NNS) gives spurious "could not find function" errors for internals.
- Fuzz harness /home/user/audit/work/X/fuzz.py (22 public functions x 11 awkward inputs). New findings X-06 (dy_dx default
  eval_point crashes, R and Py; sent to reg-smooth + R agent), X-07 (causation constant-series message), X-08 (arma short
  series IndexError; sent to ts agent). nns_moments constant -> ZeroDivisionError already covered by PM-12 (dep-pm).
- Stop hook complained about unpushed branch fix/cpp-core-bugs in the R clone: cannot push (no access); expected.

## 2026-11-01 00:40 UTC  Wave 1 merges into claude/zen-cerf-jh8gs5 (cherry-pick, linear history)
- 7c30910 lpm/upm: NaN targets propagate (my earlier commit raised; broke a causation property test via NaN targets).
- fix/ts (9 commits): TS-15/16/17/18/22/23/24/27 + X-08 + nonlin pred-int lin.resid + seas zero-mean crash. Pushed.
- fix/reg-smooth (6 commits): REG-01 smooth.spline exact (order=max n=1000 119 s -> 0.4 s), REG-02/07 dy_dx, X-06, REG-17
  nns_diff = R uniroot/zeroin2, STK-06 R-matching cor and X* rounding (cor bit-identical 180/180), docs.
- fix/sda (14 commits): SDA-01 fsd_uni continuous 10.6 s -> 5.3 ms, SDA-11 hashing, SDA-06 per-pair grids (11 assets 23 ms
  -> 0.9 ms), SDA-03 robust ANOVA = R stream (Mersenne Twister replay), SDA-04 raise, SDA-12 DataFrame nns_norm, etc.
- Agent-reported open items: tests/_r.py nns_stack_numeric uses set.seed instead of NNS.stack(seed=) (for stack tests owner);
  REG-10 (np.mean vs colMeans ulp) remains main cause of smooth/dy_dx knife-edge differences -> assigned to dep-pm agent;
  vignette tests fail in worktrees only (subprocess PYTHONPATH); property test test_nns_reg_classification_bounds_hold fails
  on base too (to investigate).
- Full suite on integration branch running -> /home/user/audit/report/suite_after_wave1.txt

## 2026-11-01 01:05 UTC  NNS-core pushed; hook notes
- NNS-core claude/zen-cerf-jh8gs5 pushed: 17 commits (CORE-01 prefix backend cancellation-free, CORE-16, 07, 12, 08, 13, 14,
  R 13.1-13.3 sync: dep anchor 0.5, part Inf, generate.vectors len, M.reg + stack kernels port, fidelity 316 cases, sync
  provenance 82a92eb, parallel global thread setting + serial threshold, sync workflow hardening, ci receiver check,
  ASan/UBSan + live-R fidelity CI jobs). Fidelity vs unpatched R: 308/316 (8 deliberate R-bug fixes); vs ~/Rlib-patched 316/316.
- R clone local branch fix/cpp-core-bugs: 7 commits mirroring C++ fixes (cannot push; export as patches at the end).
  Load patched with R_LIBS=~/Rlib-patched (R_LIBS_USER is overridden by ~/.Renviron).
- TODO native integration in NNS-python AFTER reg-perf merges: re-vendor extern/NNS-core, bind mreg kernels, n_threads/ncores,
  GIL release, gravity_exact O(n) binning (DEP-10/CORE-15), sync-from-nns-core.yml receiver + workflow-baseline entry,
  expect CORE-01 to change >32-target degree>=3 results (xfail/regen like NNS-core).
- Stop hook asked to re-author commits as Claude: declined, user asked for their identity. Commits show Unverified (session
  signing key vs user email); harmless.
- X-09 subnormal NaN (property test strategy fixed e06e641), X-10 stack R helper seed ignored (fix after api-labels merges).

## 2026-11-01 01:10 UTC  Integration branch green
- Full suite after wave 1 (ts, reg-smooth, sda + my commits): 2394 passed, 4 skipped, 0 failed (baseline 2240 pass / 2 fail).
- Remaining agents: reg-perf, api-labels, dep-pm (Python), R patches (wt/r-fixes).

## 2026-11-01 01:25 UTC  R patches done
- R agent: 50 commits on R clone branch fix/r-bugs (worktree /home/user/wt/r-fixes). Highlights: PM-01 integer targets,
  PM-06 skew, PM-07 mv survival d>=3, DEP-01 sign, DEP-04/08/13, TS-01 VAR forecast rows, TS-02/03/05/06-09/11-14,
  STK-02/03/05/09, SDA-02/04/07/08/13, REG-04/05/08/11/22/23, X-06, docs. R testthat 96/0 fail.
  Knock-on: REG-08 (NNS.part no integer rounding) changes all NNS.ARMA nonlin/both forecasts (AirPassengers 317.5 -> 333).
- Patches exported: /home/user/audit/r-patches/r-bugs (50) and /home/user/audit/r-patches/cpp-core-bugs (7... incl tests).
- Combined R: worktree /home/user/wt/r-combined branch fix/combined = r-bugs + cpp-core-bugs (clean merge), installed to
  ~/Rlib-fixed; R testthat 101 tests 0 fail. Use R_LIBS=~/Rlib-fixed for "fixed R" reference.
- Proposals (no behaviour change) in /home/user/audit/R_PROPOSALS.md.
- New Python item from R agent: nns_arma(dynamic=True, best_periods=2, modulo=12) -> 441.7 0 0 0 (zero-weight problem,
  R fix ed0a68c) -> belongs to the Python mirror phase.
- NEXT: after reg-perf, api-labels, dep-pm merge: (a) native integration agent, (b) Python mirror-of-R-fixes agent with
  cache regeneration against ~/Rlib-fixed. They overlap in files? mirror touches arma/var/dependence/classical/boost/stack/
  anova/sd/diff/part; native touches _native/_reg_engine/bindings/extern. Mostly disjoint -> can run in parallel worktrees.

## 2026-11-01 01:45 UTC  dep-pm merged; reg-perf done; mirror agent launched
- fix/dep-pm cherry-picked (12 commits: DEP-07, REG-10 copula colMeans, DEP-18 causal_matrix 3.6->1.8 s, PM-12, PM-11
  lpm_var vector 2964->24 ms, DEP-25, DEP-16/17 nns_part mean 1.41->0.038 s, DEP-15 matrix nns_dep, DEP-10 exact gravity
  + native-cap guard, PM-21, docs). Suite running -> report/suite_after_deppm.txt.
- Open from dep-pm: REG-10 remaining one-line switches in _reg_engine._dependence (~line 402) and
  regression._nns_copula_matrix to nns.copula._r_col_means. Root fix DEP-10: remove 1e6 clip in gravity_exact_impl
  (_nnscore_bindings.cpp) in native phase. X-11 (new): co-moment parity tests key the R cache on random data from
  multivariate_normal -> cache misses under other NumPy/BLAS builds (120 cases).
- reg-perf done (11 commits incl. be868cb signed-zero fix): mv nns_reg n=1e4 p=5 9.7->3.1 s (R 2.24), order=max 14.5->0.14 s,
  boost n=2000 p=5 15.4->3.4 s (R 5.0), vignette 07 93->4.4 s, REG-20 r_mean (_rmean.py). Not done: native threading
  (copula_nd/pm_matrix spawn threads per call, 44% of boost time under load), fused C++ prediction kernel.
- Mirror agent launched in /home/user/wt/mirror (fix/mirror from a23ec90): ports R fixes to Python, regenerates affected
  cache shards from ~/Rlib-fixed. Told to do STK/REG-04/REG-11 last (api-labels/reg-perf overlap).

## 2026-11-01 00:41 UTC (real clock; entries above from "00:40" onwards were estimated ~1 h ahead)  Container restart
- Container restarted (exit 137 during full suite with 3 workers + 2 agents' suites). Disk state intact: repos, worktrees,
  ~/Rlib, ~/Rlib-fixed, /home/user/audit. Resumed api-labels and mirror agents with a memory warning (-n <= 2).
- Integration branch: reg-perf cherry-picked (11 commits, HEAD 66a3662) with 2 conflict resolutions: part.py mean path
  (kept dep-pm vectorised structure + reg-perf r_group_means_no_na with `group`), NNS.reg.json shard union
  (script /home/user/audit/report/merge_shard.py). Suite after reg-perf not yet confirmed (killed by restart).
- From now on: timestamps from `date -u`.

## 2026-11-01 00:52 UTC  reg-perf merged + nns_dep root cause found; native agent launched
- Suite after reg-perf merge: 2556 passed, 0 failed. Pushed.
- 81d390a reg: default-order copula target via _r_col_means (REG-10 remainder). Copula part on R sin grid now exact.
- ROOT CAUSE (high value): src/nns/_nnscore_bindings.cpp dep_copula_signed uses np_mean (NumPy pairwise emulation) for cell
  means; R NNS_dep.cpp uses sequential sum/n. On grid/tied data points sit on the cell mean -> quadrant counts flip.
  sin grid: R dep 0.92465837278014773, Py 0.9240158818160937, Py with seq mean 0.9246583727801478. Likely root of
  DEP-09, REG-09, STK-10. Python partitions (_xonly_partition) match R quadrants exactly (verified q1/q2).
  Also: _reg_engine._dependence wraps nns_dep/copula in  (swallows errors; R uses tryCatch too).
- Native agent launched (worktree /home/user/wt/native, fix/native from 81d390a): re-vendor NNS-core c4cb21c, dep mean
  fix native+fallback, gravity clip, threads/ncores/GIL, bind mreg kernels, NNS_DISABLE_NATIVE switch + CI, sync receiver.
- Running agents now: api-labels (resumed), mirror (resumed), native.
- (correction to line above: the swallowed-error pattern is "except Exception: d = nan" around nns_dep/_copula in _reg_engine._dependence; shell ate the backticks)

## 2026-11-01 00:53 UTC  Status map so far (for the final report; commit IDs are rarely in commit messages)
- Python fixed on integration branch: PM-02,03,05,21(names) [6a0224b]; PM-04,10 [b87f75e,7c30910]; PM-09doc,11,12,21,23doc,
  DEP-07,10(py; native clip pending),15,16,17,18,22,25, REG-10 [dep-pm + 81d390a]; REG-01,02,07,17, STK-06, X-06 [reg-smooth];
  REG-06,20, STK-11 + signed-zero fix [reg-perf]; STK-01 [fc9d424]; TS-15,16,17,18,22,23,24,27, X-08 + nonlin pred-int,
  seas zero-mean [ts]; SDA-01,03,04(raise),06,11,12,14,15,16,17(partial),18,23 [sda]; X-01 [0933f2f], X-03,04 [7e31561],
  gist [7ec9339], X-09 test [e06e641].
- R fixed (fix/r-bugs, 50): PM-01,06,07,15,16,22; DEP-01,04,08,13,24,26; TS-01,02,03,05,06,07,08,09,11,12,13,14,19;
  STK-02,03,05,09; SDA-02,04,07,08,13; REG-04(partial),05,08,11,22,23; X-06; docs PM-23, DEP-20,27, REG-24, STK-13,
  SDA-10,20,21,22, TS-20.
- R C++ fixed (fix/cpp-core-bugs, 7): CORE-01/PM-13, CORE-16/PM-18, CORE-07, CORE-12, CORE-08 (+DEP-11 overflow), CORE-13, tests.
- NNS-core fixed (17): CORE-01,02,04,05,06,07,08,11,12,13,14,16,18 (+20 in port).
- Proposals only (R_PROPOSALS.md): DEP-02,03,05,12,14, TS-04, SDA-10, DEP-13 corr-p, stack univariate n.best, dy.d_ bandwidth,
  ARMA.optim equal-weights obj.fn.
- In flight: api-labels (REG-03,12,13,14,15,16,18, STK-04,07,08,12,15), mirror (Python side of R fixes),
  native (DEP-09/REG-09/STK-10 via dep mean, CORE-15/DEP-10 native, CORE-17, PM-19/CORE-06, CORE-02 py, CORE-04 py receiver).
- Not assigned yet: PM-14 hazard ties (R+Py), PM-17 frac VaR tol, PM-20 pm_matrix BLAS, CORE-09/10, CORE-19 dead code,
  REG-19 dead code (~1500 lines), REG-21, REG-25, STK-14/16/17/18, DEP-19/21/23/27(py), TS-10/21/25/26, SDA-05/09/19-24,
  X-05 plot kwargs, X-07, X-10 stack seed helper, X-11 co-moment cache keys.

## 2026-11-01 00:54 UTC  Own items while agents run
- 0933f2f smoothing lazy scipy import (import+nns_reg 0.47 -> 0.14 s). Pushed.
- PM-20 dropped: BLAS vs native pm_matrix mixed/noisy; small-size loss is per-call thread spawn (CORE-06, native agent).
- X-10 (stack seed helper) and X-12 (dy_d_best == dy_d, both wrong; REG-04) handed to mirror agent.
- API coverage: all 47 R exports have Python counterparts (closed).
- Next candidates after merges: REG-19 dead code (~1500 lines) once api-labels + native merge; X-05 plot kwargs after mirror;
  final code-review pass on whole branch diff; report artifact.

## 2026-11-01 01:02 UTC  api-labels merged
- fix/api-labels cherry-picked (6 commits: REG-03,12,13,14,15,16,18, STK-04,07,08,12,15; new src/nns/_labels.py,
  tests/parity/test_labels.py, tests/_r.py nns_frame_call). Conflicts resolved: _reg_engine imports (both), docs/conventions.md
  smoothing+factor paragraph merged by hand.
- Suite: 2572 passed, 1 XPASS (Boston factor path: STK-06 from reg-smooth fixes it) -> xfail removed. Pushed.
- Remaining agents: mirror, native.

## 2026-11-01 01:14 UTC  REG-19 dead code removed
- Coverage run of full suite: /home/user/audit/report/coverage.json (2573 passed under --cov).
- 98cd8cd deleted legacy regression engine: regression.py 1624 -> 447 lines, multivariate_regression.py 473 -> 158,
  smoothing.py removed (0933f2f lazy import is therefore moot). Kept prepare_factor_predictors + its helpers (public but
  UNTESTED: add a test later), result types, _r2. Note: 81d390a's _nns_copula_matrix edit was in dead code (now deleted);
  the live REG-10 path is _reg_engine._dependence.
- Never-executed but KEPT (pure-Python fallbacks for no-native builds): central_tendencies _discrete_mode/_continuous_mode/
  _smooth_counts_tri7/_peak_indices/_non_maximum_suppress, dependence _copula_degree0_unsigned/_simple_bin_counts,
  _helpers _as_vector_or_matrix/_as_point_matrix/_as_flat_vector (unused?), categorical _dense_factor_codes,
  stack _spearman_scores, boost _column_is_numeric -> revisit after native agent adds NNS_DISABLE_NATIVE CI run.
- Remove script: /home/user/audit/report/remove_dead.py (ast-based, keeps transitive deps of roots).

## 2026-11-01 01:19 UTC  More cleanup + tests
- 9acf805 removed 6 caller-less helpers (_helpers x3, categorical, stack, boost).
- b2c85d5 parity test for prepare_factor_predictors (matches R NNS.reg on factor frame: 3.76895402094364, 3.82348990448764).
  R factor_2_dummy_FR = one column per level (FR != full rank); Python matches. R factor_2_dummy_FR(data.frame) errors
  ("Not compatible with requested type") - minor R item, not pursued.
- Starting final report (artifact + docs/audit in repo). Waiting on mirror + native agents.

## 2026-11-01 01:20 UTC  Report scaffolding + R batch 2
- Status map seeded: /home/user/audit/report/status_seed.py -> status.json (140/176 findings with a status).
- Resumed R agent (batch 2, same fix/r-bugs): PM-14, PM-17, SDA-05, SDA-09, SDA-19, TS-25, TS-26, DEP-23, STK-14,
  CORE-09, CORE-10, PM-08 (proposal). After it: re-merge fix/combined, reinstall ~/Rlib-fixed, re-export patches,
  then a Python wave-3 agent to mirror these + Python small items (STK-16/17/18, SDA-24, REG-21, REG-25, X-05, X-07,
  X-11, TS-21 perf, CORE-19 python bindings).
- 01:21 report generator written: /home/user/audit/report/build_report.py (needs prs.json + meta.json at the end).

## 2026-11-01 01:47 UTC  R batch 2 done
- 11 more R commits on fix/r-bugs (61 total): PM-14 50d92b0, PM-17 7b7541a, SDA-05 cc082d7, SDA-09 77284f2, TS-26 334bcd4/afcd423,
  DEP-23 d299238, STK-14 f7caa99 (table_ops.cpp deleted), CORE-10 1d388de, CORE-09 1388357/81dd43a. Proposals: PM-08, TS-25,
  CORE-09 remainder (exact float tie detection = DEP-09), DEP-14 remainder. R tests 96/0.
- fix/combined re-merged (3c949cf) -> installed to NEW lib ~/Rlib-fixed2 (101 tests, 0 fail). ~/Rlib-fixed left untouched
  for the running mirror agent (its reference = batch 1 + cpp fixes).
- Patches re-exported: /home/user/audit/r-patches/r-bugs (61 files).
- Wave-3 Python agent must mirror batch-2 (PM-14, PM-17, SDA-05, SDA-09, TS-26 h validation, CORE-09/10 copula/dep) using
  R_LIBS=~/Rlib-fixed2, after mirror + native merge.

## 2026-11-01 02:05 UTC  native merged (9 commits) + rebuild
- fix/native cherry-picked onto integration: re-vendor NNS-core c4cb21c, dep copula R arithmetic (sin repro bit-identical,
  DEP A/B 55 -> 14 mismatches, REG-09 Boston NNS.dep 13/13, STK-10 iris balance 0.9643 = R), NNS-core gravity (DEP-10,
  CORE-15; 3000/3000 = R), drop nns_reg_mv, threads (set_num_threads, NNS_NUM_THREADS, ncores honoured, GIL release),
  native M.reg predict (nns_m_reg 3.16 -> 0.59 s), warning stacklevel, NNS_DISABLE_NATIVE + CI step + fallback fixes
  (mode fallback 1453/8000 mismatches -> 0), sync-from-nns-core receiver.
- Conflicts: regression.py (native edited deleted legacy code -> ours), mreg/stack/_reg_engine import blocks (union).
- Main venv rebuilt (uv sync --reinstall-package ovvo-nns). uv sync removed ad-hoc pandas -> found X-13: tests/parity/
  test_labels.py hard-imports pandas but pandas was in neither dev group nor CI installs -> added pandas to dev group +
  3 workflows (uncommitted yet, with mypy fix in central_tendencies._smooth_counts_tri7).
- Remaining from native: REG-09/STK NNS.caus/all differ via nns_norm np.corrcoef 2e-15 vs R cor() (Uni.caus(medv, chas));
  DEP mono_rank/step3/y_neg_x causation mismatches; CORE-11 binding Inf partition; iris_shuf; CORE-03.
- Suite running -> report/suite_after_native.txt

## 2026-11-01 02:08 UTC  Identity incident fixed
- The container restart (00:40) reset ~/.gitconfig to Claude <noreply@anthropic.com>. Commits made after it carried that
  author: 19 pushed on NNS-python, 11 on R fix/r-bugs (local), 19 in wt/mirror (local, in progress).
- Fixed: global identity restored (Rasheed <99349713+gitRasheed@users.noreply.github.com>); NNS-python rebased with
  conditional --reset-author from 81d390a^ (tree identical), force-pushed with lease; R fix/r-bugs rebased the same way,
  fix/combined rebuilt (tree identical to 3c949cf), patches re-exported (61, none with the old identity).
- TODO: when cherry-picking fix/mirror, reset author on its commits. After any future restart: re-check git config first.
- Suite after native merge: 2609 passed in 43 s (was 270-450 s: native threading/kernels); NNS_DISABLE_NATIVE=1 cached
  parity+invariants: 2474 passed. d53b9df pandas in dev group + CI (X-13).

## 2026-11-01 02:12 UTC  NNS-core: CORE-09 + CORE-03 ported (pushed)
- 64617c0 CORE-09 degree-0 partition (cupm_nd <=, dpm_nd diff>0, dependence.cpp discordant <=), 9ea8a4a CORE-03 mreg_setup mode
  = nns::mode for numeric cells, dc390fe fidelity: tie cases (r_fix CORE-09), mreg_setup r2 tagged CORE-03, dep_matrix cases
  only if R still has NNS_dep_matrix_cpp (R batch 2 deleted it).
- Fidelity: unpatched R 308/322 + 14 xfail, 0 fail; R + patches (~/Rlib-fixed2) 320/320. Build: /home/user/audit/work/core/b3.
- NNS-python needs a re-vendor of NNS-core HEAD (dc390fe) in wave 3, plus Python fallbacks for CORE-09 + CORE-10 (copula.py and
  binding copula_nd early return) + CORE-03 Python fallback in _reg_engine mode reducer.
- 02:12 proposals extracted (report/proposals_extract.py -> proposals.json). Plan: commit audit/ folder at NNS-python root (findings, journal, R_PROPOSALS, r-patches, repro scripts only) + sdist exclude. Waiting on mirror before wave 3 (shard conflicts otherwise).

## 2026-11-01 03:19 UTC  mirror merged; NNS-core DEP-01/08; wave 3 launch
- fix/mirror (26 commits) cherry-picked + re-authored. Conflicts: dependence.py (sign fix into native-agent fallback),
  part.py (integer-x centroid restore on guard-free native path), regression.py (legacy -> ours), sync/nns_source.json notes.
  Shards: merge_shard.py theirs-preferred union; script /home/user/audit/report/continue_pick.sh.
- Follow-ups on integration: 0732488 unused import, d0f292e re-record 7 entries from fixed R + n.best test compares R,
  94f9bef X-11 co-moment inputs via explicit Cholesky (shards pruned 301->181 etc.), 121a2fb subnormal causation strategy +
  VAR live-R examples xfail(strict=False) (TS-01: pass only vs patched R).
- Key recorder plugin for cache pruning: /tmp/claude-0/plug/keyrec.py (patch both _r and tests._r; run -n 0).
- State: native path cache-only 2714 pass, 1 property flake (causation); NNS_DISABLE_NATIVE path 15 fail (dep tests: Python
  fallback has DEP-01 sign fix, native kernel + 7 cached NNS.dep entries do not).
- NNS-core 52f2b14 DEP-01 sign + DEP-08 count compare (fidelity fixed2 320/320, r133 308+14xfail). Pushed.
- Wave 3: W3a (native re-vendor 52f2b14 + fallbacks DEP-01/08/CORE-09/CORE-10/CORE-03 + REG-08 native part rounding + regen
  dep/copula/nD/reg/stack/boost/part/caus shards from ~/Rlib-fixed2), W3b (pure-Python mirror of R batch 2: PM-14, PM-17,
  SDA-05, SDA-09, TS-26; regen CDF/VaR/SD/ARMA shards from ~/Rlib-fixed2).

## 2026-11-01 03:21 UTC  Report data
- /home/user/audit/report/prs_def.py groups commits by subject regex into proposed PRs (PY 11 groups, CORE 3, R 8) and
  fails if any commit is unassigned (currently PY 110/110, CORE 21/21, R 68). Writes prs.json for build_report.py.
- Still to write: meta.json (lede, steps, note, method, proposals from proposals.json, counts), audit/ folder commit in
  NNS-python (findings, journal, R_PROPOSALS, r-patches, repro scripts), status.json update for W3/X items.
- W3a + W3b running (launched 03:20).
- 03:22 repro script list for audit/ commit: /tmp/claude-0/repro_list2.txt (346 files, 1.9 MB; excludes R package copies, builds, data). Yahoo data not committed (document regeneration).

## 2026-11-01 03:47 UTC  W3a merged
- fix/w3a cherry-picked cleanly (6 commits): vendor NNS-core 52f2b14, fallback degree-0 rules (CORE-09, DEP-08), CORE-10 copula
  dimension-generic shortcut (py + binding copula_nd; 170 -> 0 spurious 1s), REG-08 native nns_part_off no rounding (workaround
  removed), 674 cache keys refilled from ~/Rlib-fixed2 (28 changed), copula smoke values. Main venv rebuilt; nns_dep(x,-x) = -1.
- NNS-core d81b775: sync/nns_r_source.json lists CORE-03/09, DEP-01/08. Pushed.
- 9de61f6 causation property domain (ptp > 1e-100; R also fails at 1e-249 scale, DEP-03). Pushed.
- Cache-only suite: 2738 passed, only NNS.ARMA.optim.custom|ee57704e stale (W3b owns ARMA shards; told it).

## 2026-11-01 03:50 UTC  W3b merged; all green
- fix/w3b (6 commits): PM-14 hazard ties, PM-17 uniroot VaR, SDA-05 between-points, SDA-09 tolerance, TS-26 h validation,
  ARMA seasonal_factor=True honours best_periods (R right). Clean cherry-pick.
- Integration HEAD 8150ed0: cache-only 2778 passed / 0 failed; NNS_DISABLE_NATIVE 2641 passed / 0 failed; ruff+mypy clean.
  Vignettes 9/9 in ~23 s (was ~211 s; 07: 169 s -> 3.45 s). Pushed.
- Launching W4 (small items: REG-21, REG-25, SDA-24, STK-16/17/18, TS-21, X-05) and an independent reviewer of 3d8b4e1..HEAD.

## 2026-11-01 03:53 UTC  Benchmarks (single thread where supported; box shared with 2 agents)
- Scripts /home/user/audit/work/bench/bench.py (before=3d8b4e1 with original .so in /home/user/wt/base, after=8150ed0) and bench.R
  (R 13.3 + patches). Results: before.json, after.json, r.json. Headlines: nns_reg mv n=1e4 p=5 8.12 -> 1.70 s (R 1.77);
  order=max 5.59 -> 0.026 s (R 0.024); nns_cdf bivariate n=4000 2.95 -> 0.24 s (R 0.20); nns_cdf deg 1.5 n=1e4 4.17 -> 1.53 s
  (R 1.33); causal_matrix 0.60 -> 0.16 s (R 0.30); ARMA.optim 0.149 -> 0.054 (R 0.285); stack Boston 0.345 -> 0.187 (R 0.405);
  fsd_uni continuous 0.19 -> 0.002 (R 0.002); nns_seas n=1e4 0.056 -> 0.012 (R 0.006).

## 2026-11-01 08:15 UTC  Final stretch (second container restart lost ~4 h: 03:55 -> 08:10)
- Restart reset git identity to Claude again; restored before any commit. W4 (7 commits) and the reviewer had finished before the
  restart; results read from their transcripts.
- W4 merged: X-05 plot= on 8 functions, STK-16/17 dead code, STK-18/REG-25 API edges, REG-21 bool predictors as factors,
  SDA-24 merge-based SD tests (ssd n=1e4 1.83 -> 0.94 ms), TS-21 ARMA lin recursion (VAR 5x500 ~5.6 -> ~4.3 s).
- Review: no serious regressions; fixed #1 (_fast_lm fallback != native -> meboot replicates differ) 22850a3 and #2 (ncores < 1
  now DeprecationWarning) a1953b2. Open: #3 NNS_NUM_THREADS read once / huge n TypeError, #4 lpm NaN target propagates
  (R raises), #5 removed submodule names without shims.
- Final: cache-only 2808 passed; NNS_DISABLE_NATIVE 2650 passed; ruff/mypy clean. Pushed.
- audit/ folder committed to NNS-python (findings incl. REVIEW.md, R_PROPOSALS, journal, 68 R patches, repro scripts);
  excluded from sdist and ruff.
