# Parity Results

## Executive summary

The current prototype has strong fixture-backed parity coverage for the core partial-moment machinery and several original R test areas, plus broad cache-backed parity coverage. Remaining gaps are documented and should be closed before final NNS-python migration.

This report consolidates the merged-state parity evidence from `docs/parity_status.md`, `docs/original_tests_adoption.md`, `tests/parity/`, `tests/fixtures/original_tests_expected.json`, `tests/_r_cache.json`, and `tests/invariants/test_native_original_src_coverage.py`. It does **not** claim full package parity. It also does not introduce native routing, rename `pynns`, touch NNS-python, or move the project toward publication.

## Test commands

The expected verification commands for this state are:

```bash
python -m pytest -q tests/invariants
PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity
PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity/test_original_*
ruff check .
mypy
python -m build
```

`python -m build` is a packaging check only. It should be run when the local environment already has build tooling available. If the `build` module is missing and network access or dependency installation is unavailable, that limitation should be recorded instead of treating it as a parity failure.

## Latest observed results

Verification on the `close-all-parity-gaps` branch on 2026-06-12, using the repository virtual environment, observed:

- `python -m pytest -q tests/invariants` produced `314 passed`.
- `PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity` produced `1778 passed, 11 skipped` (no failures). The previously reported `test_nns_boost_ivs_test_none_matches_r` failure no longer occurs (see "Gap closure summary" below).
- `PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity/test_original_*` produced `12 passed` (up from `9`; three new copula parity tests added).
- `ruff check .` passed.
- `mypy` passed.
- `python -m build` succeeded (built `nns_pm-0.2.0.tar.gz` and the `cp311` wheel, including the native `_nnscore` extension). CI also runs `python -m build` as a workflow step.

The 11 skips are all intentional live-R-only practical examples in `tests/parity/test_practical_examples.py`; they are not cache-backed parity coverage gaps.

### Earlier consolidation snapshot (pre-fix)

The earlier consolidation-branch snapshot recorded `1 failed, 1773 passed, 11 skipped`, where the failure was `tests/parity/test_boost.py::test_nns_boost_ivs_test_none_matches_r` for the `depth=None` / `feature_importance=False` case. That failure was triaged and resolved on this branch; the committed R cache matches Python to ~3.5e-15 and the boosted result is seed-invariant.

Historical known results from PR #7:

- `python -m pytest -q tests/invariants` produced `314 passed`.
- `PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity` produced `1765 passed, 11 skipped`.
- `ruff check .` passed.
- `mypy` passed.
- `python -m build` was blocked locally by a missing `build` module / network limits.

Historical known results from PR #8:

- `python -m pytest -q tests/invariants` produced `314 passed`.
- `PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity/test_original_*` produced `9 passed`.
- Original test adoption added fixture-backed parity tests for ANOVA, dependence/copula partial coverage, partial moments, partition, stochastic dominance, SD efficient set, and univariate SD routines.

## Native-vs-fallback coverage

Native-vs-fallback coverage is enforced in `tests/invariants/test_native_original_src_coverage.py` and documented in `docs/parity_status.md` and `docs/native_original_src_coverage.md`.

Current verified native/fallback areas include:

- Native smoke checks for symbols exported by the currently built optional extension.
- Public fallback behavior when native is disabled or unavailable.
- Partial-moment native routing for `lpm`, `upm`, `lpm_ratio`, `upm_ratio`, `co_lpm`, `co_upm`, `d_lpm`, `d_upm`, and `pm_matrix` where finite inputs and supported shapes permit native dispatch.
- Explicit non-finite fallback behavior for `lpm` and `upm`, preserving Python fallback semantics instead of forcing native execution on `NaN` inputs.
- Private/native backend smoke checks for selected original-source helpers such as fast linear-model helpers and internal utility bindings when exported.

This is not a full native-backend claim. Some C++ functions are intentionally unbound, private-only, or deferred while the Python public semantics and parity fixtures mature.

## R-cache parity coverage

The committed cache in `tests/_r_cache.json` is the offline parity source used by `PYNNS_R_CACHE_ONLY=1`. The cache currently records schema version `1`, R NNS version `12.1`, and 2,406 keyed R result entries.

Cache-backed parity covers broad public API behavior across `tests/parity/`, including ANOVA, ARMA, boosting, categorical wrappers, causation, CDF, classical helpers, co-moments, copula bivariate coverage, core partial moments, dependence, differences, distance, Monte Carlo helpers, meboot, multivariate regression, normalization, partitioning, PM matrix, practical examples, public wrappers, regression, regression helpers, SD clustering, seasonality, stack, stochastic dominance, stochastic superiority, and variance helpers.

When `PYNNS_R_CACHE_ONLY=1` is set, missing cache entries remain blocked unless the cache is regenerated in an environment with `Rscript` and R NNS installed. Any such cache miss is a parity-data gap, not evidence that Python and R match.

## Original R tests adoption coverage

`original_tests/` has been inventoried in `docs/original_tests_adoption.md`. The adopted pytest coverage uses committed R-derived fixtures in `tests/fixtures/original_tests_expected.json` and literal deterministic vectors parsed from the original R test files.

Current original-test adoption includes:

- ANOVA certainty and pairwise matrix checks from `test_ANOVA.R`.
- Bivariate continuous copula coverage from `test_Copula.R`.
- Partial-moment scalar coverage for `LPM`, `UPM`, `Co.UPM`, `Co.LPM`, `D.LPM`, `D.UPM`, `LPM.ratio`, and `UPM.ratio`.
- PM matrix covariance outputs and survival CDF behavior from `test_Partial_Moments.R`.
- Partition-map order, row labels, orientation, and regression points from `test_Partition_Map.R`.
- FSD, SSD, and TSD label parity from `test_FSD_SSD_TSD.R`.
- SD efficient-set name/order parity from `test_SD_efficient_Set.R`.
- Univariate FSD, SSD, and TSD routines from `test_Uni_SD_Routines.R`.

The original-test fixture file contains expected values for seven original R test files, including expected values for documented copula gaps that are not yet adopted as full Python API parity.

## Fully adopted functions

The following areas are fully adopted relative to the original R tests currently represented in pytest:

- `NNS.ANOVA` / `pynns.nns_anova` for original certainty and pairwise matrix behavior.
- `NNS.part` / `pynns.nns_part` for the original partition map case.
- `NNS.FSD`, `NNS.SSD`, and `NNS.TSD` / `pynns.fsd`, `pynns.ssd`, and `pynns.tsd` for original dominance-label cases.
- `NNS.SD.efficient.set` / `pynns.sd_efficient_set` for original efficient-set name and order cases.
- `NNS.FSD.uni`, `NNS.SSD.uni`, and `NNS.TSD.uni` / `pynns.fsd_uni`, `pynns.ssd_uni`, and `pynns.tsd_uni` for original unidirectional dominance cases.

These are full adoptions of the current original-test fixtures, not claims that every parameter combination or every R package behavior is complete.

## Partially adopted functions

The following areas are partially adopted and should remain clearly documented:

- Partial moments as a family: scalar original-test cases and broad cache-backed parity are strong, but this remains scoped to the tested public behavior and documented native/fallback routes.
- R cache parity generally: broad cache-backed coverage is present, but any test requiring an absent cache entry remains blocked in cache-only mode when `Rscript` is unavailable.

## Newly adopted on this branch

- `NNS.copula` / `pynns.nns_copula`: bivariate continuous, bivariate discrete (`continuous=False`), and three-column continuous/discrete (matrix input) are all adopted against the R fixtures.
- `PM.matrix` / `pynns.pm_matrix`: covariance output parity is adopted, and R data-frame naming is exposed via an optional `names` echo that does not change the numeric NumPy arrays (parity-unaffected, proven by test).

## Intentional divergences and remaining offline limitations

These are documented intentional divergences or environment limitations, **not** unresolved parity blockers:

- R plot artifacts, including `Rplots.pdf`, are intentionally not adopted because CI parity compares returned values and never creates or compares graphics-device artifacts. Policy: `docs/plot_parity_policy.md`.
- `PM.matrix` returns NumPy-first arrays; R data-frame dimnames are exposed only via the optional `names` echo. This is a documented API difference, not a numeric-parity gap.
- Any parity test requiring a missing `tests/_r_cache.json` entry would block under `PYNNS_R_CACHE_ONLY=1` when `Rscript` is unavailable; the committed cache currently covers the full offline parity suite with no such misses.
- The only suite skips are intentional live-R-only practical examples (see above).

## Known gaps

- Full package parity has not been established (this report does not claim complete R NNS coverage).
- Graphics/plot parity is intentionally out of scope for CI parity (documented policy, not a gap).
- Some native original-source functions are private-only, intentionally unbound, or not routed from public Python APIs.
- Cache-only verification depends on the committed cache. Missing cache records would require a developer-local R environment to regenerate; the committed cache currently covers the full offline parity suite.
- The current reports are snapshots of verified behavior; behavior outside the tested fixtures and cache entries should not be described as parity-complete.

The discrete and multivariate copula gaps and the `PM.matrix` naming gap recorded in earlier snapshots are now closed (see the gap closure summary at the end of this report).

## Status table

| Area | Python API | R source | Native routed | Python-vs-R cache parity | Original R test adopted | Native-vs-fallback tested | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| ANOVA | `nns_anova` | `NNS.ANOVA`, `test_ANOVA.R` | No | Yes | Yes | No | fixture-complete | Original certainty and pairwise matrix are fixture-backed. |
| Partial moments: scalar LPM/UPM | `lpm`, `upm` | `LPM`, `UPM`, `test_Partial_Moments.R` | Yes | Yes | Yes | Yes | native-complete | Includes non-finite fallback guard for public behavior. |
| Partial moments: ratios | `lpm_ratio`, `upm_ratio` | `LPM.ratio`, `UPM.ratio`, `test_Partial_Moments.R` | Yes | Yes | Yes | Yes | native-complete | Scalar and vector target routes have native/fallback coverage where supported. |
| Co/dependent partial moments | `co_lpm`, `co_upm`, `d_lpm`, `d_upm` | `Co.LPM`, `Co.UPM`, `D.LPM`, `D.UPM`, `test_Partial_Moments.R` | Yes | Yes | Yes | Yes | native-complete | Original scalar cases are adopted; broader behavior remains bounded by cache tests. |
| PM matrix covariance | `pm_matrix` | `PM.matrix`, `test_Partial_Moments.R` | Yes | Yes | Yes | Yes | fixture-complete | Numeric covariance parity is covered; R data-frame naming is exposed via an optional `names` echo (NumPy-first), proven to match R while leaving numeric arrays unchanged. |
| Survival CDF from original partial-moment tests | `nns_cdf(type="survival")` | `NNS.CDF`, `test_Partial_Moments.R` | No | Yes | Yes | No | fixture-complete | Original survival function values are adopted. |
| Copula bivariate continuous | `nns_copula` | `NNS.copula`, `test_Copula.R` | No | Yes | Yes | No | fixture-complete | Original bivariate continuous value is adopted. |
| Copula discrete mode | `nns_copula(..., continuous=False)` | `NNS.copula(..., continuous=FALSE)`, `test_Copula.R` | No | Yes (fixture) | Yes | No | fixture-complete | Bivariate discrete value (0.4472136) is adopted to `1e-5`. |
| Copula multivariate / three-column mode | `nns_copula(Z[, continuous=...])` | `NNS.copula` three-column cases, `test_Copula.R` | No | Yes (fixture) | Yes | No | fixture-complete | Three-column continuous (0.2519783) and discrete (0.2725541) values are adopted to `1e-5`. Input is an `(observations, variables)` matrix. |
| Partition map | `nns_part` | `NNS.part`, `test_Partition_Map.R` | No | Yes | Yes | No | fixture-complete | Original order, row labels, orientation, and regression points are adopted. |
| FSD/SSD/TSD labels | `fsd`, `ssd`, `tsd` | `NNS.FSD`, `NNS.SSD`, `NNS.TSD`, `test_FSD_SSD_TSD.R` | No | Yes | Yes | No | fixture-complete | Original dominance labels are adopted for represented cases. |
| Univariate SD routines | `fsd_uni`, `ssd_uni`, `tsd_uni` | `NNS.FSD.uni`, `NNS.SSD.uni`, `NNS.TSD.uni`, `test_Uni_SD_Routines.R` | No | Yes | Yes | No | fixture-complete | Original unidirectional cases are adopted. |
| SD efficient set | `sd_efficient_set` | `NNS.SD.efficient.set`, `test_SD_efficient_Set.R` | No | Yes | Yes | No | fixture-complete | Python indices are mapped back to original R names for parity. |
| Broad cached parity suite | Many public `pynns` APIs | Installed R NNS via test harness | Mixed | Yes | Mixed | Mixed | partial | `tests/parity/` is broad and cache-backed, but not full package parity. The full cache-only suite now passes with no failures (`1778 passed, 11 skipped`). |
| Native original-source smoke coverage | Optional `_nnscore` routes and helpers | Vendored NNS-core C++ | Yes, where bound | No | No | Yes | native-complete | Covers currently exported native symbols and public fallback behavior. |
| R plot artifact | No Python API | `Rplots.pdf` and plot flags | No | No | No | No | intentional-divergence | CI intentionally compares returned values, never graphics-device artifacts. Policy in `docs/plot_parity_policy.md`. Not a migration blocker. |
| R testthat harness | pytest invocation | `testthat.R` | No | No | No | No | no-python-equivalent | Python uses pytest rather than R testthat. |
| Python-only invariants | Various Python APIs | n/a | Mixed | No | No | Yes where relevant | python-only | These verify Python contracts rather than R parity. |
| Missing R-cache entries offline | Any affected API | Installed R NNS | n/a | No | n/a | n/a | blocked | Cache misses require online regeneration with `Rscript` and R NNS. |

## Release-readiness assessment

The current merged state is suitable for continued prototype validation and internal parity hardening. It is not release-ready as a full R NNS replacement and should not be described as complete package parity.

Positive signals:

- Invariant checks pass at `314 passed`.
- Cache-only parity passes at `1778 passed, 11 skipped` with no failures.
- Original-test parity passes at `12 passed`.
- Ruff and mypy both pass.
- `python -m build` succeeds locally and in CI.
- Core partial-moment native/fallback behavior has targeted tests.

Release blockers or cautions:

- The parity claim is bounded by committed fixtures and cache entries; full package parity is not claimed.
- R plotting behavior and artifacts remain intentionally unported (documented policy).
- Some native original-source functions remain private-only or unbound.

## What remains before NNS-python migration

The previously enumerated pre-migration gaps are now closed or formally resolved:

1. Discrete and multivariate copula gaps — closed (implemented and adopted).
2. Plot artifacts and graphics behavior — resolved as a permanent, documented out-of-scope policy (`docs/plot_parity_policy.md`).
3. R data-frame naming for `PM.matrix` — resolved via an optional NumPy-first `names` echo with a parity test; numeric parity unaffected.
4. R cache review — the committed cache covers the full offline parity suite with no misses; controlled-environment regeneration remains available via `scripts/regenerate_r_cache.py`.

Ongoing discipline (not blockers):

5. Expand original R test adoption where additional upstream tests or stable public examples are available.
6. Keep native routing limited to verified public behavior and avoid adding new routes without parity and fallback tests.
7. Maintain the `pynns` package name until a separate migration plan explicitly covers NNS-python naming, compatibility, packaging, and publication.

## Gap closure summary (final status)

This section is the clean, current status summary for the `close-all-parity-gaps` branch.

- **Full-suite parity failures:** none. `PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity` → `1778 passed, 11 skipped`.
- **Remaining skips:** intentional and documented only — 11 live-R-only practical examples in `tests/parity/test_practical_examples.py` that regenerate vignette-scale results from installed R NNS on demand. They are not cache-backed parity coverage gaps.
- **`nns_boost` cache parity:** resolved. The previously reported `test_nns_boost_ivs_test_none_matches_r` failure was triaged as CV-split seed-sensitivity on an unseeded call. The boosted result is empirically seed-invariant and matches the committed R cache to ~3.5e-15. The parity test now pins a seed, and `test_nns_boost_ivs_test_none_is_seed_invariant` guards against regression. No tolerance was loosened.
- **Copula discrete status:** implemented and adopted. `nns_copula(x, y, continuous=False)` matches R `NNS.copula(A, continuous=FALSE)` = 0.4472136 to `1e-5`.
- **Copula multivariate status:** implemented and adopted. `nns_copula(Z)` and `nns_copula(Z, continuous=False)` match R `NNS.copula(Z, continuous=TRUE/FALSE)` = 0.2519783 / 0.2725541 to `1e-5`. Matrix orientation: rows are observations, columns are variables; any column count `>= 2` is supported; per-column targets default to column means and can be overridden via `target`.
- **PM.matrix naming status:** resolved as an optional NumPy-first `names` echo. Numeric covariance parity is unchanged; a parity test proves names match R's data-frame dimname behavior while the numeric arrays are byte-for-byte identical.
- **Plot policy status:** formalized in `docs/plot_parity_policy.md`. Graphics-device artifacts (including `original_tests/testthat/Rplots.pdf`) are inventoried but never compared in CI; parity compares returned values only.
- **Build status:** `python -m build` succeeds locally (sdist + `cp311` wheel with the native `_nnscore` extension) and runs as a CI workflow step.
- **Native routing:** no new native routing was added; the `pm_matrix` `names` echo is attached in Python after any native call and does not change native dispatch.
- **Migration verdict:** the NNS-python migration **remains out of scope** and is **blocked** until this PR is merged and green. No `pynns` → `nns` rename, no publication, and no NNS-python changes are part of this work.
