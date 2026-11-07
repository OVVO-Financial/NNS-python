# NNS-Python: R 14.0 parity integration gate

> **Do not merge, squash, force-push, or close the open function-port PRs while the cumulative R 14.0 gate is red.** This is a coordination ledger for the existing implementation, not a proposal to rewrite it.

## Local repair verification (2026-11-07; not yet published)

The repair patch described in [R14 local parity repair](R14_LOCAL_PARITY_REPAIR.md)
passes 2,961 native tests (13 existing skips), 2,815 public-fallback tests
(37 skips), and 9 live-R practical examples (2 missing optional-caret skips).
Ruff, mypy, the sdist-to-wheel build, and an isolated wheel smoke test also pass.
The cache now has 2,764 R-authored entries in 80 shards. `extern/NNS-core` is
unchanged; separately attributed native R14 overlays replace stale dispatch.

These are **local** results for a patch targeting remote head `71ae8e8...`.
They are not a claim that the patch has been pushed or that GitHub's matrix is
green. The earlier baseline table below is preserved as historical evidence.
Do not merge until the repaired head passes remote CI.

## Authoritative source and branch structure

- R behavior: `OVVO-Financial/NNS@7b07a837c130ba7220dbe16c584221508757d5ae`, NNS 14.0.
- The 41 still-open port PRs #148–#189 (excluding #177, which is not open) are **a single linear stack**: each change's base is the immediately preceding open head. All their code is present in `rasheed/py-co-moment-lengths` (#189).
- `main → #189`: **223 commits, 36 Python source files** (comparison at initial R14 triage).
- #191 (`integration/r14-parity-on-python-stack-20261106`) is **6 commits on top of #189, with no Python source changes**; it imports and audits R14 reference fixtures.
- #192 (`automation/r14-stack-cache-20261106`) is based on #191; it contains rebuilt R14 fixtures **and source-level Python repairs**. It is the only active integration/fix head for this audit.
- #190 (`automation/r-cache-nns-14.0`, based on `main`) is a separate initial cache-generation PR. Its cache-only update **cannot be landed into `main` separately** from the cumulative Python implementations.
- Do not let the cache-generation workflow replace #192 from #191: that discards its Python repairs. Regenerate fixtures **in place** and preserve `sync/nns_source.json`.

## Verified R14 baseline (before current code-fix cycle)

The corrected live R14 run generated **2,745 source-authored entries in 78 shards**. Cache-only run on the same refreshed head: **88 failed, 2,021 passed, 12 skipped**; independent invariants: **14 failed, 694 passed, 1 skipped, 3 xfailed**. These are gate failures, not permission to weaken or drop tests.

| Cluster | Remaining cache-only parity failures | Original feature PRs to preserve | Current R14 work |
|---|---:|---|---|
| `NNS.stack` | 21 | #154, #161, #168, #176 | #192 nominal-class routing and shared regression; inspect OOF candidate selection |
| `NNS.boost` | 13 | #154, #161 | Depends on both `nns_reg_engine` and `nns_stack`; isolate independent sampling/frequency mismatches |
| Label/factor dispatch | 11 | #155, #163 | #192 regression categorical exact-cell correction; resolve live-R fixture exceptions separately |
| `NNS.ARMA` / optimizer | 11 | #150, #159, #166, #173, #174, #182 | #192 complete R14 optimizer return and split-only bias policy |
| `NNS.VAR` | 11 | #159, #180 | Audit shared stack/ARMA forecast path, then VAR-specific lag and ensemble logic |
| `NNS.reg` | 10 | #151, #154, #155, #157, #169 | #192 R14 exact joint-cell rules; investigate factor and multiclass fit separately |
| `NNS.diff` / derivatives | 4 | #151, #161, #168, #186 | Remains independent |
| Other (including R13 smoke) | 7 | Feature-specific | Do not relabel legacy-version failures as current R14 success |

**These counts are by test file, not proven unique root causes.** Several downstream failures can disappear together; do not claim that without rerunning the same R14 fixtures.

## Execution / landing policy

1. Treat the **R 14.0 code and its generated fixture payload as the oracle**. No old cache carry-forward, invented expected output, threshold relaxation, suppressing checks or artificial xfail/skip.
2. For each cluster, compare the **earliest differing intermediate**: predictor metadata → encoded design → regression points / RPM → fitted values → OOF predictions → candidate selection → final output. Fix the underlying function, not each failure assertion.
3. Isolate regression and nominal classification first; rerun labels, regression, multivariate fast-path invariants, stack, boost. Then ARMA optimizer and VAR. Finally diff and native-vs-fallback invariants.
4. Separate harness defects from algorithmic defects. A missing R fixture must be traced to the live R exception or codec; a heterogeneous output must be compared recursively rather than coerced blindly into `float64`.
5. Keep the fix on #192 during R14 integration so all existing source work is preserved. Record the originating feature PR numbers above. **Do not transplant fixes into individual old heads and trigger 41 cascading merge conflicts mid-audit**.
6. Once strict R14 native, fallback, cached parity and invariant gates pass, decide a controlled **ancestry-preserving** landing sequence for the existing stack (or a reviewed integration merge) based on repository merge policy. Never squash an upstream stack commit without rebasing downstream heads; never report draft or failing work as merged.
7. Each CI pass must report failures by function and Python version, with explicit before/after comparisons. Keep legacy R13 compatibility cases visible as separately versioned tests.

## Current source changes on #192

- `tests/_r.py`: heterogeneous nested R lists are preserved rather than rejected by NumPy coercion.
- `tests/invariants/test_r_cache_codec.py`: round-trip regression tests for codec and source provenance.
- `src/nns/_reg_engine.py`: a fully categorical predictor design uses R14's maximum-order exact joint cells with default neighbor count one.
- `src/nns/arma.py`: R14 `NNS.ARMA.optim` now exposes `raw.results`, `validation.forecast`, `bias.validation` and an explicit `bias_correction` policy (`none` or `split`).
- Invariant coverage: numeric multivariate max-order fast path is tested separately from R14 nominal-class posterior output; optimizer tests assert the revised return contract.

**Status:** work in progress, PR #192 remains draft. Counts above are a reference baseline prior to these newest source fixes, not a claim of all-green CI.
