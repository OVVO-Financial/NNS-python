# R 14.0 source-faithful Python alignment: local verification

Date: 2026-11-07.

**Delivery status: locally implemented and tested; not pushed, merged, or released.**
The current GitHub connection exposes read-only actions. These changes are supplied
as an apply-ready patch for the existing PR #192 branch, not as a replacement for
its inherited function work. GitHub CI must run on the resulting commit before
any merge or release decision.

## Exact source and ancestry

- Python repository: `OVVO-Financial/NNS-python`.
- Target branch: `automation/r14-stack-cache-20261106` (draft PR #192).
- Expected remote HEAD: `71ae8e820a099c46f04c14a638610af49f810d62`.
- Imported source archive: `9ee110aa03dfde0e1d4ecd21429292305d6c3518`.
  The GitHub comparison confirms that only `.github/workflows/r14-reproduction.yml`
  differs between that archive and the target HEAD. This patch does not touch it.
- R behavior: `OVVO-Financial/NNS@7b07a837c130ba7220dbe16c584221508757d5ae`.
- Installed oracle package: NNS 14.0, compiled from `NNS_14.0.tar.gz`.
- Oracle archive SHA256:
  `5060f6ec426c889501740bf8a585d41f37a746554cb6fdf81f301ca41232cd5b`.
- The cumulative implementation under #189 -> #191 -> #192 is preserved.
  No original feature PR was merged, closed, or replaced by this repair.
- `extern/NNS-core` is unchanged. R14 native additions have separate, hashed
  source attribution in `sync/nns_source.json`; the historical vendor hash is
  not represented as the hash of the new algorithms.

## Measured results

The downloaded Python 3.13 CI baseline at `d2ea97b...` contained 77 failing
cached-parity tests and 9 failing invariant tests. Some failures were genuine
implementation differences; others asserted obsolete R13 contracts. The latter
were replaced by checks of the actual pinned R14 contract, not by skips,
expected-failure markers, or wider parity tolerances.

| Verification gate | Passed | Failed / errors | Skipped |
| --- | ---: | ---: | ---: |
| Native: complete parity, invariants, properties, plotting, vignettes | 2,961 | 0 | 13 |
| Native invariant subset of that run | 721 | 0 | 1 |
| Native cached-parity subset of that run | 2,118 | 0 | 12 |
| Native property subset | 59 | 0 | 0 |
| Native plotting subset | 54 | 0 | 0 |
| Native vignette subset | 9 | 0 | 0 |
| Public pure-Python fallback: parity and invariants | 2,815 | 0 | 37 |
| Live installed-R practical examples | 9 | 0 | 2 |

Rows overlap; their counts must not be added as unique tests. The fallback run
uses `NNS_DISABLE_NATIVE=1`, matching the repository's fallback gate. The compiled
extension remains installed, so tests explicitly calling it may still run.

Additional checks:

- `ruff check .`: passed with Ruff 0.17.4.
- `mypy`: passed, 54 source files, with mypy 2.4.1.
- `git diff --check`: passed.
- Editable native build: passed.
- Source distribution build: passed.
- Native wheel built from that source distribution, rather than the working tree:
  passed.
- Isolated wheel smoke test, bypassing the editable import finder: passed for
  partial moments, versioned R14 dependence, categorical exact-cell regression,
  and pipeline VAR. The extension reports the pinned R source commit.
- Native provenance and dispatch checks: passed.

### Skips and limits

The native suite's 13 skips are the 11 intentionally live-R-only practical
examples, an existing empty example parameter set, and one already-skipped
obsolete seed-invariance assertion. The live-R run independently exercises nine
of those eleven examples; two require the optional R `caret` package, absent in
this environment. The fallback run additionally skips 24 native-only checks.
No skip or xfail was added to hide a mismatch. Three obsolete native-mode xfails
and two obsolete live-VAR xfails were removed only after they passed.

This is Linux x86_64 / Python 3.13.5 verification with NumPy 2.3.5, SciPy 1.17.0,
pytest 9.0.2, Hypothesis 6.170.15, and R 4.5.0. The existing Hypothesis CI profile
was used without reducing its example counts. CI's exact older Ruff/mypy pins,
Python 3.11/3.12/3.14, and Windows/macOS were not run here. The downloaded R runtime
emits TBB loading warnings; reference calls use the serial `ncores=1` path.
There is no performance or all-input equivalence claim beyond the tested cases.

## Actual implementation defects repaired

### Shared regression, labels, and classification

`R/Multivariate_Regression.R` retains both atoms of binary coordinates even at
finite order. Python previously merged a terminal cell. The shared partition
engine now uses the R14 support IDs. Default neighbor selection also averages
the copula summaries for `y` and `-y`, as R14 does, instead of allowing a response
sign convention to change the smoothing count.

Unordered predictor levels are canonicalized while explicitly ordered levels
retain their order. Class response maps remain distinct from predictor metadata.
The univariate posterior interpolation uses R's fraction-first arithmetic; an
otherwise tiny rounding difference was changing tied class decisions.

### Stack and Boost

The regression component of Stack now receives the selected original predictor
coordinates. It previously received two copies of the reduced synthetic index,
so its cross-validation candidate was not the model R14 deployed. The old
response-offset workaround is gone; numeric internal fits explicitly request
REG rather than accidentally triggering nominal classification.

Fold balancing follows R's first-observed class order and permits the singleton
case R accepts. Boost no longer decodes an already-decoded final Stack result;
that second conversion collapsed valid 0/1 predictions to zeros. Both the
ordinary numeric and factor/nominal test families are covered by the final gate.

### ARMA, seasonality, and VAR

Python's nonlinear ARMA shrinkage was missing. Its component-level shrink
operation, and the single-lag linear shortcut's shrink guard, now follow
`R/ARMA.R`. R14 optimizer scoring uses the matching mean/zero-denominator rule.
Seasonality modulo augmentation now applies the R14 baseline-CV rule for periods
already examined and rejected.

VAR now defaults to R14 pipeline validation. Preprocessing, univariate forecasts,
and the lagged forecast predictors are reconstructed at the earlier validation
origin; held-out future observations do not enter those predictors. The explicit
conditional-validation path remains available. A regression guard perturbs only
held-out observations and verifies that training/predictor arrays do not move.
Trailing missing-value extrapolation uses R14's index-cell gradients.

### Native kernels

The recorded vendor tree contains older central-tendency and dependence kernels.
Explicit R14 overlays supply gravity/mode and the multilevel, null-corrected
dependence estimator. They are attributed to the exact R source files. Their
non-R numerical adaptations (vector marshalling, erfc, and hypergeometric
initialization) are documented, rather than described as byte-identical source.

Versioned native entry points (`gravity_r14`, `mode_r14`, `gravity_exact_r14`,
`nns_dep_pair_r14`) prevent new Python code from silently binding an older native
extension. Public functions use the native R14 path when available and the
independent Python fallback otherwise. Both paths pass their comparison gates.

## Cache and test-transport repairs

The committed oracle cache now contains **2,764 entries across 80 shards**, up
from 2,745 entries across 78 shards. Added or changed reference values came from
the installed pinned R14 oracle, never from Python output.

- Four missing label references were recovered by suppressing plotting in the
  headless reference calls, not by substituting fabricated outputs.
- New shrinkage and multiclass-Boost interval references were generated by R14.
- Variable-length nominal prediction sets are compared recursively; coercing
  them to a rectangular numeric array previously failed before values were
  compared.
- Live curve-fitting inputs are exported with `digits=17`. The former 15-digit
  JSON export moved observations across exact partition boundaries.
- Small integer-power R callbacks and derivative anchors were checked against
  the actual oracle rather than excused by looser tolerances.

## Test changes that reflect an R version change

These are explicit contract updates, not algorithmic corrections disguised as
changed expected values:

- R14 nominal regression returns posterior probabilities and prediction sets,
  not ordinal confidence intervals. Properties now check normalization, valid
  class membership, ordered sets, and the requested estimated probability mass.
- Stack/Boost retain binary interval compatibility; multiclass intervals are
  NULL. A newly generated R14 Boost fixture verifies that distinction.
- Positive-degree multivariate cumulative hazard is rejected by R14 because the
  generalized partial-moment ratio is not a joint distribution. Its property
  test now asserts that exact rejection.
- The classification vignette displays R14's class-specific signed-index
  equations instead of treating species labels as one ordinal response.
- Version-specific smoke cases keep their inputs but identify R14 behavior
  explicitly. Historical numerical constants are not silently rebranded as
  proof that both R13 and R14 behave identically.

## Reproduction commands

Install the package and ordinary test/build dependencies, including the native
extension. Ordinary cached tests require no R process. The verification environment
hid `Rscript` from PATH for cached gates, as ordinary GitHub CI does.

```sh
python -m pip install -e . --no-build-isolation --no-deps
NNS_R_CACHE_ONLY=1 python -m pytest -q tests/invariants tests/parity \
  tests/property tests/plotting tests/docs/test_vignette_examples.py
NNS_R_CACHE_ONLY=1 NNS_DISABLE_NATIVE=1 python -m pytest -q \
  tests/invariants tests/parity
ruff check .
mypy
python -m build --sdist
python -m pip wheel --no-deps --no-build-isolation dist/ovvo_nns-2.1.0.tar.gz
```

For live-R practical examples, install the exact archive above, verify
`packageVersion("NNS") == "14.0"`, and clear all CI/offline-cache toggles before
running `tests/parity/test_practical_examples.py`. `caret` is needed for all eleven.
Do not regenerate the cache from an unpinned installed NNS version.

## Landing discipline

Apply the supplied patch only to the clean, named PR #192 head. Its helper checks
the branch, remote repository, exact HEAD, patch digest, and `git apply --check`.
It does not commit, push, merge, force-push, or close PRs. Keep the cumulative
ancestry and run the complete GitHub matrix after pushing the repair commit.
The green local gates do not change the remote branch's status until that happens.
