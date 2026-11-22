# Parity

## Target and status

R NNS 14.2 is the parity target: `NNS_14.2.tar.gz` as committed at
`OVVO-Financial/NNS@f0387ed857de3d3bf047fa581b854dc5e3d0c3a6` (SHA256
`a4e04b26b3ea01aeae1d6676448194b275bdd3f2aacbafd78b82ba5c2a649201`). Its `R/` and
`src/` files equal the Git tree at that commit apart from CRLF line endings and
final newlines. Every shard in `tests/_r_cache/` is stamped `nns_version`
`14.2`. NNS-core remains the native C++ foundation; R 14 kernels it does not
carry enter as hashed overlays recorded in `sync/nns_source.json`.

Parity is bounded by the committed tests and cache:

- `tests/_r_cache/` — cache-only R result fixtures, sharded into one JSON
  file per R function label (keys are `<label>|<sha256>`, schema version
  `2`, every shard stamped with the generating `nns_version`),
- `tests/parity/` — public behavior parity checks,
- `tests/invariants/` — Python-native contracts and invariants, and
- `tests/fixtures/original_tests_expected.json` — adopted original R tests
  (`tests/parity/test_original_*`).

Any cache miss under `PYNNS_R_CACHE_ONLY=1` is a parity-data gap until the cache
is regenerated with Rscript and the installed R NNS 14.2 oracle.

Plot artifact policy is unchanged: plots and `Rplots.pdf` artifacts are not
parity outputs in pytest; returned values are. See
[`plot_parity_policy.md`](plot_parity_policy.md).

### Bit-level parity

The port aims at R's own doubles rather than values within a tolerance. It
reproduces R's arithmetic: long-double `sum`, `mean`, `var`, `prod` and
`rowSums`; Rcpp's double accumulations and R's C++ operation order; R's `^`;
libm `exp`, `log` and `pow` in place of NumPy's vectorised versions; R's nmath
densities and quantiles; `optimize`; the partial sort behind
`mean(trim = )`; and R's smoothing spline. It does so with the native extension
and with the NumPy fallback (`NNS_DISABLE_NATIVE=1`). Where R draws random
numbers, `random_seed = s` (`seed` for `nns_causation`) reproduces R's stream
after `set.seed(s)`.

Live differential sweeps (`scripts/live_parity/`, see its README) compare the
port with installed R at exact equality (`rtol = atol = 0`). At the time of
writing every case below matches, with and without the native extension
(`python scripts/live_parity/run.py --mode both`):

| Area | Cases |
| --- | ---: |
| Partial moments, VaR, co-moments, `PM.matrix`, `NNS.moments` | 2,932 |
| Central tendencies, `NNS.CDF`, `NNS.dep`, `NNS.copula`, `NNS.caus`, `NNS.norm`, `NNS.rescale` | 793 |
| `NNS.distance`, `NNS.distance.bulk` | 928 |
| `NNS.part` | 920 |
| `NNS.reg` (univariate, multivariate, classification, dimension reduction, smoothing) | 480 |
| R's smoothing spline under `NNS.reg(smooth = TRUE)` | 264 |
| `NNS.seas`, `NNS.ARMA`, `NNS.ARMA.optim`, `NNS.diff`, `dy.dx`, `dy.d_` | 345 |
| `NNS.VAR` | 40 |
| `NNS.stack`, `NNS.boost` | 56 |
| Classification `NNS.stack` built to tie (coarse predictors, pooled folds) | 72 |
| `NNS.ANOVA` | 224 |
| Stochastic dominance (`NNS.FSD`/`SSD`/`TSD`, `.uni`, `NNS.SD.cluster`, `NNS.SD.efficient.set`), edge cases included | 2,583 |
| `NNS.meboot`, `NNS.MC`, `NNS.SS` and ARMA intervals under `set.seed` | 502 |
| Permutation p-values (`NNS.seas`, `NNS.dep`, `NNS.caus`) and `NNS.dep` permutation matches | 348 |
| Multivariate `NNS.reg` predictions on 900-1,500 rows (every distance, `n.best`, tied coordinates) | 84 |
| Compiled paths: integer-degree `LPM.VaR`/`UPM.VaR`, `NNS.copula`, `NNS.dep` matrices, classification predictions, mixed `dy.d_` | 110 |
| Input validation against R 14.2's input layer (accepted inputs and messages; see [conventions](conventions.md#input-validation)) | 546 |

The `*_r14.py` parity tests pin these results with exact cache entries. The
cache builder runs every sweep in both modes after regenerating the cache for a
new R release and reports the table in its pull request.

Scope: the oracle is R 4.5.0 on x86-64 Linux (glibc libm, the reference BLAS,
no FMA contraction), which is where the port was verified. Another platform's
libm, BLAS or compiler can move a result by an ulp on either side (R on 64-bit
Windows, for instance, evaluates `^` with `powl`). The documented API
conventions (NumPy arrays and dicts, finite-input validation, labels) are listed
in [conventions](conventions.md).

### Known retarget fix

The univariate `NNS.reg(..., multivariate.call = TRUE)` regression-point path
used internally by nonlinear ARMA follows R's central-point weighting: R appends
the central regression point again during endpoint consolidation, and Python
preserves that contribution.

## Verifying parity

```bash
python -m pytest -q tests/invariants
PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity
PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity/test_original_*
ruff check .
mypy
python -m build   # packaging check
```

## Installing the R NNS 14.2 oracle

Install the pinned R 14.2 source package, never the CRAN release, into R 4.5.0
(see the scope note under Bit-level parity):

```bash
curl --fail --location -o NNS_14.2.tar.gz \
  https://raw.githubusercontent.com/OVVO-Financial/NNS/f0387ed857de3d3bf047fa581b854dc5e3d0c3a6/NNS_14.2.tar.gz
```

```bash
echo 'a4e04b26b3ea01aeae1d6676448194b275bdd3f2aacbafd78b82ba5c2a649201  NNS_14.2.tar.gz' | sha256sum -c -
```

```bash
python scripts/install_local_r_nns.py --source NNS_14.2.tar.gz --expected-version 14.2
```

`tools/NNS` and `tools/NNS_13.0.tar.gz` hold the earlier R 13.0 source for
reference; they are not the parity target.

## Regenerating the parity cache

After confirming `packageVersion("NNS") == "14.2"`, regenerate the committed
cache with cache-only/offline toggles unset:

```bash
unset PYNNS_R_CACHE_ONLY PYNNS_OFFLINE CI
python scripts/regenerate_r_cache.py -- -n 0 tests/parity
```

If full regeneration is slow or unstable, regenerate deterministic chunks one
file at a time, for example
`python scripts/regenerate_r_cache.py -- -n 0 tests/parity/test_core.py`, then
continue through the remaining parity files. The committed result must remain a
valid `tests/_r_cache/` shard directory: every shard carries the same
`nns_version`, `schema_version == 2`, a `label` matching its filename, and
non-empty `entries`; `scripts/regenerate_r_cache.py` enforces those
guardrails after the pytest run.

Validate the regenerated cache offline:

```bash
PYNNS_R_CACHE_ONLY=1 python -m pytest -q -n 0 tests/parity
```

A `RuntimeError: R cache miss ...` means the cache is incomplete (regenerate the
missing live R entries); an `AssertionError`/numeric mismatch means Python
behavior differs from R NNS 14.2 and the Python implementation must be fixed
without loosening tolerances.

Where an R toolchain is unavailable (for example sandboxed CI or
proxy-restricted runners that cannot install R), the cache cannot be regenerated
live; rerun the local-source install and `scripts/regenerate_r_cache.py` on a
host with R.

## Cache provenance: R 14.2

The committed shards in `tests/_r_cache/` were generated from the pinned R NNS
14.2 package above under R 4.5.0 and carry `nns_version` `14.2`: every entry was
recomputed with `scripts/regenerate_r_cache.py --fresh`. The cache builder's own
regeneration for the same commit (`automation/r-cache-nns-14.2`, `bb05d3d`)
agrees with it on every entry the two share. To reproduce, install the package
and point `R_LIBS` at its library when regenerating
(`R_LIBS=<lib> python scripts/regenerate_r_cache.py ...`).

Earlier, the R 14.0 update of 13 November 2026 (`9c8bd21`) moved interpreted
loops into compiled code without changing a value that the first 14.0 build
(`7b07a83`) returned; 37 entries that differed then had been written by another
R installation (an R that evaluates `x^3` as `x * x * x`, for one). Generate the
cache with R 4.5.0, as the scope note above requires.

## Automated parity check (with deferred autofix agent)

`NNS-python` automates R-behavior fidelity detection. The automated **fixing
agent is deferred**; for now the chain verifies against live R and hands
divergences to a maintainer via a parity-review PR.

1. **Check** — when upstream `OVVO-Financial/NNS` changes an R API, the
   `inspect-r-api-update` workflow plans which Python modules / parity tests are
   affected and records a report.
2. **Verify** — the `parity-autofix` workflow installs **live R** at the
   recorded R commit and re-runs the mapped parity. If behavior diverged (or
   live R could not be verified), it opens a **parity-review PR** carrying the
   reports for a maintainer to fix.

```text
NNS R API change
  -> inspect-r-api-update.yml          (plan + cache gates + inspection PR)
       -> dispatch nns-parity-divergence   (via OVVO_SYNC_TOKEN)
            -> parity-autofix.yml      (live-R verify -> parity-review PR)
                 -> maintainer applies reviewed fix -> merge
```

### Fix rules (applied by the maintainer; later by the agent)

- Edit **`src/nns/**` only**. Never edit `extern/NNS-core/**`, `tools/NNS/**`,
  or `tests/_r_cache/**` to make a check pass.
- Classify the root cause and act accordingly:
  - **Python port bug** → fix in `src/nns/**`.
  - **R changed behavior** → do not chase a cache value; cache regeneration is a
    separate, reviewed step.
  - **Native kernel change** → do not edit; native code enters Python only
    through accepted `NNS-core` commits.
- Verify against **live R**, not the committed cache.
- **Human-merge only.** The workflow never merges.

### Tokens

Everything runs on a single GitHub secret while the agent is deferred:

| Secret | Kind | Used for | Required? |
| --- | --- | --- | --- |
| `OVVO_SYNC_TOKEN` | GitHub fine-grained PAT — **Contents: R/W**, **Pull requests: R/W** on `NNS-python` | emit the `inspect -> autofix` `repository_dispatch` and open all sync / inspection / parity-review PRs (so they trigger `native-backend-ci`) | yes (workflows fall back to `github.token`, but then PRs won't trigger CI and the auto-chain is skipped) |

Add it at repo **Settings → Secrets and variables → Actions → New repository
secret**. Without it, `inspect-r-api-update` prints the manual trigger command
and you run `parity-autofix` yourself via **workflow_dispatch** (inputs:
`r_commit`, `r_version`).

### Enabling the autofix agent later

When you want the agent to draft fixes automatically, re-add the
`anthropics/claude-code-action` step to `parity-autofix.yml` (gated on a live-R
divergence) with the fix rules above as its prompt, and add an
**`ANTHROPIC_API_KEY`** (`sk-ant-…`) from `console.anthropic.com` (or run
`/install-github-app`). This authenticates the agent to Claude and is **not**
substitutable by a GitHub token. The agent would open a `src/nns/**`-only
parity-correction PR, still human-merged. Bedrock / Vertex are alternatives via
the action's `use_bedrock` / `use_vertex` inputs with OIDC; see the
[cloud providers docs](https://github.com/anthropics/claude-code-action/blob/main/docs/cloud-providers.md).
