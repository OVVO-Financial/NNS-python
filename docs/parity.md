# Parity

## Target and status

R NNS 13.0 is the release parity target. It is the tensorized architecture
target; the earlier R NNS 12.1 cache has been superseded. NNS-core is v13.0.0
and remains the native C++ foundation for accelerated partial-moment routines.

Full package parity is **not** claimed. Parity is bounded by the committed tests
and cache:

- `tests/_r_cache.json` — cache-only R result fixtures (2,406 keyed entries,
  schema version `1`, `nns_version == "13.0"`),
- `tests/parity/` — public behavior parity checks,
- `tests/invariants/` — Python-native contracts and invariants, and
- `tests/fixtures/original_tests_expected.json` — adopted original R tests
  (`tests/parity/test_original_*`).

Any cache miss under `PYNNS_R_CACHE_ONLY=1` is a parity-data gap until the cache
is regenerated with Rscript and installed R NNS 13.0.

Plot artifact policy is unchanged: plots and `Rplots.pdf` artifacts are not
parity outputs in pytest; returned values are. See
[`plot_parity_policy.md`](plot_parity_policy.md).

### Known retarget fix

The univariate `NNS.reg(..., multivariate.call = TRUE)` regression-point path
used internally by nonlinear ARMA now follows R NNS 13.0's central-point
weighting: R appends the central regression point again during endpoint
consolidation, and Python preserves that contribution. The airline nonseasonal
nonlinear ARMA smoke forecast changed from the old Python value
`[125.25, 107.75, 158.75, 213.6667]` to the R NNS 13.0 value
`[128.5, 113.5, 155.5, 213.6667]`.

## Verifying parity

```bash
python -m pytest -q tests/invariants
PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity
PYNNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity/test_original_*
ruff check .
mypy
python -m build   # packaging check
```

## Installing R NNS 13.0 from local source

The R package source is vendored in this repository, so NNS is installed from
local source, **never from CRAN**:

- Extracted package directory: `tools/NNS` (`tools/NNS/DESCRIPTION` reports
  `Version: 13.0`).
- Vendored tarball: `tools/NNS_13.0.tar.gz`.

Install with the helper script (prefers `tools/NNS`, falls back to the tarball,
and verifies the loaded version):

```bash
python scripts/install_local_r_nns.py
```

Or run the commands directly:

```bash
R CMD INSTALL tools/NNS
Rscript -e "suppressPackageStartupMessages(library(NNS)); cat(as.character(packageVersion('NNS')))"
# expected output: 13.0
```

Do not run `install.packages("NNS")`; the parity target is the local `tools/NNS`
source, not the CRAN release.

## Regenerating the parity cache

After confirming `packageVersion("NNS") == "13.0"`, regenerate the committed
cache with cache-only/offline toggles unset:

```bash
unset PYNNS_R_CACHE_ONLY PYNNS_OFFLINE CI
python scripts/regenerate_r_cache.py -- -n 0 tests/parity
```

If full regeneration is slow or unstable, regenerate deterministic chunks one
file at a time, for example
`python scripts/regenerate_r_cache.py -- -n 0 tests/parity/test_core.py`, then
continue through the remaining parity files. The committed result must remain a
single valid `tests/_r_cache.json` with `nns_version == "13.0"`,
`schema_version == 1`, and non-empty `entries`; `scripts/regenerate_r_cache.py`
enforces those guardrails after the pytest run.

Validate the regenerated cache offline:

```bash
PYNNS_R_CACHE_ONLY=1 python -m pytest -q -n 0 tests/parity
```

A `RuntimeError: R cache miss ...` means the cache is incomplete (regenerate the
missing live R entries); an `AssertionError`/numeric mismatch means Python
behavior differs from R NNS 13.0 and the Python implementation must be fixed
without loosening tolerances.

Where an R toolchain is unavailable (for example sandboxed CI or
proxy-restricted runners that cannot install R), the cache cannot be regenerated
live; rerun the local-source install and `scripts/regenerate_r_cache.py` on a
host with R.

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
  or `tests/_r_cache.json` to make a check pass.
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
