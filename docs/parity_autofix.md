# Automated parity check (with deferred autofix agent)

`NNS-python` automates R-behavior fidelity detection. The automated **fixing
agent is deferred for a later date**; for now the chain verifies against live R
and hands divergences to a maintainer via a parity-review PR (the current
method).

1. **Check** — when upstream `OVVO-Financial/NNS` changes an R API, the
   `inspect-r-api-update` workflow plans which Python modules / parity tests are
   affected and records a report.
2. **Verify** — the `parity-autofix` workflow installs **live R** at the
   recorded R commit and re-runs the mapped parity. If behavior diverged (or live
   R could not be verified), it opens a **parity-review PR** carrying the reports
   for a maintainer to fix.

```text
NNS R API change
  -> inspect-r-api-update.yml          (plan + cache gates + inspection PR)
       -> dispatch nns-parity-divergence   (via OVVO_SYNC_TOKEN)
            -> parity-autofix.yml      (live-R verify -> parity-review PR)
                 -> maintainer applies reviewed fix -> merge
```

## Fix rules (applied by the maintainer; later by the agent)

* Edit **`src/nns/**` only**. Never edit `extern/NNS-core/**`, `tools/NNS/**`,
  or `tests/_r_cache.json` to make a check pass.
* Classify the root cause and act accordingly:
  * **Python port bug** → fix in `src/nns/**`.
  * **R changed behavior** → do not chase a cache value; cache regeneration is a
    separate, reviewed step.
  * **Native kernel change** → do not edit; native code enters Python only
    through accepted `NNS-core` commits.
* Verify against **live R**, not the committed cache.
* **Human-merge only.** The workflow never merges.

## Tokens

Everything runs on a single GitHub secret while the agent is deferred:

| Secret | Kind | Used for | Required? |
| --- | --- | --- | --- |
| `OVVO_SYNC_TOKEN` | GitHub fine-grained PAT — **Contents: R/W**, **Pull requests: R/W** on `NNS-python` | emit the `inspect -> autofix` `repository_dispatch` and open all sync / inspection / parity-review PRs (so they trigger `native-backend-ci`) | yes (workflows fall back to `github.token`, but then PRs won't trigger CI and the auto-chain is skipped) |

Add it at repo **Settings → Secrets and variables → Actions → New repository
secret**. Without it, `inspect-r-api-update` prints the manual trigger command
and you run `parity-autofix` yourself via **workflow_dispatch** (inputs:
`r_commit`, `r_version`).

## Enabling the autofix agent later

When you want the agent to draft fixes automatically, re-add the
`anthropics/claude-code-action` step to `parity-autofix.yml` (gated on a live-R
divergence) with the fix rules above as its prompt, and add:

* **`ANTHROPIC_API_KEY`** — an Anthropic key (`sk-ant-…`) from
  `console.anthropic.com` (or run `/install-github-app`). This authenticates the
  agent to Claude and is **not** substitutable by a GitHub token.

The agent would open a `src/nns/**`-only parity-correction PR, still
human-merged. Bedrock / Vertex are alternatives via the action's `use_bedrock` /
`use_vertex` inputs with OIDC; see the
[cloud providers docs](https://github.com/anthropics/claude-code-action/blob/main/docs/cloud-providers.md).
