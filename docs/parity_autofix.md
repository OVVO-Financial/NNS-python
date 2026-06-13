# Automated parity check and fix

`NNS-python` automates both halves of R-behavior fidelity:

1. **Check** — when upstream `OVVO-Financial/NNS` changes an R API, the
   `inspect-r-api-update` workflow plans which Python modules / parity tests are
   affected and records a report.
2. **Fix if required** — the `parity-autofix` workflow verifies the affected
   public behavior against **live R** at the recorded R commit and, if behavior
   drifted, hands a structured divergence report to an agent
   (`anthropics/claude-code-action`) that drafts a fix and opens a **separate,
   human-reviewed** parity-correction PR.

```text
NNS R API change
  -> inspect-r-api-update.yml          (plan + cache gates + inspection PR)
       -> dispatch nns-parity-divergence
            -> parity-autofix.yml      (live-R verify -> fix or escalate -> PR)
                 -> human review + merge
```

## What the agent may and may not do

Hard rules enforced in the agent prompt and the workflow gate:

* Edit **`src/nns/**` only**. Never edit `extern/NNS-core/**`, `tools/NNS/**`,
  or `tests/_r_cache.json` to make a check pass.
* Classify the root cause and act accordingly:
  * **Python port bug** → fix in `src/nns/**`.
  * **R changed behavior** → do not chase a cache value; escalate (cache
    regeneration is a separate, reviewed step).
  * **Native kernel change** → do not edit; native code enters Python only
    through accepted `NNS-core` commits. Escalate.
* Verify against **live R**, not the committed cache. If live R cannot be
  installed in CI, the workflow opens an **escalation PR** instead of letting
  the agent guess a fix.
* **Human-merge only.** The workflow never merges; every fix PR is reviewed.

## Required setup

### 1. Anthropic credentials (required)

The autofix agent needs an API key exposed as the `ANTHROPIC_API_KEY` repo
secret.

Easiest path — from Claude Code, run:

```text
/install-github-app
```

This installs the official Claude GitHub App, adds the `ANTHROPIC_API_KEY`
secret, and (optionally) scaffolds a workflow. You must be a repo admin.

Manual path:

1. Repo **Settings → Secrets and variables → Actions → New repository secret**.
2. Name `ANTHROPIC_API_KEY`, value = your key from `console.anthropic.com`.

Bedrock / Vertex are supported via the action's `use_bedrock` / `use_vertex`
inputs with OIDC (`id-token: write`) instead of a static key; see the
[cloud providers docs](https://github.com/anthropics/claude-code-action/blob/main/docs/cloud-providers.md).

### 2. Fix-PR token so CI runs on the fix (recommended)

A PR opened with the default `GITHUB_TOKEN` does **not** trigger other workflows
(GitHub's recursion guard), so `native-backend-ci` would not run on the fix PR.
To get CI on fix PRs, let the agent open the PR with a token that does trigger
workflows:

* Installing the **GitHub App** (above) already provides this, or
* add a fine-grained PAT / GitHub App token as the `PARITY_APP_TOKEN` secret;
  the workflow passes it to the action as `github_token`. If unset, it falls
  back to `github.token`.

### 3. Auto-chain token (optional)

`inspect-r-api-update` chains to `parity-autofix` via a `repository_dispatch`,
which the default `GITHUB_TOKEN` cannot emit. To enable the automatic chain, add
a fine-grained PAT with **Contents: read and write** on `NNS-python` as the
`OVVO_SYNC_TOKEN` secret. Without it, `inspect-r-api-update` prints the manual
trigger command and you run `parity-autofix` yourself via **workflow_dispatch**
(inputs: `r_commit`, `r_version`).

## Secret summary

| Secret | Kind | Used for | Required? |
| --- | --- | --- | --- |
| `ANTHROPIC_API_KEY` | Anthropic key (`sk-ant-…`, from console.anthropic.com) | authenticate the agent to Claude | **yes** — a GitHub token does not work here |
| `PARITY_APP_TOKEN` | GitHub fine-grained PAT (Contents R/W, Pull requests R/W) | open the fix PR so CI runs on it | recommended |
| `OVVO_SYNC_TOKEN` | GitHub fine-grained PAT (Contents R/W) | emit the `inspect -> autofix` `repository_dispatch` | optional (manual trigger otherwise) |

## Model

The agent model defaults to `claude-sonnet-4-6` and is overridable via the
`workflow_dispatch` `model` input. Use a more capable model (e.g.
`claude-opus-4-8`) for harder divergences. Every fix is still verified by the
full gate set against live R and reviewed by a human before merge.
