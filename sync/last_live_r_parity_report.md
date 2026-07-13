# Live R parity report

- Plan: `sync/last_r_api_plan.json`
- R checkout: `upstream/NNS`
- Fresh cache requested: `False`
- Skip install: `True`
- Live R recompute: `False`

## Result: fresh cache required

The plan reports `requires_fresh_cache=true` (for example a `DESCRIPTION` version change). Re-run this workflow with `--fresh-cache` / `fresh_cache=true` to regenerate the parity cache from empty against live R.

## Workflow step outcome

- `run_live_r_parity_for_changed_api.py` step outcome: `failure`
- Fresh cache requested: `false`
- DESCRIPTION changed: `true`
