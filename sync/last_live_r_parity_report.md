# Live R parity report

- Plan: `sync/last_r_api_plan.json`
- R checkout: `upstream/NNS`
- Fresh cache requested: `False`
- Skip install: `True`
- Live R recompute: `False`

## Result: manual review required

The plan reports unmapped R files. A human must extend `sync/r_api_map.json` before automated parity can run:

- `R/NNS_meboot.R`

## Workflow step outcome

- `run_live_r_parity_for_changed_api.py` step outcome: `failure`
- Fresh cache requested: `false`
- DESCRIPTION changed: `false`
