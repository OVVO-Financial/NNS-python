# Live R parity report

- Plan: `sync/last_r_api_plan.json`
- R checkout: `upstream/NNS`
- Fresh cache requested: `False`
- Skip install: `True`
- Live R recompute: `False`

## Result: manual review required

The plan reports unmapped R files. A human must extend `sync/r_api_map.json` before automated parity can run:

- `R/ARMA_optim.R`
- `R/Causal_matrix.R`
- `R/Multivariate_Regression.R`
- `R/NNS_Distance.R`
- `R/NNS_Distance_bulk.R`
- `R/NNS_MC.R`
- `R/NNS_VAR.R`
- `R/NNS_meboot.R`
- `R/Normalization.R`
- `R/Partition_Map.R`
- `R/RcppExports.R`
- `R/SD_Cluster.R`
- `R/dy_d_wrt.R`
- `R/dy_dx.R`
- `R/gvload.R`
- `R/print_methods.R`

## Workflow step outcome

- `run_live_r_parity_for_changed_api.py` step outcome: `failure`
- Fresh cache requested: `false`
- DESCRIPTION changed: `true`
