# R API parity review plan

## Changed files

- `DESCRIPTION`
- `NNS_13.0.tar.gz`
- `NNS_13.0.zip`
- `NNS_13.1.tar.gz`
- `NNS_13.1.zip`
- `R/ARMA_optim.R`
- `R/NNS_VAR.R`
- `README.md`
- `man/NNS.ARMA.optim.Rd`
- `man/NNS.VAR.Rd`
- `src/NNS.dll`

## Affected Python modules

- `pyproject.toml`
- `tests/_r_cache.json`
- `tools/NNS`

## Parity tests to run

- `tests/parity`

## Cache scope

- None mapped

## Required actions

- Fresh cache required: `True`
- Export review required: `False`
- Unmapped R files present: `True`

## Unmapped R files

- `R/ARMA_optim.R`
- `R/NNS_VAR.R`
