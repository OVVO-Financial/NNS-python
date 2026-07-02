# R API parity review plan

## Changed files

- `DESCRIPTION`
- `NNS_13.1.tar.gz`
- `NNS_13.1.zip`
- `R/ARMA.R`
- `R/ARMA_optim.R`
- `R/Multivariate_Regression.R`
- `R/NNS_VAR.R`
- `R/Regression.R`
- `R/Stack.R`

## Affected Python modules

- `pyproject.toml`
- `src/nns/arma.py`
- `src/nns/multivariate_regression.py`
- `src/nns/regression.py`
- `src/nns/stack.py`
- `tests/_r_cache.json`
- `tools/NNS`

## Parity tests to run

- `tests/docs/test_vignette_examples.py`
- `tests/parity`
- `tests/parity/test_practical_examples.py`
- `tests/parity/test_r13_smoke.py`

## Cache scope

- `NNS.ARMA`
- `NNS.ARMA.optim`
- `NNS.M.reg`
- `NNS.VAR`
- `NNS.reg`
- `NNS.stack`

## Required actions

- Fresh cache required: `True`
- Export review required: `False`
- Unmapped R files present: `True`

## Unmapped R files

- `R/ARMA_optim.R`
- `R/Multivariate_Regression.R`
- `R/NNS_VAR.R`
