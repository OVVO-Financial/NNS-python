# R API parity review plan

## Changed files

- `R/ARMA.R`
- `R/Dependence.R`
- `R/RcppExports.R`
- `R/Regression.R`
- `src/RcppExports.cpp`
- `src/central_tendencies.cpp`
- `src/central_tendencies.h`
- `src/nns_reg_points.cpp`

## Affected Python modules

- `src/nns/arma.py`
- `src/nns/dependence.py`
- `src/nns/multivariate_regression.py`
- `src/nns/regression.py`

## Parity tests to run

- `tests/docs/test_vignette_examples.py`
- `tests/parity/test_practical_examples.py`
- `tests/parity/test_r13_smoke.py`

## Cache scope

- `NNS.ARMA`
- `NNS.ARMA.optim`
- `NNS.M.reg`
- `NNS.VAR`
- `NNS.copula`
- `NNS.dep`
- `NNS.reg`
- `PM.matrix`

## Required actions

- Fresh cache required: `False`
- Export review required: `False`
- Unmapped R files present: `True`

## Unmapped R files

- `R/RcppExports.R`
