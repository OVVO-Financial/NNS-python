# R API parity review plan

## Changed files

- `R/Multivariate_Regression.R`
- `R/RcppExports.R`
- `R/Stack.R`
- `src/NNS_mreg_predict.cpp`
- `src/RcppExports.cpp`
- `tests/testthat/test-regression-audit-repairs.R`

## Affected Python modules

- `src/nns/stack.py`

## Parity tests to run

- `tests/docs/test_vignette_examples.py`

## Cache scope

- `NNS.stack`

## Required actions

- Fresh cache required: `False`
- Export review required: `False`
- Unmapped R files present: `True`

## Unmapped R files

- `R/Multivariate_Regression.R`
- `R/RcppExports.R`
