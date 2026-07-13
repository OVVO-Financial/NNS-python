# R API parity review plan

## Changed files

- `R/Boost.R`
- `R/Multivariate_Regression.R`
- `R/Partition_Map.R`
- `R/Regression.R`
- `R/Stack.R`

## Affected Python modules

- `src/nns/boost.py`
- `src/nns/multivariate_regression.py`
- `src/nns/regression.py`
- `src/nns/stack.py`

## Parity tests to run

- `tests/docs/test_vignette_examples.py`
- `tests/parity/test_practical_examples.py`
- `tests/parity/test_r13_smoke.py`

## Cache scope

- `NNS.M.reg`
- `NNS.boost`
- `NNS.reg`
- `NNS.stack`

## Required actions

- Fresh cache required: `False`
- Export review required: `False`
- Unmapped R files present: `True`

## Unmapped R files

- `R/Multivariate_Regression.R`
- `R/Partition_Map.R`
