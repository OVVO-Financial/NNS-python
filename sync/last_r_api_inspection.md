# R API parity review plan

## Changed files

- `R/ARMA.R`
- `R/RcppExports.R`
- `src/RcppExports.cpp`
- `src/internal_functions.cpp`

## Affected Python modules

- `src/nns/arma.py`

## Parity tests to run

- `tests/parity/test_practical_examples.py`
- `tests/parity/test_r13_smoke.py`

## Cache scope

- `NNS.ARMA`
- `NNS.ARMA.optim`
- `NNS.VAR`

## Required actions

- Fresh cache required: `False`
- Export review required: `False`
- Unmapped R files present: `True`

## Unmapped R files

- `R/RcppExports.R`
