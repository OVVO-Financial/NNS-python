# R API parity review plan

## Changed files

- `.gitignore`
- `R/Regression.R`

## Affected Python modules

- `src/nns/multivariate_regression.py`
- `src/nns/regression.py`

## Parity tests to run

- `tests/parity/test_practical_examples.py`
- `tests/parity/test_r13_smoke.py`

## Cache scope

- `NNS.M.reg`
- `NNS.reg`

## Required actions

- Fresh cache required: `False`
- Export review required: `False`
- Unmapped R files present: `False`
