# R API parity review plan

## Changed files

- `R/Boost.R`
- `R/Stack.R`
- `tests/testthat/test-boost-duplicate-predictors.R`
- `tests/testthat/test-stack-duplicate-predictors.R`

## Affected Python modules

- `src/nns/boost.py`
- `src/nns/stack.py`

## Parity tests to run

- `tests/docs/test_vignette_examples.py`

## Cache scope

- `NNS.boost`
- `NNS.stack`

## Required actions

- Fresh cache required: `False`
- Export review required: `False`
- Unmapped R files present: `False`
