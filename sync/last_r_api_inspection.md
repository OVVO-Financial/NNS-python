# R API parity review plan

## Changed files

- `NNS_13.1.tar.gz`
- `NNS_13.1.zip`
- `R/Boost.R`
- `R/Stack.R`
- `man/NNS.boost.Rd`
- `man/NNS.stack.Rd`
- `src/NNS.dll`
- `tests/testthat/Rplots.pdf`

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
