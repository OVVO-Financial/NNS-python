# R API parity review plan

## Changed files

- `NNS_13.0.tar.gz`
- `NNS_13.0.zip`
- `R/ARMA.R`
- `src/NNS.dll`
- `tests/testthat/Rplots.pdf`

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
- Unmapped R files present: `False`
