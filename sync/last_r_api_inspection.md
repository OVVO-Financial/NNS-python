# R API parity review plan

## Changed files

- `DESCRIPTION`
- `NNS_13.0.tar.gz`
- `NNS_13.0.zip`
- `R/ARMA.R`
- `src/NNS.dll`

## Affected Python modules

- `pyproject.toml`
- `src/nns/arma.py`
- `tests/_r_cache.json`
- `tools/NNS`

## Parity tests to run

- `tests/parity`
- `tests/parity/test_practical_examples.py`
- `tests/parity/test_r13_smoke.py`

## Cache scope

- `NNS.ARMA`
- `NNS.ARMA.optim`
- `NNS.VAR`

## Required actions

- Fresh cache required: `True`
- Export review required: `False`
- Unmapped R files present: `False`
