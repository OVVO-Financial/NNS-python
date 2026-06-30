# R API parity review plan

## Changed files

- `DESCRIPTION`
- `NAMESPACE`
- `R/Copula.R`
- `R/Multivariate_Regression.R`
- `R/Partial_Moments.R`
- `R/gvload.R`
- `doc/NNSvignette_01_Overview.html`
- `src/NNS.dll`
- `vignettes/NNSvignette_01_Overview.html`

## Affected Python modules

- `pyproject.toml`
- `src/nns/__init__.py`
- `src/nns/partial_moments.py`
- `src/nns/var.py`
- `tests/_r_cache.json`
- `tools/NNS`

## Parity tests to run

- `tests/invariants`
- `tests/parity`
- `tests/parity/test_r13_smoke.py`

## Cache scope

- `LPM`
- `LPM.VaR`
- `LPM.ratio`
- `UPM`
- `UPM.VaR`
- `UPM.ratio`

## Required actions

- Fresh cache required: `True`
- Export review required: `True`
- Unmapped R files present: `True`

## Unmapped R files

- `R/Copula.R`
- `R/Multivariate_Regression.R`
- `R/gvload.R`
