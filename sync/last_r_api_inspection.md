# R API parity review plan

## Changed files

- `DESCRIPTION`
- `NNS_13.0.tar.gz`
- `NNS_13.0.zip`
- `R/NNS_meboot.R`
- `doc/NNSvignette_05_Sampling.R`
- `doc/NNSvignette_05_Sampling.Rmd`
- `doc/NNSvignette_05_Sampling.html`
- `man/NNS.meboot.Rd`
- `src/NNS.dll`
- `vignettes/NNSvignette_05_Sampling.R`
- `vignettes/NNSvignette_05_Sampling.Rmd`
- `vignettes/NNSvignette_05_Sampling.html`

## Affected Python modules

- `pyproject.toml`
- `tests/_r_cache.json`
- `tools/NNS`

## Parity tests to run

- `tests/parity`

## Cache scope

- None mapped

## Required actions

- Fresh cache required: `True`
- Export review required: `False`
- Unmapped R files present: `True`

## Unmapped R files

- `R/NNS_meboot.R`
