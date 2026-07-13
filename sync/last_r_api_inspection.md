# R API parity review plan

## Changed files

- `DESCRIPTION`
- `NAMESPACE`
- `NNS_13.1.tar.gz`
- `NNS_13.1.zip`
- `R/Multivariate_Regression.R`
- `R/Partition_Map.R`
- `R/RcppExports.R`
- `R/Regression.R`
- `R/Stack.R`
- `R/gvload.R`
- `doc/NNSvignette_01_Overview.R`
- `doc/NNSvignette_01_Overview.Rmd`
- `doc/NNSvignette_01_Overview.html`
- `doc/NNSvignette_03_Correlation_and_Dependence.html`
- `doc/NNSvignette_07_Clustering_and_Regression.R`
- `doc/NNSvignette_07_Clustering_and_Regression.Rmd`
- `doc/NNSvignette_07_Clustering_and_Regression.html`
- `doc/NNSvignette_08_Classification.R`
- `doc/NNSvignette_08_Classification.Rmd`
- `doc/NNSvignette_08_Classification.html`
- `doc/NNSvignette_09_Forecasting.html`
- `man/NNS.boost.Rd`
- `man/NNS.part.Rd`
- `man/NNS.stack.Rd`
- `src/NNS.dll`
- `src/RcppExports.cpp`
- `tests/testthat/Rplots.pdf`
- `vignettes/NNSvignette_01_Overview.R`
- `vignettes/NNSvignette_01_Overview.Rmd`
- `vignettes/NNSvignette_01_Overview.html`
- `vignettes/NNSvignette_03_Correlation_and_Dependence.html`
- `vignettes/NNSvignette_07_Clustering_and_Regression.R`
- `vignettes/NNSvignette_07_Clustering_and_Regression.Rmd`
- `vignettes/NNSvignette_07_Clustering_and_Regression.html`
- `vignettes/NNSvignette_08_Classification.R`
- `vignettes/NNSvignette_08_Classification.Rmd`
- `vignettes/NNSvignette_08_Classification.html`
- `vignettes/NNSvignette_09_Forecasting.html`

## Affected Python modules

- `pyproject.toml`
- `src/nns/__init__.py`
- `src/nns/multivariate_regression.py`
- `src/nns/regression.py`
- `src/nns/stack.py`
- `tests/_r_cache.json`
- `tools/NNS`

## Parity tests to run

- `tests/docs/test_vignette_examples.py`
- `tests/parity`
- `tests/parity/test_practical_examples.py`
- `tests/parity/test_r13_smoke.py`

## Cache scope

- `NNS.M.reg`
- `NNS.reg`
- `NNS.stack`

## Required actions

- Fresh cache required: `True`
- Export review required: `True`
- Unmapped R files present: `True`

## Unmapped R files

- `R/Multivariate_Regression.R`
- `R/Partition_Map.R`
- `R/RcppExports.R`
- `R/gvload.R`
