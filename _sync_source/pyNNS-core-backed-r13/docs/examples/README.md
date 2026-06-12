# PyNNS Examples

These examples are Python-native companions to the upstream R NNS documentation,
not one-for-one copies of the R reports. Each script is runnable,
deterministic, topic-focused, and covered by `tests/invariants/test_examples.py`.

The upstream R repository contains several kinds of material:

- `reference/NNS/man/`: function reference pages.
- `reference/NNS/doc/` and `reference/NNS/vignettes/`: CRAN-style tutorials.
- `reference/NNS/book/`: conceptual book chapters.
- `reference/NNS/examples/`: larger applied reports, PDFs, HTML demos, and case
  studies.

Use those upstream files as conceptual references. Use the examples here when
you want short Python call patterns that are kept in sync with PyNNS.

## Runnable Examples

| Topic | Script | What it demonstrates | Upstream analogue |
|---|---|---|---|
| Partial moments | [partial_moments.py](partial_moments.py) | `lpm`, `upm`, degree-zero probability split, variance decomposition, `nns_moments` | `NNSvignette_Partial_Moments.Rmd` |
| Dependence | [dependence.py](dependence.py) | `nns_dep`, `nns_cor`, linear vs nonlinear relationships | `NNSvignette_Correlation_and_Dependence.Rmd` |
| Distributions / ANOVA | [distributions_anova.py](distributions_anova.py) | `nns_cdf`, `nns_anova`, certainty output | `NNSvignette_Comparing_Distributions.Rmd` |
| Regression | [regression.py](regression.py) | `nns_reg`, fitted values, point estimates, regression output shape | `NNSvignette_Clustering_and_Regression.Rmd` |
| Classification | [classification.py](classification.py) | `nns_reg(..., type="class")`, numeric class-code predictions | `NNSvignette_Classification.Rmd` |
| Forecasting | [forecasting.py](forecasting.py) | `nns_arma`, `nns_arma_optim`, `nns_var` | `NNSvignette_Forecasting.Rmd` |
| Nowcast panel | [nowcast_panel.py](nowcast_panel.py) | deterministic user-supplied panel, date metadata, VAR-backed forecast output | `NNS.VAR` nowcast/frequency-alignment material |

## Notebooks

| Topic | Notebook |
|---|---|
| Partial-moment risk workflow | [01_partial_moments_risk_workflow.ipynb](notebooks/01_partial_moments_risk_workflow.ipynb) |
| Regression, classification, factors | [02_regression_classification_workflow.ipynb](notebooks/02_regression_classification_workflow.ipynb) |
| Forecasting and local nowcast panel | [03_forecasting_nowcast_workflow.ipynb](notebooks/03_forecasting_nowcast_workflow.ipynb) |
| Distribution, dominance, simulation | [04_distribution_dominance_simulation_workflow.ipynb](notebooks/04_distribution_dominance_simulation_workflow.ipynb) |
| Boston Housing regression parity example | [05_boston_housing_regression_workflow.ipynb](notebooks/05_boston_housing_regression_workflow.ipynb) |

Run one example:

```bash
uv run python docs/examples/partial_moments.py
```

Run all examples:

```bash
for example in docs/examples/*.py; do uv run python "$example"; done
```

The main R parity guarantees still live in `tests/parity/`. These examples and
notebooks are usage references, not a replacement for the parity suite.
