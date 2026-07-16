# Canonical NNS examples

The R package is the source of truth for the NNS example curriculum. Its nine
numbered vignettes define the statistical narrative, datasets, section order,
and interpretation. Python implements the same public workflows with
language-appropriate containers and syntax.

| # | Canonical topic | R source | Python entry point |
|---|---|---|---|
| 01 | Overview | `NNSvignette_01_Overview.Rmd` | `01_overview.py` |
| 02 | Partial Moments | `NNSvignette_02_Partial_Moments.Rmd` | `02_partial_moments.py` |
| 03 | Correlation and Dependence | `NNSvignette_03_Correlation_and_Dependence.Rmd` | `03_correlation_and_dependence.py` |
| 04 | Normalization and Rescaling | `NNSvignette_04_Normalization_and_Rescaling.Rmd` | `04_normalization_and_rescaling.py` |
| 05 | Sampling and Simulation | `NNSvignette_05_Sampling.Rmd` | `05_sampling_and_simulation.py` |
| 06 | Comparing Distributions | `NNSvignette_06_Comparing_Distributions.Rmd` | `06_comparing_distributions.py` |
| 07 | Clustering and Regression | `NNSvignette_07_Clustering_and_Regression.Rmd` | `07_clustering_and_regression.py` |
| 08 | Classification | `NNSvignette_08_Classification.Rmd` | `08_classification.py` |
| 09 | Forecasting | `NNSvignette_09_Forecasting.Rmd` | `09_forecasting.py` |

The unnumbered scripts remain available as focused implementation examples and
backward-compatible entry points. New canonical material should be added to R
first and then ported to the corresponding numbered Python example.

Run one example:

```bash
uv run python examples/vignettes/02_partial_moments.py
```

Run the full example suite:

```bash
uv run python examples/run_all_vignettes.py
```
