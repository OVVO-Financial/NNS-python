# Canonical NNS instructional vignettes

The R package is the source of truth for the NNS curriculum. Its nine numbered
vignettes define the statistical narrative, datasets, section order, examples,
and interpretation. The numbered Python files are section-by-section
instructional ports—not wrappers, API samplers, or smoke tests.

| # | Canonical topic | R source | Python vignette |
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

Each script:

- preserves the R section order and explanatory progression;
- uses the same datasets, with exact local copies of `iris`, `mtcars`, and
  `AirPassengers` committed under `examples/data`;
- reproduces every executable demonstration supported by the Python API;
- saves equivalent figures under `examples/vignettes/output/<script-stem>`;
- prints the returned structures and numerical diagnostics needed to understand
  the example; and
- labels genuine Python API gaps in the section where the R feature appears.

The complete section and gap audit is in [`PARITY.md`](PARITY.md). The
machine-readable acceptance contract is in [`manifest.yml`](manifest.yml).

Run one full vignette:

```bash
uv run python examples/vignettes/07_clustering_and_regression.py
```

Run all nine full vignettes:

```bash
uv run python examples/run_all_vignettes.py
```

CI uses a reduced-runtime mode without removing sections:

```bash
NNS_VIGNETTE_FAST=1 uv run python examples/run_all_vignettes.py
```

The unnumbered scripts remain available as focused implementation examples and
backward-compatible entry points. New canonical material begins in R and is then
ported into the corresponding numbered Python vignette.
