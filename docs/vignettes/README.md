# NNS Python vignettes

Python translations of the R NNS vignette curriculum, written against the
public `nns` API. Each topic has a Markdown explainer here and a matching
runnable script under [`examples/vignettes/`](../../examples/vignettes).

These examples are Python translations of the R NNS vignette curriculum
(the vendored sources under `tools/NNS/vignettes` and `tools/NNS/inst/doc`).
The code assumes the fresh R NNS 13.0 parity fixes from PR #3 (or the latest
`main` after PR #3 merged).

## Contents

| Vignette | Topic |
| --- | --- |
| [00 Overview](00_overview.md) | What NNS is and a one-screen tour of every pillar. |
| [01 Partial moments](01_partial_moments.md) | LPM/UPM, variance and CDF reconstruction, value-at-risk. |
| [02 Descriptive & distributional tools](02_descriptive_distributional_tools.md) | Moments, modes, covariance from partial moment matrices, quantile tables. |
| [03 Dependence & nonlinear association](03_dependence_nonlinear_association.md) | Partial-moment dependence vs Pearson correlation, copulas. |
| [04 Normalization & rescaling](04_normalization_rescaling.md) | `nns_norm` and `nns_rescale` (min-max and risk-neutral). |
| [05 Hypothesis: ANOVA & stochastic superiority](05_hypothesis_anova_stochastic_superiority.md) | `nns_anova` certainty and `nns_ss` superiority probabilities. |
| [06 Regression, boosting, stacking, causality](06_regression_boosting_stacking_causality.md) | `nns_reg`, `nns_boost`, `nns_stack`, `nns_causation`. |
| [07 Time series forecasting](07_time_series_forecasting.md) | `nns_seas`, `nns_arma`, `nns_arma_optim`, `nns_var`. |
| [08 Simulation, bootstrap, risk-neutral](08_simulation_bootstrap_riskneutral.md) | `nns_meboot` and `nns_mc`. |
| [09 Portfolios & stochastic dominance](09_portfolio_stochastic_dominance.md) | `fsd_uni`/`ssd_uni`/`tsd_uni`, `sd_efficient_set`, `nns_sd_cluster`. |

## Running the examples

Every script is self-contained and deterministic where possible (seeded RNG,
small data, no plotting in the default path):

```bash
python examples/vignettes/partial_moments.py
```

The whole set is exercised by `tests/docs/test_vignette_examples.py`, which
runs each script and fails on a nonzero exit code:

```bash
python -m pytest -q tests/docs/test_vignette_examples.py
```

## A note on stochastic outputs

Bootstrap and Monte Carlo routines (`nns_meboot`, `nns_mc`, the `nns_ss`
confidence interval, the `nns_anova` robust interval) produce sampled outputs.
The vignettes and their tests validate these by structure and range, never by
exact value. Deterministic routines (partial moments, dependence, regression
points, the numeric ARMA/stack/boost designs shown here) are compared exactly.
