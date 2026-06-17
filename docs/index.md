# NNS Python

[![PyPI package](https://img.shields.io/pypi/v/ovvo-nns?label=ovvo-nns&color=blue)](https://pypi.org/project/ovvo-nns/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0--only-blue)](https://github.com/OVVO-Financial/NNS-python/blob/main/LICENSE)

`ovvo-nns` brings **Nonlinear Nonparametric Statistics** to Python as the `nns`
import package. It is a parity-focused port of the R `NNS` 13.0+ package,
designed for real-world data that violate symmetry, linearity, or distributional
assumptions.

NNS is built around partial moments — the lower and upper components of variance —
and uses them across nonlinear dependence, correlation, causation, regression,
classification, forecasting, stochastic dominance, stochastic superiority, Monte
Carlo simulation, and numerical differentiation workflows.

!!! note "Origin"
    NNS was created by Fred Viole as the companion R package to Viole, F. and
    Nawrocki, D. (2013), *Nonlinear Nonparametric Statistics: Using Partial
    Moments*. **Book (2nd Edition):** <https://ovvo-financial.github.io/NNS/book/>.
    For a direct quantitative finance implementation of NNS, see
    [OVVO Labs](https://www.ovvolabs.com).

## Package at a glance

| Item | Value |
|---|---|
| Distribution package | `ovvo-nns` |
| Import package | `nns` |
| Python | `>=3.11` |
| Required runtime dependencies | NumPy, SciPy |
| R required at runtime | No |
| Native acceleration | Private, optional `nns._nnscore` kernels where available |
| Public API status | Stable, parity-focused |
| License | GPL-3.0-only |

The public package is Python-native and does not call R at runtime. Some core
kernels can use the private `_nnscore` extension when it is present, while public
functions keep Python implementations and explicit fallback behavior.

## Get started

<div class="grid cards" markdown>

-   :material-download: **[Install](install.md)**

    `pip install ovvo-nns`, then `import nns`.

-   :material-rocket-launch: **[Quick start](quick_start.md)**

    Partial moments, dependence, regression, and forecasting in a few lines.

-   :material-book-open-variant: **[API reference](api_reference.md)**

    Function-by-function index with R `NNS` name crosswalks.

-   :material-check-decagram: **[API status](api_status.md)**

    Implemented, partial, guarded, and known-gap paths.

</div>

## Main API areas

| Area | Representative functions |
|---|---|
| Partial moments | `lpm`, `upm`, `lpm_ratio`, `upm_ratio`, `pm_matrix` |
| Classical moment helpers | `mean_pm`, `var_pm`, `skew_pm`, `kurt_pm`, `nns_moments` |
| Dependence, correlation, copula | `nns_dep`, `nns_cor`, `nns_copula` |
| Causation | `nns_causation`, `causal_matrix` |
| Regression and classification | `nns_reg`, `nns_m_reg`, `nns_stack`, `nns_boost` |
| Forecasting | `nns_seas`, `nns_arma`, `nns_arma_optim`, `nns_var` |
| Distribution tools | `nns_cdf`, `nns_anova`, `nns_norm` |
| Stochastic dominance | `fsd`, `ssd`, `tsd`, `nns_sd_cluster`, `sd_efficient_set` |
| Stochastic superiority and simulation | `nns_ss`, `nns_mc`, `nns_meboot` |
| Differentiation | `nns_diff`, `dy_dx`, `dy_d` |
| Categorical helpers | `encode_factor_codes`, `factor_2_dummy`, `factor_2_dummy_fr`, `prepare_factor_predictors` |

See [API status](api_status.md) for implemented, partial, guarded, and known-gap
paths.

## Design boundaries

NNS Python prioritizes stable public behavior from installed R NNS 13.0+, not
private helper parity. The package returns NumPy arrays and plain dictionaries
rather than R `data.table` objects, uses explicit Python errors for several
unsafe R coercions, and generally ignores plotting side effects.

See [behavior conventions](conventions.md) for detailed compatibility notes and
[parity with R NNS](parity.md) for the parity target and automation.

## Attribution

Upstream R package and reference implementation:
[OVVO-Financial/NNS](https://github.com/OVVO-Financial/NNS).
