# NNS Python API Reference Manual

This is the CRAN-style reference manual for `ovvo-nns`, the Python package imported as `nns`.

It complements the README, examples, conventions, and API status notes with a function-by-function public API index. The public Python export list in `src/nns/__init__.py` is the source of truth for this manual.

## How to read this manual

Each entry gives:

- the public Python name,
- the closest R `NNS` API name where there is one,
- the current implementation status from `docs/api_status.md`,
- the purpose and return convention to expect in Python.

Python names follow snake_case. R names generally use dot-separated names, for example `NNS.dep` in R becomes `nns_dep` in Python.

NNS Python intentionally returns NumPy arrays, Python scalars, dataclasses, and plain dictionaries rather than R `data.table` objects. Some R behaviors that silently coerce, truncate, warn, or return unusable values are explicit Python errors. See `docs/conventions.md` for detailed compatibility notes.

## Regenerating this reference

Run this after changing public exports, signatures, or docstrings:

```bash
uv run python scripts/generate_api_reference.py
```

The generator introspects `nns.__all__`, imports each public object, records signatures where available, and merges the API-area metadata below into a reproducible Markdown reference.

## Public API crosswalk

| Area | Python API | Closest R NNS API | Status | Notes |
|---|---|---|---|---|
| Partial moments | `lpm` | `LPM` | implemented | Lower partial moment. |
| Partial moments | `upm` | `UPM` | implemented | Upper partial moment. |
| Partial moments | `lpm_ratio` | `LPM.ratio` | implemented | Lower partial moment ratio. |
| Partial moments | `upm_ratio` | `UPM.ratio` | implemented | Upper partial moment ratio. |
| Partial moments | `pm_matrix` | `PM.matrix` | implemented | Partial moment matrix helper. |
| Co-moments | `co_lpm` | `Co.LPM` | implemented | Pairwise co-lower partial moment. |
| Co-moments | `co_upm` | `Co.UPM` | implemented | Pairwise co-upper partial moment. |
| Co-moments | `d_lpm` | `D.LPM` | implemented | Pairwise distance lower partial moment. |
| Co-moments | `d_upm` | `D.UPM` | implemented | Pairwise distance upper partial moment. |
| Co-moments | `co_lpm_nd` | `Co.LPM_nD` | implemented | N-dimensional co-lower partial moment wrapper. |
| Co-moments | `co_upm_nd` | `Co.UPM_nD` | implemented | N-dimensional co-upper partial moment wrapper. |
| Co-moments | `dpm_nd` | `DPM_nD` | implemented | N-dimensional directional partial moment wrapper. |
| Classical moments | `ecdf_pm` | ECDF partial-moment helper | implemented | Empirical distribution helper used by moment routines. |
| Classical moments | `mean_pm` | `Mean` partial-moment helper | implemented | Mean via partial-moment conventions. |
| Classical moments | `var_pm` | `Variance` partial-moment helper | implemented | Variance via partial-moment conventions. |
| Classical moments | `skew_pm` | `Skewness` partial-moment helper | implemented | Skewness via partial-moment conventions. |
| Classical moments | `kurt_pm` | `Kurtosis` partial-moment helper | implemented | Kurtosis via partial-moment conventions. |
| Classical moments | `nns_moments` | `NNS.moments` | implemented | Bundled NNS moment diagnostics. |
| Dependence | `nns_dep` | `NNS.dep` | implemented | Nonlinear dependence and correlation. |
| Dependence | `nns_cor` | `NNS.cor` | implemented | Nonlinear correlation convenience API. |
| Dependence | `nns_copula` | `NNS.copula` | implemented | Bivariate copula surface. |
| Causation | `nns_causation` | `NNS.caus` | implemented | Directional causation. |
| Causation | `causal_matrix` | `Causal.matrix` | implemented | Matrix form of causation relationships. |
| Regression | `nns_reg` | `NNS.reg` | implemented | Bivariate regression, classification, and interval surfaces. |
| Regression | `nns_m_reg` | `NNS.M.reg` | partial | Multivariate regression. Raw factor expansion is guarded. |
| Regression | `nns_stack` | `NNS.stack` | implemented | Stacked ensemble path. |
| Regression | `nns_boost` | `NNS.boost` | partial | Boosted ensemble path with one high-feature stochastic threshold guard. |
| Regression helpers | `FactorDesign` | Python helper | implemented | Dataclass returned by factor-preparation helpers. |
| Regression helpers | `prepare_factor_predictors` | Factor preparation path | implemented | Builds regression-ready factor design matrices. |
| Categorical helpers | `encode_factor_codes` | Factor-code helper | implemented | Encodes categorical labels into deterministic numeric codes. |
| Categorical helpers | `factor_2_dummy` | `factor.2.dummy` | implemented | Dummy expansion helper. |
| Categorical helpers | `factor_2_dummy_fr` | full-rank dummy helper | implemented | Full-rank dummy expansion helper. |
| Forecasting | `nns_seas` | `NNS.seas` | implemented | Seasonality detection. |
| Forecasting | `nns_arma` | `NNS.ARMA` | partial | Univariate ARMA-style forecast surface. |
| Forecasting | `nns_arma_optim` | `NNS.ARMA.optim` | partial | ARMA optimizer surface. |
| Forecasting | `nns_var` | `NNS.VAR` | partial | Multivariate VAR-style forecast surface. |
| Distribution and ANOVA | `nns_cdf` | `NNS.cdf` | implemented | NNS empirical CDF path. |
| Distribution and ANOVA | `nns_anova` | `NNS.ANOVA` | implemented | Binary, multi-group, and pairwise ANOVA-style comparisons. |
| Distribution and ANOVA | `nns_norm` | `NNS.norm` | implemented | NNS normalization helper. |
| Distance and partitioning | `nns_distance` | `NNS.distance` | implemented | Single distance computation. |
| Distance and partitioning | `nns_distance_bulk` | `NNS.distance.bulk` | implemented | Bulk distance computation. |
| Distance and partitioning | `nns_part` | `NNS.part` | implemented | Partitioning helper returning Python-native structures. |
| Central tendencies | `nns_gravity` | `NNS.gravity` | implemented | NNS gravity center helper. |
| Central tendencies | `nns_mode` | `NNS.mode` | implemented | NNS mode helper. |
| Central tendencies | `nns_rescale` | `NNS.rescale` | implemented | Rescaling helper. |
| Stochastic dominance | `fsd` | `NNS.FSD` | implemented | First-order stochastic dominance. |
| Stochastic dominance | `ssd` | `NNS.SSD` | implemented | Second-order stochastic dominance. |
| Stochastic dominance | `tsd` | `NNS.TSD` | implemented | Third-order stochastic dominance. |
| Stochastic dominance | `fsd_uni` | `NNS.FSD.uni` | implemented | Univariate FSD wrapper. |
| Stochastic dominance | `ssd_uni` | `NNS.SSD.uni` | implemented | Univariate SSD wrapper. |
| Stochastic dominance | `tsd_uni` | `NNS.TSD.uni` | implemented | Univariate TSD wrapper. |
| Stochastic dominance | `nns_sd_cluster` | `NNS.SD.cluster` | implemented | Stochastic-dominance clustering. |
| Stochastic dominance | `sd_efficient_set` | `SD.efficient.set` | implemented | Stochastic-dominance efficient set. |
| Stochastic superiority | `nns_ss` | `NNS.SS` | implemented | Stochastic superiority. |
| Simulation | `nns_mc` | `NNS.MC` | implemented | Monte Carlo helper. |
| Simulation | `nns_meboot` | `NNS.meboot` | implemented | Maximum-entropy bootstrap helper. |
| Differentiation | `nns_diff` | `NNS.diff` | implemented | Numerical differentiation. |
| Differentiation | `dy_dx` | `dy.dx` | implemented | Scalar derivative helper. |
| Differentiation | `dy_d` | `dy.d_` | partial | Multivariate derivative helper. |
| VaR helpers | `lpm_var` | `LPM.VaR` | implemented | Lower partial-moment VaR helper. |
| VaR helpers | `upm_var` | `UPM.VaR` | implemented | Upper partial-moment VaR helper. |

## Reference by API area

### Partial moments

These functions are the core building blocks for lower and upper components of variance and related partial-moment ratios.

#### `lpm`

Closest R API: `LPM`.

Computes a lower partial moment for a numeric vector at a scalar or vector target. Degree zero follows the R NNS equality convention by counting observations less than or equal to the target.

Returns a Python `float` for scalar targets and a NumPy array for vector targets.

#### `upm`

Closest R API: `UPM`.

Computes an upper partial moment for a numeric vector at a scalar or vector target. Degree zero counts observations greater than the target.

Returns a Python `float` for scalar targets and a NumPy array for vector targets.

#### `lpm_ratio`

Closest R API: `LPM.ratio`.

Computes the lower partial-moment share of total lower plus upper partial moment mass.

#### `upm_ratio`

Closest R API: `UPM.ratio`.

Computes the upper partial-moment share of total lower plus upper partial moment mass.

#### `pm_matrix`

Closest R API: `PM.matrix`.

Builds a partial-moment matrix surface used by dependence and co-moment routines.

### Co-moments and n-dimensional partial moments

#### `co_lpm`

Closest R API: `Co.LPM`.

Computes pairwise co-lower partial moments. NNS Python raises explicit errors for incompatible lengths rather than relying on R recycling or truncation behavior.

#### `co_upm`

Closest R API: `Co.UPM`.

Computes pairwise co-upper partial moments.

#### `d_lpm`

Closest R API: `D.LPM`.

Computes pairwise lower directional partial moments.

#### `d_upm`

Closest R API: `D.UPM`.

Computes pairwise upper directional partial moments.

#### `co_lpm_nd`, `co_upm_nd`, `dpm_nd`

N-dimensional wrappers around the public co-moment and directional partial-moment concepts.

### Classical moment helpers

#### `ecdf_pm`

Empirical CDF helper used by partial-moment-based moment routines.

#### `mean_pm`

Computes mean using the NNS partial-moment convention.

#### `var_pm`

Computes variance using the NNS partial-moment convention.

#### `skew_pm`

Computes skewness using the NNS partial-moment convention.

#### `kurt_pm`

Computes kurtosis using the NNS partial-moment convention.

#### `nns_moments`

Returns bundled NNS moment diagnostics.

### Dependence, correlation, copula, and causation

#### `nns_dep`

Closest R API: `NNS.dep`.

Returns NNS nonlinear dependence and correlation for bivariate input, or matrix-style dependence and correlation results for matrix input.

#### `nns_cor`

Closest R API: `NNS.cor`.

Convenience API for the NNS nonlinear correlation surface.

#### `nns_copula`

Closest R API: `NNS.copula`.

Computes the bivariate scalar copula surface supported by NNS Python.

#### `nns_causation`

Closest R API: `NNS.caus`.

Computes directional causation relationships, including supported numeric lag and time-series paths.

#### `causal_matrix`

Closest R API: `Causal.matrix`.

Computes a matrix representation of causation relationships.

### Regression, classification, and ensembles

#### `nns_reg`

Closest R API: `NNS.reg`.

Bivariate NNS regression and classification. Supports numeric, class-code, confidence interval, smoothing, dimension-reduction, and public factor-expansion paths.

Returns Python-native structures, generally dictionaries containing estimates, diagnostics, residuals, intervals, and related arrays.

#### `nns_m_reg`

Closest R API: `NNS.M.reg`.

Multivariate regression and classification surface. Numeric and class paths are implemented. Direct raw-factor `factor_2_dummy=True` is guarded because the installed R direct raw-factor path errors; use `prepare_factor_predictors(...)` first.

#### `nns_stack`

Closest R API: `NNS.stack`.

Stacked ensemble API for numeric and classification workflows.

#### `nns_boost`

Closest R API: `NNS.boost`.

Boosted ensemble API. Deterministic and stochastic structures are implemented. The high-feature stochastic threshold path is guarded to make an installed-R failure explicit.

### Categorical and factor helpers

#### `FactorDesign`

Python helper dataclass used to hold prepared factor design matrices and metadata.

#### `prepare_factor_predictors`

Builds a regression-ready full-rank factor design matrix before calling `nns_m_reg(...)`.

#### `encode_factor_codes`

Encodes labels into deterministic numeric factor codes. Pass explicit levels when class or factor ordering matters.

#### `factor_2_dummy`

Builds dummy variables from factor inputs.

#### `factor_2_dummy_fr`

Builds full-rank dummy variables from factor inputs.

### Forecasting

#### `nns_seas`

Closest R API: `NNS.seas`.

Detects seasonality and supported modulo structures in a univariate series.

#### `nns_arma`

Closest R API: `NNS.ARMA`.

Univariate ARMA-style forecasting surface. Stochastic interval streams use NumPy RNG and are structural or statistical parity targets rather than exact stream parity.

#### `nns_arma_optim`

Closest R API: `NNS.ARMA.optim`.

Optimization surface for NNS ARMA-style forecasting.

#### `nns_var`

Closest R API: `NNS.VAR`.

Multivariate VAR-style forecast surface with supported dimension-reduction paths.

### Distribution, ANOVA, normalization, and distance

#### `nns_cdf`

Closest R API: `NNS.cdf`.

NNS empirical CDF path. Deterministic non-plotting behavior is implemented.

#### `nns_anova`

Closest R API: `NNS.ANOVA`.

NNS ANOVA-style comparison helper covering binary, multi-group, pairwise, and degenerate conventions.

#### `nns_norm`

Closest R API: `NNS.norm`.

NNS normalization helper for numeric matrix-style inputs.

#### `nns_distance`

Closest R API: `NNS.distance`.

Computes a single NNS distance result.

#### `nns_distance_bulk`

Closest R API: `NNS.distance.bulk`.

Computes distance results in bulk.

#### `nns_part`

Closest R API: `NNS.part`.

Partitioning helper. Returns plain Python structures instead of R `data.table` objects.

### Central tendencies

#### `nns_gravity`

Closest R API: `NNS.gravity`.

Computes the NNS gravity center helper.

#### `nns_mode`

Closest R API: `NNS.mode`.

Computes the NNS mode helper.

#### `nns_rescale`

Closest R API: `NNS.rescale`.

Rescales inputs using NNS conventions.

### Stochastic dominance, superiority, and simulation

#### `fsd`, `ssd`, `tsd`

Closest R APIs: `NNS.FSD`, `NNS.SSD`, and `NNS.TSD`.

Compute first-, second-, and third-order stochastic dominance.

#### `fsd_uni`, `ssd_uni`, `tsd_uni`

Closest R APIs: `NNS.FSD.uni`, `NNS.SSD.uni`, and `NNS.TSD.uni`.

Univariate wrappers for stochastic dominance workflows.

#### `nns_sd_cluster`

Closest R API: `NNS.SD.cluster`.

Stochastic-dominance clustering helper.

#### `sd_efficient_set`

Closest R API: `SD.efficient.set`.

Computes the stochastic-dominance efficient set.

#### `nns_ss`

Closest R API: `NNS.SS`.

Computes stochastic superiority.

#### `nns_mc`

Closest R API: `NNS.MC`.

Monte Carlo simulation helper.

#### `nns_meboot`

Closest R API: `NNS.meboot`.

Maximum-entropy bootstrap helper.

### Differentiation

#### `nns_diff`

Closest R API: `NNS.diff`.

NNS numerical differentiation surface.

#### `dy_dx`

Closest R API: `dy.dx`.

Scalar derivative helper for overall and pointwise evaluation modes.

#### `dy_d`

Closest R API: `dy.d_`.

Multivariate derivative helper. Scalar and vectorized point and distribution modes are covered. Multi-row mixed derivative point matrices use pointwise Python semantics rather than R's order-dependent list-matrix packing quirk.

### VaR helpers

#### `lpm_var`

Closest R API: `LPM.VaR`.

Lower partial-moment VaR helper used by deterministic interval paths.

#### `upm_var`

Closest R API: `UPM.VaR`.

Upper partial-moment VaR helper used by deterministic interval paths.

## Documentation maintenance checklist

When a public API changes:

1. Update or add the function docstring in `src/nns`.
2. Update implementation status in `docs/api_status.md` if parity or support changed.
3. Run `uv run python scripts/generate_api_reference.py`.
4. Review examples in `examples/vignettes` if the signature or return shape changed.
5. Confirm `README.md` still points to the manual and the correct API status page.
