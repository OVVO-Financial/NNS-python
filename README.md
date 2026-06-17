<img src="https://raw.githubusercontent.com/OVVO-Financial/NNS/NNS-Beta-Version/vignettes/images/NNS_hex_sticker.png" width="150" style="border: none; outline: none; margin: 0; padding: 0; display: block;"/>

# NNS Python

[![PyPI package](https://img.shields.io/badge/package-ovvo--nns-blue)](https://pypi.org/project/ovvo-nns/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/docs-ovvo--financial.github.io-blue)](https://ovvo-financial.github.io/NNS-python/)
[![License](https://img.shields.io/badge/license-GPL--3.0--only-blue)](LICENSE)

`ovvo-nns` brings Nonlinear Nonparametric Statistics to Python as the `nns` import package. It is a parity-focused port of the R `NNS` 13.0+ package, designed for real-world data that violate symmetry, linearity, or distributional assumptions.

NNS is built around partial moments, the lower and upper components of variance, and uses them across nonlinear dependence, correlation, causation, regression, classification, forecasting, stochastic dominance, stochastic superiority, Monte Carlo simulation, and numerical differentiation workflows.


> NNS was created by Fred Viole as the companion R package to Viole, F. and Nawrocki, D. (2013), *Nonlinear Nonparametric Statistics: Using Partial Moments*. **Book (2nd Edition):** https://ovvo-financial.github.io/NNS/book/
>   
>  **Implementation:** For a direct quantitative finance implementation of NNS, see [OVVO Labs](https://www.ovvolabs.com)


## Package at a glance

| Item | Value |
|---|---|
| Distribution package | `ovvo-nns` |
| Import package | `nns` |
| Current version | `1.0.6` |
| Python | `>=3.11` |
| Required runtime dependencies | NumPy, SciPy |
| R required at runtime | No |
| Native acceleration | Private, optional `nns._nnscore` kernels where available |
| Public API status | Stable, parity-focused |
| License | GPL-3.0-only |

The public package is Python-native and does not call R at runtime. Some core kernels can use the private `_nnscore` extension when it is present, while public functions keep Python implementations and explicit fallback behavior.

## Install

```bash
pip install ovvo-nns
```

This includes the matplotlib plotting API (`nns.plotting`); matplotlib is a
regular dependency and is imported lazily, so `import nns` stays light. See
[`docs/plot_parity_policy.md`](docs/plot_parity_policy.md).

Use the package as `nns`:

```python
import nns

print(nns.__version__)
```

Source builds use `scikit-build-core` and `nanobind` for the optional native extension. Published wheels should be preferred when available.

## Quick start

```python
import numpy as np
from nns import lpm, nns_dep, nns_reg, upm

x = np.array([-2.0, -1.0, 0.5, 3.0], dtype=np.float64)

lower = lpm(degree=2, target=0.0, x=x)
upper = upm(degree=2, target=0.0, x=x)

print("lower partial moment:", lower)
print("upper partial moment:", upper)
```

Measure nonlinear dependence:

```python
import numpy as np
from nns import nns_cor, nns_dep

grid = np.linspace(-2.0, 2.0, 80, dtype=np.float64)
y = grid**2

print("NNS dependence:", nns_dep(grid, y))
print("NNS correlation:", nns_cor(grid, y))
```

Fit a nonlinear regression and estimate new points:

```python
import numpy as np
from nns import nns_reg

x = np.linspace(-3.0, 3.0, 80, dtype=np.float64)
y = np.sin(x) + 0.2 * x
points = np.array([-1.5, 0.0, 1.5], dtype=np.float64)

fit = nns_reg(x, y, point_est=points, confidence_interval=None)

print("R2:", fit["R2"])
print(np.column_stack((points, fit["Point.est"])))
```

Forecast a univariate series:

```python
import numpy as np
from nns import nns_arma, nns_seas

t = np.arange(1, 60, dtype=np.float64)
series = 10.0 + np.sin(t / 3.0) + 0.05 * t

seasonality = nns_seas(series, modulo=[3, 4, 6], mod_only=True)
forecast = nns_arma(series, h=3, seasonal_factor=4, method="lin")

print("best seasonal period:", seasonality["best.period"])
print("forecast:", forecast)
```

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

See [API status](docs/api_status.md) for implemented, partial, guarded, and known-gap paths.

## Design boundaries

NNS Python prioritizes stable public behavior from installed R NNS 13.0+, not private helper parity. The package returns NumPy arrays and plain dictionaries rather than R `data.table` objects, uses explicit Python errors for several unsafe R coercions, and generally ignores plotting side effects.

Important boundaries:

- R is used only for parity tests and local cache regeneration, not normal runtime use.
- Stochastic exact stream parity is not expected because Python paths use NumPy random generation.
- Factor and class ordering should be passed explicitly when ordering matters.
- Direct raw-factor `nns_m_reg(..., factor_2_dummy=True)` is intentionally guarded. Use `prepare_factor_predictors(...)` before `nns_m_reg(...)`.
- Compute functions still return values, not figures; passing `plot=True` (where R has it) additionally renders a Matplotlib figure as a side effect via the `nns.plotting` layer, which is color/element-faithful to R but not pixel-diffed. The plot functions can also be called directly on a computed result.

See [behavior conventions](docs/conventions.md) for detailed compatibility notes.

## Examples

Runnable, self-checking example scripts live in
[`examples/vignettes`](examples/vignettes), mirroring the R NNS vignettes. They
are exercised in CI by `tests/docs/test_vignette_examples.py`, so they stay in
sync with the package.

| Topic | Script |
|---|---|
| Overview | [`overview.py`](examples/vignettes/overview.py) |
| Partial moments | [`partial_moments.py`](examples/vignettes/partial_moments.py) |
| Descriptive and distributional tools | [`descriptive_distributional_tools.py`](examples/vignettes/descriptive_distributional_tools.py) |
| Dependence and nonlinear association | [`dependence_nonlinear_association.py`](examples/vignettes/dependence_nonlinear_association.py) |
| Normalization and rescaling | [`normalization_rescaling.py`](examples/vignettes/normalization_rescaling.py) |
| Hypothesis, ANOVA and stochastic superiority | [`hypothesis_anova_stochastic_superiority.py`](examples/vignettes/hypothesis_anova_stochastic_superiority.py) |
| Regression, boosting, stacking and causality | [`regression_boosting_stacking_causality.py`](examples/vignettes/regression_boosting_stacking_causality.py) |
| Time series forecasting | [`time_series_forecasting.py`](examples/vignettes/time_series_forecasting.py) |
| Simulation, bootstrap and risk-neutral | [`simulation_bootstrap_riskneutral.py`](examples/vignettes/simulation_bootstrap_riskneutral.py) |
| Portfolio and stochastic dominance | [`portfolio_stochastic_dominance.py`](examples/vignettes/portfolio_stochastic_dominance.py) |

Run one example:

```bash
uv run python examples/vignettes/partial_moments.py
```

Run all of them with a PASS/FAIL summary:

```bash
uv run python examples/run_all_vignettes.py
```

## Documentation

The full documentation site is hosted at
**<https://ovvo-financial.github.io/NNS-python/>**.

- [API reference manual](docs/api_reference.md)
- [API status and known gaps](docs/api_status.md)
- [Behavior conventions and intentional divergences](docs/conventions.md)
- [Parity target, cache regeneration, and automation](docs/parity.md)
- [Benchmarks](docs/benchmarks.md)
- [Examples](examples/vignettes)

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy
```

Run benchmark tests explicitly:

```bash
uv run pytest -n0 -m benchmark --benchmark-enable tests/benchmarks/
```

The default parity suite is cache-backed and does not require `Rscript`. `Rscript` and the R `NNS` package are needed only when regenerating parity caches or running live R comparison scripts.

## Benchmarks

Benchmarks compare selected Python paths with installed R NNS 13.0+ baselines. Many core operations are faster in Python, while some large stochastic-dominance workloads remain faster in R because the R package uses compiled kernels for those paths. See [benchmarks](docs/benchmarks.md) for current measurements and commands.

## Authors and contributors

- **Fred Viole** — author and maintainer
- **Roberto Spadim** — contributor
- **Rasheed Khoshnaw** — contributor

## Attribution

Upstream R package and reference implementation: [OVVO-Financial/NNS](https://github.com/OVVO-Financial/NNS)
