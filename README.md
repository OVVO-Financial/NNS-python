<img src="https://raw.githubusercontent.com/OVVO-Financial/NNS/NNS-Beta-Version/vignettes/images/NNS_hex_sticker.png" width="150" style="border: none; outline: none; margin: 0; padding: 0; display: block;"/>

# NNS Python

[![PyPI package](https://img.shields.io/badge/package-ovvo--nns-blue)](https://pypi.org/project/ovvo-nns/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/docs-ovvo--financial.github.io-blue)](https://ovvo-financial.github.io/NNS-python/)
[![License](https://img.shields.io/badge/license-GPL--3.0--only-blue)](https://github.com/OVVO-Financial/NNS-python/blob/main/LICENSE)

`ovvo-nns` brings Nonlinear Nonparametric Statistics to Python as the `nns`
import package. It is a parity-focused port of the R `NNS` 13.0+ package,
designed for real-world data that violate symmetry, linearity, or
distributional assumptions.

The R package is the reference implementation and the source of truth for the
statistical behavior, terminology, and canonical example curriculum. Python is
native and does not call R at runtime.

> NNS was created by Fred Viole as the companion R package to Viole, F. and
> Nawrocki, D. (2013), *Nonlinear Nonparametric Statistics: Using Partial
> Moments*. **Book (2nd Edition):** https://ovvo-financial.github.io/NNS/book/
>
> **Implementation:** For a direct quantitative finance implementation of NNS,
> see [OVVO Labs](https://www.ovvolabs.com).

## Package at a glance

| Item | Value |
|---|---|
| Distribution package | `ovvo-nns` |
| Import package | `nns` |
| Current version | `1.5.0` |
| Python | `>=3.11` |
| Required runtime dependencies | NumPy, SciPy, Matplotlib |
| R required at runtime | No |
| Native acceleration | Private, optional `nns._nnscore` kernels where available |
| Public API status | Stable, parity-focused |
| License | GPL-3.0-only |

## Install

```bash
pip install ovvo-nns
```

Use the package as `nns`:

```python
import nns

print(nns.__version__)
```

Published wheels are preferred. Source builds use `scikit-build-core` and
`nanobind` for the optional native extension.

## Quick start

```python
import numpy as np
from nns import lpm, nns_dep, nns_reg, upm

x = np.array([-2.0, -1.0, 0.5, 3.0])
print("LPM2:", lpm(2, 0.0, x))
print("UPM2:", upm(2, 0.0, x))

grid = np.linspace(-2.0, 2.0, 80)
print("nonlinear dependence:", nns_dep(grid, grid**2))

fit = nns_reg(grid, np.sin(grid), point_est=np.array([-1.0, 0.0, 1.0]))
print("point estimates:", fit["Point.est"])
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

See [API status](https://ovvo-financial.github.io/NNS-python/api_status/) for
implemented, partial, guarded, and known-gap paths.

## Canonical examples

The R package's nine numbered vignettes define the canonical NNS curriculum.
Python follows the same numbering, topic names, and statistical intent:

| # | Topic | Python entry point |
|---|---|---|
| 01 | Overview | [`01_overview.py`](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/01_overview.py) |
| 02 | Partial Moments | [`02_partial_moments.py`](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/02_partial_moments.py) |
| 03 | Correlation and Dependence | [`03_correlation_and_dependence.py`](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/03_correlation_and_dependence.py) |
| 04 | Normalization and Rescaling | [`04_normalization_and_rescaling.py`](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/04_normalization_and_rescaling.py) |
| 05 | Sampling and Simulation | [`05_sampling_and_simulation.py`](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/05_sampling_and_simulation.py) |
| 06 | Comparing Distributions | [`06_comparing_distributions.py`](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/06_comparing_distributions.py) |
| 07 | Clustering and Regression | [`07_clustering_and_regression.py`](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/07_clustering_and_regression.py) |
| 08 | Classification | [`08_classification.py`](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/08_classification.py) |
| 09 | Forecasting | [`09_forecasting.py`](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/09_forecasting.py) |

The mapping is recorded in
[`examples/vignettes/manifest.yml`](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/manifest.yml) and
validated in CI. The older unnumbered scripts remain as focused examples and
backward-compatible entry points.

Run one canonical example:

```bash
uv run python examples/vignettes/02_partial_moments.py
```

Run all nine in R curriculum order:

```bash
uv run python examples/run_all_vignettes.py
```

## Design boundaries

NNS Python prioritizes stable public behavior from installed R NNS 13.0+, not
private helper parity. The package returns NumPy arrays and plain dictionaries
rather than R `data.table` objects and uses explicit Python errors for unsafe R
coercions.

- R is used for parity tests and local cache regeneration, not normal runtime.
- Exact stochastic stream parity is not expected for every randomized path.
- Factor and class ordering should be supplied explicitly when it matters.
- Classification codes follow the R contract and start at 1.
- Compute functions return values; `plot=True` adds Matplotlib rendering as a
  side effect without changing the statistical result.

See [behavior conventions](https://ovvo-financial.github.io/NNS-python/conventions/)
for detailed compatibility notes.

## Documentation

- [API reference](https://ovvo-financial.github.io/NNS-python/api_reference/)
- [API status and known gaps](https://ovvo-financial.github.io/NNS-python/api_status/)
- [Behavior conventions](https://ovvo-financial.github.io/NNS-python/conventions/)
- [Parity policy and cache regeneration](https://ovvo-financial.github.io/NNS-python/parity/)
- [Benchmarks](https://ovvo-financial.github.io/NNS-python/benchmarks/)
- [Canonical examples](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/README.md)

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

The default parity suite is cache-backed and does not require `Rscript`.
`Rscript` and the R `NNS` package are needed only when regenerating parity
caches or running live R comparison scripts.

## Authors and contributors

- **Fred Viole** — author and maintainer
- **Roberto Spadim** — contributor
- **Rasheed Khoshnaw** — contributor

## Attribution

Upstream R package and reference implementation:
[OVVO-Financial/NNS](https://github.com/OVVO-Financial/NNS)
