# PyNNS

Python port of the R NNS 13.0 package.

- PyPI package: `nns-pm`
- Import name: `pynns`
- Runtime dependencies: NumPy, SciPy
- R required for normal use: no
- Status: alpha, parity-focused
- License: GPL-3.0-only

## Install

```bash
pip install nns-pm
```

## Quick Use

```python
import numpy as np
from pynns import lpm, nns_dep, nns_reg

x = np.array([-2.0, -1.0, 0.5, 3.0])
downside = lpm(2, 0.0, x)

grid = np.linspace(-2.0, 2.0, 50)
dep = nns_dep(grid, grid**2)

fit = nns_reg(grid, np.sin(grid), point_est=np.array([-1.0, 0.0, 1.0]))
```

## Documentation

- [API status and known gaps](docs/api_status.md)
- [Behavior conventions and intentional divergences](docs/conventions.md)
- [Benchmarks](docs/benchmarks.md)
- [Examples](docs/examples/README.md)
- [Nowcast design](docs/specs_nowcast.md)

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy
```

R and the R `NNS` package are only needed to regenerate parity fixtures.

## Attribution

NNS was created by Fred Viole as the companion R package to Viole, F. and
Nawrocki, D. (2013), *Nonlinear Nonparametric Statistics: Using Partial Moments*.

Upstream: [OVVO-Financial/NNS](https://github.com/OVVO-Financial/NNS)
