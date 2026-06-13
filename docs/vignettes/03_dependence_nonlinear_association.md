# 03 — Dependence and nonlinear association

Pearson correlation measures *linear* co-movement. When a relationship is
nonlinear, correlation can collapse toward zero even though the variables are
perfectly dependent. NNS measures dependence directly from partial moments, so
it sees structure the linear coefficient misses.

```python
import numpy as np
from nns import nns_dep, nns_copula, pm_matrix
```

## Linear baseline

For `y = 2x`, both correlation and dependence are ~1:

```python
x = np.arange(0.0, 3.01, 0.01)
lin = nns_dep(x, 2.0 * x)
# lin["Correlation"] ~ 1, lin["Dependence"] ~ 1
```

## Where Pearson collapses

For `y = sin(x)` over many periods, Pearson correlation is weak while
partial-moment dependence stays high:

```python
xs = np.arange(0.0, 12.0 * np.pi, np.pi / 100.0)
ys = np.sin(xs)
sine = nns_dep(xs, ys)
# sine["Correlation"] ~ 0.20 (weak), sine["Dependence"] ~ 0.81 (strong)
assert sine["Dependence"] > 3.0 * abs(sine["Correlation"])
```

The partial moment dependence vs Pearson correlation contrast is the whole
point: correlation answers "how linear?", dependence answers "how related?".

## Asymmetric dependence

Dependence need not be symmetric — `D(y | x)` can differ from `D(x | y)`:

```python
asym_xy = nns_dep(xs, ys, asym=True)["Dependence"]
asym_yx = nns_dep(ys, xs, asym=True)["Dependence"]
```

## Multivariate dependence and copulas

`pm_matrix` exposes the co-partial-moment blocks for a frame, and `nns_copula`
summarizes the joint dependence structure (near 0.5 for independent columns):

```python
rng = np.random.default_rng(123)
frame = np.column_stack((rng.normal(size=1000),
                         rng.normal(size=1000),
                         rng.normal(size=1000)))
pm = pm_matrix(1, 1, "mean", frame, True, names=["a", "b", "c"])
nns_copula(frame, continuous=True)
```

```bash
python examples/vignettes/dependence_nonlinear_association.py
```
