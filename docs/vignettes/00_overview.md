# 00 — Overview

NNS (Nonlinear Nonparametric Statistics) builds its entire toolkit on **partial
moments** — the pieces of variance that lie above and below a target. Because
partial moments make no assumption of symmetry, linearity, or a parametric
distribution, the same primitives reconstruct classical statistics (variance,
covariance, the CDF) *and* extend naturally to nonlinear dependence,
regression, forecasting, and stochastic dominance.

This curriculum follows the R NNS vignettes:

1. Partial moments — the foundational LPM/UPM primitives.
2. Descriptive and distributional tools.
3. Dependence and nonlinear association.
4. Normalization and rescaling.
5. Hypothesis testing: ANOVA and stochastic superiority.
6. Regression, boosting, stacking, and causality.
7. Time series forecasting.
8. Simulation, bootstrap, and risk-neutral sampling.
9. Portfolios and stochastic dominance.

## Quick import

```python
import numpy as np
from nns import (
    lpm, upm, lpm_ratio,
    nns_moments, nns_dep, nns_copula, pm_matrix,
)
```

## A one-screen tour

```python
rng = np.random.default_rng(42)

# Variance is the sum of second-degree partial moments about the mean.
y = rng.normal(size=3000)
mu = float(np.mean(y))
n = y.size
pm_variance = (lpm(2, mu, y) + upm(2, mu, y)) * (n / (n - 1))
assert np.isclose(pm_variance, np.var(y, ddof=1))

# The empirical CDF is LPM.ratio with degree 0.
assert np.isclose(lpm_ratio(0, 0.0, y), np.mean(y <= 0.0))

# Pearson correlation misses y = x**2; partial-moment dependence does not.
x = rng.uniform(-1, 1, size=2000)
yq = x**2 + rng.normal(scale=0.05, size=2000)
print("Pearson:", np.corrcoef(x, yq)[0, 1])      # ~0
print("Dependence:", nns_dep(x, yq)["Dependence"])  # clearly positive
```

Run the full tour:

```bash
python examples/vignettes/overview.py
```

The remaining vignettes unpack each of these ideas in turn.
