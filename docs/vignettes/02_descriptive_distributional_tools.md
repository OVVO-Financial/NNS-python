# 02 — Descriptive and distributional tools

Partial moments give a full descriptive toolkit: moment summaries, robust
modes, covariance matrices, and quantile tables — all without distributional
assumptions.

```python
import numpy as np
from nns import nns_moments, nns_mode, pm_matrix, lpm_var, lpm_ratio

rng = np.random.default_rng(123)
x = rng.normal(size=200)
y = rng.normal(size=200)
```

## Moment summaries

`nns_moments` returns mean, variance, skewness, and kurtosis. The `population`
flag toggles the `n/(n-1)` rescaling:

```python
nns_moments(x, population=True)   # {'mean', 'variance', 'skewness', 'kurtosis'}
nns_moments(x, population=False)  # sample variance is larger
```

## Modes (continuous and discrete)

`nns_mode` estimates a continuous mode by default, or returns discrete /
multiple modes:

```python
nns_mode(x)                                                   # continuous estimate
nns_mode(np.array([1, 2, 2, 3, 3, 4, 4, 5], dtype=float),
         discrete=True, multi=True)                            # several modes
```

## Covariance reconstruction from a partial moment matrix

`pm_matrix` returns the four co-partial-moment blocks (`clpm`, `cupm`, `dlpm`,
`dupm`). The covariance matrix is recovered as `clpm + cupm - dlpm - dupm`:

```python
pm = pm_matrix(1, 1, "mean", np.column_stack((x, y)), True, names=["x", "y"])
reconstructed = pm["clpm"] + pm["cupm"] - pm["dlpm"] - pm["dupm"]
assert np.allclose(reconstructed, np.cov(x, y))
```

This mirrors the R vignette's covariance-matrix reassembly and shows that the
classical covariance is just a difference of co-partial moments.

## Quantile table via `lpm_var`

A quantile table is a sweep of `lpm_var` over percentiles; `lpm_ratio` recovers
the CDF at each threshold as a round-trip check:

```python
p = np.arange(0.05, 0.96, 0.1)
thresholds = np.array([lpm_var(q, 0.0, x) for q in p])
recovered = np.array([lpm_ratio(0, t, x) for t in thresholds])  # equals p
```

```bash
python examples/vignettes/descriptive_distributional_tools.py
```
