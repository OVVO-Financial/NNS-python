# 09 — Portfolios and stochastic dominance

Stochastic dominance ranks distributions without assuming a utility function.
NNS provides fast univariate dominance tests and portfolio-level routines that
build efficient sets and dominance-based clusters from a return panel.

```python
import numpy as np
from nns import fsd_uni, ssd_uni, tsd_uni, sd_efficient_set, nns_sd_cluster

rng = np.random.default_rng(123)
```

## Pairwise dominance tests

`fsd_uni`, `ssd_uni`, and `tsd_uni` return `1` when the first argument
dominates the second at first, second, or third order, and `0` otherwise. A
constant upward shift is a textbook first-order dominance, and first-order
dominance implies the higher orders:

```python
x = rng.normal(size=1000)
y = x + 1.0                 # y dominates x by a constant shift

fsd_uni(y, x)  # 1
fsd_uni(x, y)  # 0
ssd_uni(y, x)  # 1  (FSD implies SSD)
tsd_uni(y, x)  # 1  (FSD implies TSD)
```

## A small portfolio return example

```python
ra = rng.normal(0.005, 0.03, 240)
rb = rng.normal(0.003, 0.02, 240)
rc = rng.normal(0.006, 0.04, 240)
returns = np.column_stack((ra, rb, rc))
```

### Efficient set

`sd_efficient_set` returns the indices of assets not dominated at the chosen
degree — the dominance-efficient frontier:

```python
sd_efficient_set(returns, degree=1)   # e.g. [2, 0, 1]
```

### Dominance clustering

`nns_sd_cluster` groups assets by their dominance relationships:

```python
clusters = nns_sd_cluster(returns, degree=1, names=["A", "B", "C"])
clusters["Clusters"]
```

These portfolio tools let you screen and group assets purely on distributional
dominance, with no mean-variance or utility assumptions.

```bash
python examples/vignettes/portfolio_stochastic_dominance.py
```
