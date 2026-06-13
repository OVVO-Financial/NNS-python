# 01 — Partial moments

Partial moments split a distribution at a target `t`. The **lower partial
moment** `lpm(degree, t, x)` accumulates deviations below `t`; the **upper
partial moment** `upm(degree, t, x)` accumulates deviations above it. Their
ratios give probabilities, and their inverses give quantiles. Everything else
in NNS is built from these.

```python
import numpy as np
from nns import lpm, upm, lpm_ratio, upm_ratio, lpm_var, upm_var

rng = np.random.default_rng(123)
x = rng.normal(size=100)
mu = float(np.mean(x))
n = x.size
```

## The mean as a partial-moment balance point

The first-degree upper and lower partial moments about 0 balance at the mean:

```python
mean_via_pm = upm(1, 0.0, x) - lpm(1, 0.0, x)
assert np.isclose(mean_via_pm, np.mean(x))
```

## Variance decomposition around the mean

Second-degree partial moments about the mean sum to the **population**
variance; multiply by `n / (n - 1)` for the sample variance:

```python
population_variance = upm(2, mu, x) + lpm(2, mu, x)
sample_variance = population_variance * (n / (n - 1))
assert np.isclose(sample_variance, np.var(x, ddof=1))
```

This is the central NNS idea: variance is not a monolithic quantity but the sum
of an upside and a downside piece, each measurable on its own.

## Empirical CDF via `lpm_ratio(0, t, x)`

The degree-0 lower partial moment ratio is the proportion of mass at or below
`t` — exactly the empirical CDF. `upm_ratio` is the complementary survival
function:

```python
for t in (-1.0, 0.0, 1.0):
    assert np.isclose(lpm_ratio(0, t, x), np.mean(x <= t))
    assert np.isclose(upm_ratio(0, t, x), 1.0 - lpm_ratio(0, t, x))
```

## Value-at-risk quantiles

`lpm_var(p, 0, x)` inverts the degree-0 CDF, returning the `p`-quantile;
`upm_var(p, 0, x)` returns the right-tail `(1 - p)` quantile:

```python
p = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
left = np.array([lpm_var(q, 0.0, x) for q in p])
assert np.allclose(left, np.quantile(x, p, method="linear"))

right = np.array([upm_var(q, 0.0, x) for q in p])
assert np.allclose(right, np.quantile(x, 1.0 - p, method="linear"))
```

For integer degrees 1–4, `lpm_var`/`upm_var` solve the exact partial-moment
ratio inversion (the polynomial root-finder ported from R NNS 13.0 in PR #3),
producing continuous VaR estimates rather than raw order statistics.

```bash
python examples/vignettes/partial_moments.py
```
