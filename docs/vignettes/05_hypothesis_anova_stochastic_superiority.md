# 05 — Hypothesis testing: ANOVA and stochastic superiority

NNS reframes hypothesis testing around **certainty** and **stochastic
superiority** rather than p-values. `nns_anova` reports a certainty that groups
share a distribution; `nns_ss` reports the probability that one sample exceeds
another, ties included.

```python
import numpy as np
from nns import nns_anova, nns_ss

rng = np.random.default_rng(123)
```

## `nns_anova` certainty

Certainty is high when groups share a center and low when they are shifted
apart:

```python
x = rng.normal(0, 1, 1000)
y_equal = rng.normal(0, 2, 1000)     # same mean, different spread
y_shifted = rng.normal(1, 1, 1000)   # shifted mean

nns_anova(x, y_equal, means_only=True)["Certainty"]    # higher
nns_anova(x, y_shifted, means_only=True)["Certainty"]  # lower
```

Interpretation: certainty near 1 means the partial-moment evidence cannot
distinguish the groups; certainty near 0 means it clearly can.

## `nns_ss` stochastic superiority

`nns_ss(x, y)` returns `p_gt` (the probability a `y` draw exceeds an `x` draw),
`p_tie` (the tie mass), and `p_star` (the tie-adjusted superiority):

```python
ss = nns_ss(x, y_shifted)
ss["p_gt"], ss["p_tie"], ss["p_star"]
```

### Stochastic superiority with ties

On discrete data, ties carry real probability mass, so `p_tie` is nonzero:

```python
xd = rng.integers(1, 6, 100).astype(float)
yd = rng.integers(1, 6, 100).astype(float)
nns_ss(xd, yd)["p_tie"]   # > 0
```

### Confidence intervals are stochastic

Requesting `confidence_interval=True` runs a bootstrap. **Test these by range,
not by exact value** — the `lower`/`upper` bounds are sampled and will vary run
to run:

```python
ss_ci = nns_ss(x, y_shifted, confidence_interval=True, reps=199, ci=0.95, random_seed=1)
assert 0.0 <= ss_ci["lower"] <= ss_ci["upper"] <= 1.0
```

The same caution applies to the `nns_anova` robust interval and its
`Effect_Size_LB`/`Effect_Size_UB` fields.

```bash
python examples/vignettes/hypothesis_anova_stochastic_superiority.py
```
