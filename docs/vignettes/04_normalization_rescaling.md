# 04 — Normalization and rescaling

NNS provides two complementary transforms: `nns_norm` aligns variables onto a
common scale (linearly or nonlinearly), and `nns_rescale` maps a vector onto an
explicit interval or a risk-neutral target.

```python
import numpy as np
from nns import nns_norm, nns_rescale
```

## `nns_norm` — linear and nonlinear

Given columns with wildly different means and spreads, linear normalization
brings them onto a shared mean scale:

```python
rng = np.random.default_rng(123)
X = np.column_stack((
    rng.normal(0, 1, 100),
    rng.normal(0, 5, 100),
    rng.normal(10, 1, 100),
    rng.normal(10, 10, 100),
))

linear = nns_norm(X, linear=True)     # columns share a common mean
nonlinear = nns_norm(X, linear=False) # partial-moment normalization
```

After linear normalization every column has the same mean — the precondition
for comparing series on one axis.

## `nns_rescale` — min-max

Map a vector onto an explicit `[a, b]` interval:

```python
raw = np.array([-2.5, 0.2, 1.1, 3.7, 5.0])
scaled = nns_rescale(raw, a=5.0, b=10.0, method="minmax")
# scaled.min() == 5.0, scaled.max() == 10.0
```

## `nns_rescale` — risk-neutral

The `"riskneutral"` method rescales a price path so its mean matches a
risk-neutral target. With `type="Terminal"` the rescaled mean equals the
forward `S0 * exp(r * T)`; with `type="Discounted"` it equals `S0`:

```python
s0, r, t = 100.0, 0.03, 1.0
prices = s0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 250)))

terminal = nns_rescale(prices, a=s0, b=r, method="riskneutral",
                       time_to_maturity=t, type="Terminal")
assert np.isclose(terminal.mean(), s0 * np.exp(r * t))

discounted = nns_rescale(prices, a=s0, b=r, method="riskneutral",
                         time_to_maturity=t, type="Discounted")
assert np.isclose(discounted.mean(), s0)
```

> Note: the R vignette also illustrates these transforms with overlaid
> histograms. Plotting is optional and omitted from the docs tests; the
> numeric invariants above are what matter for parity.

```bash
python examples/vignettes/normalization_rescaling.py
```
