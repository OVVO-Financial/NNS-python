# 08 — Simulation, bootstrap, and risk-neutral sampling

NNS resampling preserves the dependence structure of the original data. The
maximum-entropy bootstrap `nns_meboot` generates replicates that retain the
series' shape and rank ordering, and `nns_mc` draws Monte Carlo paths targeting
a chosen rank correlation with the original series.

```python
import numpy as np
from nns import nns_meboot, nns_mc

rng = np.random.default_rng(123)
x = np.cumsum(rng.normal(scale=0.7, size=80))
```

## `nns_meboot` — maximum-entropy bootstrap

```python
mb = nns_meboot(x, reps=10, rho=0.95, random_seed=1)
mb["ensemble"]      # ensemble series aligned to the original length
mb["replicates"]    # the individual bootstrap replicates
```

`rho` controls how tightly each replicate tracks the original ordering.

## `nns_mc` — dependence-preserving Monte Carlo

`nns_mc` sweeps a grid of target rank correlations and returns replicates keyed
by their `rho`, plus an averaged `ensemble`:

```python
mc = nns_mc(x, reps=1, lower_rho=-1.0, upper_rho=1.0, by=0.5, random_seed=1)
list(mc["replicates"].keys())   # ['rho = 1', 'rho = 0.5', 'rho = 0', 'rho = -0.5', 'rho = -1']
mc["ensemble"]
```

Higher target `rho` produces replicates more positively rank-correlated with
`x`; negative `rho` inverts the ordering.

## Stochastic output caveat

Both routines are **stochastic**. Seed the RNG for reproducibility within a
run, but validate outputs by **structure and rank** (lengths, key sets,
correlation sign), never by exact resampled values. The example script asserts
only on structure for exactly this reason.

```bash
python examples/vignettes/simulation_bootstrap_riskneutral.py
```
