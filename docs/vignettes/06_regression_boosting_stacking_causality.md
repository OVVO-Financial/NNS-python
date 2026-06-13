# 06 — Regression, boosting, stacking, and causality

NNS regression partitions the predictor space by dependence and fits locally,
so it captures nonlinear structure without a model formula. The same base
learner powers boosting and stacking, and the dependence machinery gives a
directional causality measure.

```python
import numpy as np
from nns import nns_reg, nns_boost, nns_stack, nns_causation
```

## `nns_reg` — nonlinear regression

```python
x = np.arange(-5.0, 5.05, 0.05)
y = x**3
reg = nns_reg(x, y, point_est=np.array([-2.0, 0.0, 2.0]))
reg["R2"], reg["Point.est"]
```

`nns_reg` returns the fit quality (`R2`), the regression points, fitted values
(`Fitted.xy`), and point estimates (`Point.est`).

## Deterministic numeric stack and boost

The following numeric design is deterministic and was verified against live R
NNS 13.0 (PR #3). It is a good regression-test fixture because the outputs are
exactly reproducible.

```python
xb = np.linspace(-2.0, 2.0, 30)
variable = np.column_stack((xb, np.sin(xb), np.cos(xb)))
target = xb + np.sin(xb) + 0.25 * np.cos(xb)
point = variable[:5]
```

`nns_stack` returns the base regression (`reg`), the dimension-reduction
ensemble (`dim.red`), and the stacked ensemble (`stack`):

```python
stack = nns_stack(variable, target, point, method=(1, 2), cv_size=0.25, folds=1)
stack["reg"]
# [-3.013334 -2.821165 -2.821165 -2.410226 -2.410226]
stack["dim.red"]
# [-3.013334 -2.914306 -2.781248 -2.589941 -2.429359]
stack["stack"]
# [-3.013334 -2.913733 -2.781494 -2.588834 -2.429242]
```

`nns_boost` returns `results`, `feature.weights`, and `feature.frequency`
(R NNS 13.0 no longer returns `n.best`):

```python
boost = nns_boost(variable, target, point,
                  learner_trials=10, cv_size=0.25, depth=None,
                  feature_importance=False)
boost["results"]            # [-3.013334 -2.821165 -2.821165 -2.410226 -2.410226]
boost["feature.weights"]    # [0.6666667 0.3333333]
boost["feature.frequency"]  # [2. 1.]
```

A classification example is only included when it is deterministic and stable;
the balanced-`type="CLASS"` Iris boost from the R vignette is RNG-driven and is
therefore left out of the docs tests (see PR #3's documented stochastic gap).

## `nns_causation` — directional causality

Causation is directional. `nns_causation` returns the conditional causation in
each direction plus a net-direction summary whose key (`C(x--->y)` or
`C(y--->x)`) names whichever direction dominates:

```python
caus = nns_causation(driver, response)
caus["Causation.x.given.y"], caus["Causation.y.given.x"]
net_key = next(k for k in caus if k.startswith("C(") and "--->" in k)
caus[net_key]
```

```bash
python examples/vignettes/regression_boosting_stacking_causality.py
```
