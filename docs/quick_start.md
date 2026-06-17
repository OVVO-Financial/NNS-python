# Quick start

These snippets mirror the runnable
[example vignettes](https://github.com/OVVO-Financial/NNS-python/tree/main/examples/vignettes),
which are exercised in CI so they stay in sync with the package.

## Partial moments

```python
import numpy as np
from nns import lpm, upm

x = np.array([-2.0, -1.0, 0.5, 3.0], dtype=np.float64)

lower = lpm(degree=2, target=0.0, x=x)
upper = upm(degree=2, target=0.0, x=x)

print("lower partial moment:", lower)
print("upper partial moment:", upper)
```

## Nonlinear dependence

```python
import numpy as np
from nns import nns_cor, nns_dep

grid = np.linspace(-2.0, 2.0, 80, dtype=np.float64)
y = grid**2

print("NNS dependence:", nns_dep(grid, y))
print("NNS correlation:", nns_cor(grid, y))
```

## Nonlinear regression

Fit a nonlinear regression and estimate new points:

```python
import numpy as np
from nns import nns_reg

x = np.linspace(-3.0, 3.0, 80, dtype=np.float64)
y = np.sin(x) + 0.2 * x
points = np.array([-1.5, 0.0, 1.5], dtype=np.float64)

fit = nns_reg(x, y, point_est=points, confidence_interval=None)

print("R2:", fit["R2"])
print(np.column_stack((points, fit["Point.est"])))
```

## Forecasting

Forecast a univariate series:

```python
import numpy as np
from nns import nns_arma, nns_seas

t = np.arange(1, 60, dtype=np.float64)
series = 10.0 + np.sin(t / 3.0) + 0.05 * t

seasonality = nns_seas(series, modulo=[3, 4, 6], mod_only=True)
forecast = nns_arma(series, h=3, seasonal_factor=4, method="lin")

print("best seasonal period:", seasonality["best.period"])
print("forecast:", forecast)
```

## Next steps

- Browse the full [API reference](api_reference.md).
- Check the [API status](api_status.md) for partial, guarded, and known-gap paths.
- Read the [behavior conventions](conventions.md) for intentional divergences from R.
