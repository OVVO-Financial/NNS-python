# 07 — Time series forecasting

NNS forecasting detects seasonality nonparametrically, then projects component
series forward with linear or nonlinear partial-moment regression. The same
machinery extends to multivariate forecasting through `nns_var`.

```python
import numpy as np
from nns import nns_seas, nns_arma, nns_arma_optim, nns_var
```

## `nns_arma` — deterministic forecasts

Using the AirPassengers-style 24-point series, the nonseasonal nonlinear and
the seasonal linear forecasts are fully deterministic and match live R NNS 13.0
(PR #3):

```python
series = np.array(
    [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
     115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140],
    dtype=float,
)

nns_arma(series, h=4, seasonal_factor=False, method="nonlin")
# [128.5, 113.5, 155.5, 213.6667]

nns_arma(series, h=6, seasonal_factor=12, method="lin")
# [118., 134., 150., 141., 129., 163.]
```

## `nns_seas` — seasonality detection

`nns_seas` returns the full period table (`all.periods`), the single
`best.period`, and the selected `periods`:

```python
z = np.sin(np.arange(1, 121) / 8.0)
seas = nns_seas(z, plot=False)
seas["periods"]
```

## `nns_arma_optim` — validated forecasting

`nns_arma_optim` searches candidate seasonal factors and methods, returning the
selected configuration and prediction bands. This deterministic run matches the
structure verified in PR #3:

```python
optim = nns_arma_optim(z, h=12, seasonal_factor=[10, 20, 30],
                       plot=False, print_trace=False)
optim["periods"]          # selected seasonal factor(s)
optim["obj.fn"]           # objective value at the optimum
optim["method"]           # 'lin' | 'nonlin' | 'both'
optim["results"]          # length-h forecast
optim["lower.pred.int"]   # lower band (<= results)
optim["upper.pred.int"]   # upper band (>= results)
```

The `results` vector has length `h`, and `lower.pred.int <= upper.pred.int`
element-wise.

## `nns_var` — multivariate forecasting

`nns_var` forecasts a panel of series jointly, returning per-series univariate
and ensemble forecasts shaped `(h, n_series)`:

```python
t = np.arange(1, 61)
panel = np.column_stack((np.sin(t / 6.0), np.cos(t / 5.0), np.sin(t / 4.0) + 0.5))
var = nns_var(panel, h=4, tau=3, ncores=1, status=False)
var["ensemble"].shape   # (4, 3)
```

```bash
python examples/vignettes/time_series_forecasting.py
```
