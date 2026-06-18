# N-HiTS is a neural breakthrough — NNS needs no network 🚦

[N-HiTS](https://arxiv.org/abs/2201.12886) (Neural Hierarchical Interpolation for
Time Series, AAAI 2023) is one of the strongest deep-learning forecasters around.
By combining **multi-rate input sampling** with **hierarchical interpolation** —
built on the N-BEATS lineage — it reports roughly **20% better long-horizon
accuracy than Transformer models at ~50× less compute**. It is, however, still a
neural network: it needs a training loop, hyperparameters, and the hardware to go
with them.

This example forecasts the last **120 steps** of an hourly traffic-volume series
with `NNS.ARMA` — **no neural network, no training loop, no hyperparameter
search** — in a handful of lines. The seasonal periods are discovered directly
from the data by `nns_seas`.

## Result

![NNS traffic forecast](nhits_traffic_forecast.png)

```
MAE on traffic data: 236.16791666666677
```

On a series that swings from ~200 to ~6,200 vehicles, NNS tracks both the sharp
rush-hour peaks and the overnight troughs with a mean absolute error around
**236** — a competitive long-horizon forecast produced in seconds, without any
of the training machinery a neural model requires.

## Run it

```bash
pip install ovvo-nns pandas numpy scikit-learn matplotlib
python nhits_traffic.py
```

Writes `nhits_traffic_forecast.png` next to the script. Data is the hourly
traffic-volume series from the
[OVVO-Financial/NNS](https://github.com/OVVO-Financial/NNS) repo.

## How it works

```python
seas_result = NNS.nns_seas(train_set, plot=False)
periods = seas_result.get("periods", [])

nns_estimates = NNS.nns_arma_optim(
    variable=train_set,
    h=120,
    seasonal_factor=periods,
    obj_fn=lambda predicted, actual: np.mean(np.abs(predicted - actual)),
    objective="min",
    negative_values=False,
)
```

`nns_seas` finds the candidate seasonal periods; `nns_arma_optim` cross-validates
combinations of them and forecasts `h=120` steps ahead. The `obj_fn` is set to
MAE so the optimizer targets the same metric the benchmark reports. (`nns_arma_optim`
silently discards any periods too long to estimate, so the full `nns_seas` list can
be passed straight in.)

---

N-HiTS background from Marco Peixeiro's
[*All About N-HiTS*](https://www.datasciencewithmarco.com/blog/all-about-n-hits-the-latest-breakthrough-in-time-series-forecasting)
and the [original paper](https://arxiv.org/abs/2201.12886) (Challu, Olivares, et al.).
