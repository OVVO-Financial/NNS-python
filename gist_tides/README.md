# Lord Kelvin, meet the data scientist 🌊

Tides were the original "hard" forecasting problem. In **1872, Lord Kelvin
(William Thomson)** built a brass [tide-predicting
machine](https://tidesandcurrents.noaa.gov/predmach.html) — a mechanical analog
computer that summed **10 astronomical harmonic constituents** (M2, S2, N2, K1,
O1, K2, L2, P1, M4, MS4) to trace a tide curve. Later machines summed **37**.
Every one of those constituents had to be *identified from the physics* of the
Earth–Moon–Sun system and its tuned by hand for each port.

This gist reproduces that forecast **from the raw water-level series alone** —
no astronomy, no named constituents, no hand-tuned periods. The
[`ovvo-nns`](https://pypi.org/project/ovvo-nns/) implementation of NNS.ARMA
discovers the seasonal structure empirically and forecasts a held-out two-week
window.

## Result

![NNS tide forecast](tides_forecast.png)

```
nns version: 1.0.6
selected periods: [46204 13166 45957 42597 45955]
method: lin
objective: 0.010669447504996599
R-squared: 0.9645500950621061
seasonality elapsed: 0.00 seconds
forecast elapsed:    137.14 seconds
total elapsed:       138.25 seconds
```

**R² ≈ 0.965** on a genuine out-of-sample fortnight — and notice the selected
periods are *not* the textbook ~12.4 h semidiurnal harmonic. NNS finds whatever
minimizes the validation objective; it rediscovers the tidal rhythm without
being told any of the physics Kelvin spent a career formalizing.

## Run it

```bash
pip install ovvo-nns pandas matplotlib
python tides_nns.py
```

Data: NOAA tide-gauge series via the
[OVVO-Financial/NNS](https://github.com/OVVO-Financial/NNS) repo. Runtime is
dominated by `NNS.ARMA.optim` (~2 min); seasonality detection is near-instant.

## A note on the one non-obvious line

```python
nns_periods = all_periods[all_periods < training_set // 4][:100]
```

`nns_seas` returns *every* candidate period, sorted strongest-first. For a long
series the strongest periods are very large, and `NNS.ARMA.optim` only accepts
periods short enough to fit ~3–4 full cycles in the training window
(`period < training_set / denominator`). So filter **before** you slice —
otherwise the top-100 can be entirely oversized and the optimizer rejects them.

---

Inspired by John Mount's
[*Lord Kelvin, Data Scientist*](https://win-vector.com/2019/08/06/lord-kelvin-data-scientist/)
(Win-Vector).
