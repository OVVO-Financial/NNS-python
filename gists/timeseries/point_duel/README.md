# Point-model duel — NNS vs. recursive ridge, one honest protocol

A fair head-to-head on **point forecasting**, with prediction intervals as a
secondary check. Every method makes the same commitment NNS makes:

> At each origin *t*, given only data through *t*, forecast an `h`-step block
> (`h = implied_h = t·(1−0.9)/0.9`) with **no online updating**. No method may
> peek at a realized value to predict the next one.

This is the comparison the earlier conformal benchmark lacked: there the
baselines forecast one step ahead with the *true* lag and refreshed every step,
while NNS forecast a multi-step block blind. Here everyone forecasts the block
blind, so the contest measures forecasting-by-design, not access to the truth.

## Contenders

- **NNS block** — `NNS.ARMA.optim`, model (periods/method/bias) selected on a
  strictly historical validation tail (leak-free), native block forecast.
- **Ridge (recursive)** — ridge on `N_LAGS` lags, fit on all data ≤ *t*, then
  projected `h` steps **recursively** (its own predictions become the lags; no
  true intermediate values). This is the baseline stripped of the h=1 crutch.
- **Persistence** — last value carried forward `h` steps (floor).

Intervals: each point method is wrapped with the **same** per-lead-time
split-conformal band (calibrated on that method's own realized block residuals,
pooled across past origins). NNS also reports its native PI.

## Results

Mean over 10 seeds, T = 3500, ~12 blocks/seed. Lower is better throughout.

**Point forecast (the headline):**

```
           method   MAE  RMSE  median_AE
        NNS block 1.514 2.056      1.122
Ridge (recursive) 2.664 3.230      2.403
      Persistence 2.493 3.241      1.991
```

**Intervals (same protocol; per-method split-CP + NNS native PI):**

```
                method  marg_cov  worst_win_cov  width  interval_score
         NNS native PI     0.845          0.561  5.658           8.674
        NNS block + CP     0.894          0.288  8.306          11.339
Ridge (recursive) + CP     0.835          0.124  9.819          15.189
      Persistence + CP     0.860          0.000 11.701          18.674
```

## Interpretation

- **NNS wins the point duel decisively.** MAE 1.51 vs. recursive ridge 2.66
  (~43% lower) and persistence 2.49; same ordering on RMSE and median absolute
  error. The result is consistent across all 10 seeds, not seed-driven.
- **Recursive ridge is *worse than persistence* on MAE.** Once you remove the
  h=1 crutch (true lag every step) and force a genuine multi-step block, the
  linear AR model's error compounds over the horizon until naive carry-forward
  beats it. This is the concrete version of the forecaster-by-design vs.
  forecaster-by-compulsion distinction: a structural seasonal model projects a
  real trajectory; a reactive linear model that only ever learned the one-step
  map degrades to noise without the truth fed back.
- **Intervals follow point quality.** NNS's native PI is far the best Winkler
  score (8.67) at the tightest width — because better, smaller residuals make a
  tighter, better-centred band. The conformal wrappers on the weaker point
  models inherit their larger errors and score worse.

### Caveat on the interval table

The shared per-lead-time split-conformal wrapper here has a cold-start fallback
(it has no realized block residuals for the first origins), which depresses
**worst-window** coverage for the `+ CP` rows (e.g. NNS block + CP at 0.288).
Read the **point** table (MAE / RMSE / median AE) as the clean, primary result
and the interval table as directional. The carefully-calibrated interval study
lives separately; this branch's job is the point-model duel.

## Run it

```bash
pip install ovvo-nns numpy pandas scipy scikit-learn
python run_point_duel.py
```

Writes `results_duel/point.csv`, `interval.csv` (aggregated) and the per-seed
`*_all.csv`.

## Note

Marginal coverage of every flat band falls short of the 0.90 target on this
heteroskedastic DGP — that is the exchangeability-violation penalty, common to
all methods, not specific to any one. The point tables (MAE / RMSE) are the
clean comparison; the interval tables should be read as efficiency-at-similar-
(under)coverage, not as a coverage guarantee.
