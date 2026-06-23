# NNS under one honest forecasting protocol

A single block walk-forward over one heteroskedastic data-generating process,
emitting two coherent analyses from the *same* leak-free NNS forecast:

1. **Point duel** — does NNS actually forecast better than a fair baseline?
2. **Interval study** — given NNS's forecast, is its native prediction interval
   better or worse than conformalizing the same residuals?

This replaces the benchmark withdrawn under issue #57, which (a) leaked each
evaluation chunk into `NNS.ARMA.optim`'s validation tail and (b) compared methods
across mismatched protocols (online h=1 baselines vs. an NNS multi-step block).

## The protocol (one rule for everyone)

> At each origin *t*, given only data through *t*, every method forecasts an
> `h`-step block (`h = implied_h = t·(1−0.9)/0.9`) with **no online updating** —
> no method may peek at a realized value to predict the next one.

This is *forecasting by design*, exposed to its own compounding error — the
opposite of an online method that re-anchors on the truth every step. NNS's model
(periods/method/bias) is selected on a strictly **historical** validation tail, so
the scored block is never shown to the optimizer (the #57 leak is fixed).

- **DGP:** non-linear, heteroskedastic AR(1), slow trend, two seasonal components
  (periods 50, 200), piecewise volatility regimes (σ → 2.5, 0.55, 1.8). 10 seeds,
  T = 3500, ~12 blocks/seed.

## Point duel

- **NNS block** — `NNS.ARMA.optim` native block forecast (leak-free selection).
- **Ridge (recursive)** — ridge on `N_LAGS` lags, fit on all data ≤ *t*, projected
  `h` steps recursively (its own predictions become the lags; no true intermediate
  values — this removes the h=1 true-lag crutch the original baselines enjoyed).
- **Persistence** — last value carried forward (floor).

```
           method   MAE  RMSE  median_AE
        NNS block 1.514 2.056      1.122
Ridge (recursive) 2.664 3.230      2.403
      Persistence 2.493 3.241      1.991
```

## Interval study

We deliberately **adapt** conformal into a forecast wrapper here, and say so. Online
adaptive conformal (ACI/PID/NexCP) gets its coverage from a per-step feedback loop —
a control-theoretic property, not forecast skill — so pitting it against a multi-step
block forecaster on a 1-step task tests the wrong thing. Instead we hold the NNS point
forecast **fixed** and vary only the band, to discern **coverage guarantees on a
heteroskedastic process** — conditional (per-regime / worst-window) coverage, not just
the marginal rate:

- **NNS native PI** — `results ± pi_width`, NNS's own flat nonparametric rule.
- **NNS + split-CP (flat)** — empirical (1−α) conformal quantile of NNS residuals.
- **NNS + split-CP (per-lead)** — a quantile *per lead-time k* (the only band that
  widens with horizon).
- **NNS + Gaussian (flat)** — `z · std(residuals)`.
- **Ridge / Persistence + split-CP (per-lead)** — same wrapper on the weaker point
  models, to show interval quality follows point quality.

```
                                 method  marg_cov  worst_win_cov  cov_lowvol  cov_hivol  cond_cov_gap  width  interval_score
                          NNS native PI     0.845          0.561       0.927      0.853         0.149  5.658           8.674
                  NNS + split-CP (flat)     0.824          0.515       0.899      0.850         0.169  5.398           8.735
                  NNS + Gaussian (flat)     0.821          0.516       0.894      0.842         0.170  5.332           8.773
              NNS + split-CP (per-lead)     0.918          0.649       0.996      0.858         0.097  8.595          10.505
Ridge (recursive) + split-CP (per-lead)     0.865          0.553       0.955      0.826         0.149 10.169          13.853
      Persistence + split-CP (per-lead)     0.887          0.333       0.981      0.807         0.162 12.107          16.440
```

Point table sorted by RMSE, interval table by interval (Winkler) score; lower is
better throughout.

## Findings

- **NNS wins the point duel decisively** — MAE 1.51 vs. recursive ridge 2.66
  (~43% lower) and persistence 2.49, same ordering on RMSE/median, consistent
  across all 10 seeds. Recursive ridge falls *below* persistence on MAE: strip the
  h=1 true-lag crutch and a linear AR model's error compounds over the block until
  naive carry-forward beats it. Structural forecasting by design vs. reactive by
  compulsion, made concrete.
- **NNS's native PI is the efficiency winner and ≈ a flat split-conformal band.**
  Best interval score (8.674) at the tightest competitive width, and it tracks the
  empirical split-conformal quantile on the *same* residuals (8.735) — confirming
  the native band is, in effect, flat split conformal, slightly more conservative.
- **Every flat band under-covers the volatile regime** (marg 0.82–0.85, worst-window
  ~0.51–0.56, `cov_hivol` ~0.84–0.85). Calibrating a single width on historical
  residuals cannot transport to a multi-step block that compounds error and crosses
  σ-regimes — a textbook exchangeability failure under heteroskedasticity.
- **Only the horizon-adaptive per-lead wrapper recovers the guarantee** — near-nominal
  0.918 marginal, the smallest conditional gap (0.097) and best worst-window (0.649) —
  at ~52% wider intervals and the worst Winkler score among the NNS bands. No free
  lunch: coverage on a heteroskedastic process costs width.
- **Interval quality follows point quality.** The *same* per-lead wrapper on the
  weaker point models scores far worse (NNS 10.5 ≪ ridge 13.9 ≪ persistence 16.4)
  with much wider bands — a better forecast makes a tighter, better-centred interval.

## How to read it

- **Marginal coverage near 0.90 is necessary but cheap.** On a heteroskedastic DGP
  the honest test is the **conditional** gap between calm (`cov_lowvol`) and volatile
  (`cov_hivol`) regimes and the **worst rolling window** — a flat band that hits 0.90
  on average by over-covering calm and under-covering volatile has not solved the
  problem. Every flat band here under-covers the volatile regime: that is the
  exchangeability-violation penalty, common to all of them, not specific to any one.
- **Width is only a fair tie-breaker at matched coverage.** Among methods with
  different coverage, a narrower band may just be under-covering more. The interval
  (Winkler) score combines the two (`width + (2/α)·exceedance`), which is why it is
  the sort key.

## Run it

```bash
pip install ovvo-nns numpy pandas scipy scikit-learn
python run_conformal.py
```

Writes `results/point.csv`, `results/interval.csv` (aggregated) and the per-seed
`*_all.csv`.

## Not yet addressed

`NNS.ARMA.optim` still treats the tail after `training_set` as its validation target
with no guard or warning when out-of-range data is passed — the upstream gap behind
#57. Adding that guard is left to a follow-up; until then, treat this as the
corrected-content restoration, not a closed case.
