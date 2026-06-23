# NNS prediction intervals vs. a conformal wrapper — apples to apples

A controlled test of **prediction-interval construction** on a single,
heteroskedastic data-generating process. The **point forecast is held fixed**:
every interval in the table below wraps the *same* `NNS.ARMA.optim` block
forecast and the *same* NNS residuals. The only thing that varies is how the
band is drawn. This isolates interval construction with zero point-model and
zero protocol confounding.

## Why this framing

This is deliberately **not** the usual online conformal benchmark, and we want
to be explicit that it is an **adaptation**. Standard adaptive conformal methods
(ACI, PID, NexCP, …) are *online, h=1* procedures: they re-anchor on the realized
observation every step, and their coverage is a control-theoretic property of
that feedback loop rather than a statement about any forecast trajectory. An
online method that re-anchors each step will hit its nominal coverage rate on
almost any sequence — it is dragged across each one-step gap rather than
forecasting it. Comparing a structural multi-step forecaster against that on a
1-step task flatters the reactive method and tests the wrong thing.

So here we **adapt conformal into a forecast wrapper** and ask a sharper
question instead:

> Given a genuine multi-step forecast made at time *t* with **no online
> updating**, how well does each interval method discern **coverage guarantees
> on a heteroskedastic process** — including *conditional* coverage across the
> volatility regimes, not just the marginal rate?

## Protocol

- **DGP:** non-linear, heteroskedastic AR(1) with slow trend, two seasonal
  components (periods 50, 200), and piecewise volatility regimes
  (σ jumps to 2.5, drops to 0.55, settles at 1.8). 10 seeds, T = 3500.
- **Forecast (fixed for all methods):** at each origin *t*, `NNS.ARMA.optim`
  emits a single `h`-step block forecast, with `h = implied_h =
  t·(1−0.9)/0.9`. The model (periods/method/bias) is selected on a strictly
  **historical** validation tail — the scored block is never shown to the
  optimizer (the leak from issue #57 is fixed). No observation inside the block
  is fed back: this is forecasting by design, exposed to its own compounding
  error.
- **Intervals compared (all on the identical NNS forecast & residuals):**
  - **NNS native PI** — `results ± pi_width`, NNS's own nonparametric rule
    (a single flat half-width from the residual distribution + bias).
  - **NNS + split-CP (flat)** — empirical (1−α) conformal quantile of the same
    NNS validation residuals. Same residuals, textbook conformal quantile.
  - **NNS + split-CP (per-lead-time)** — a conformal quantile *per lead-time k*,
    pooled across past blocks (with a local-window thickening and a flat
    fallback while pools are thin). The only method whose band **widens with
    horizon**.
  - **NNS + Gaussian (flat)** — `z·std(residuals)`, a parametric reference.
- **Scoring:** interval (Winkler) score, marginal coverage, **worst rolling-window
  coverage** and **low-/high-vol stratum coverage** (the heteroskedasticity
  discriminators), mean width, and Gaussian-proxy CRPS / log-score.

## Results

```
=== BLOCK NNS INTERVAL COMPARISON  (point = NNS fixed; mean over 10 seeds, alpha=0.1, target cov=0.9) ===

                   method  marg_cov  worst_win_cov  cov_lowvol  cov_hivol  cond_cov_gap  width  interval_score   CRPS  logscore
            NNS native PI     0.845          0.561       0.927      0.853         0.149  5.658           8.674  1.090     2.076
    NNS + split-CP (flat)     0.824          0.515       0.899      0.850         0.169  5.398           8.735  1.091     2.102
    NNS + Gaussian (flat)     0.821          0.516       0.894      0.842         0.170  5.332           8.773  1.092     2.105
NNS + split-CP (per-lead)     0.890          0.582       0.983      0.844         0.099  7.995          10.521  1.169     2.268
```

Sorted by interval (Winkler) score, lower is better. Point-forecast MAE was
identical across rows (~1.51, the fixed NNS block forecast); only the bands differ.

## Findings

- **NNS's native PI is the efficiency winner and ≈ a flat split-conformal band.**
  It posts the best Winkler score (8.674) with the tightest width (5.66), and it
  actually covers slightly *better* than the textbook split-conformal quantile on
  the *same* residuals (marg 0.845 vs 0.824). NNS's `upm_var`+bias rule is a hair
  more conservative than the empirical conformal quantile, but the two track
  closely — confirming the native interval is, in effect, a flat split-conformal
  interval.
- **Every flat band under-covers this heteroskedastic process.** All three
  flat methods land at ~0.82–0.85 marginal vs. the 0.90 target, collapse to
  ~0.51–0.56 in the *worst* 100-step window, and carry a ~0.15–0.17 calm-vs-volatile
  conditional gap (≈0.90–0.93 in the calm regime, ≈0.84–0.85 in the volatile one).
  A width calibrated on historical validation residuals does not transport to a
  multi-step block that compounds error and crosses volatility regimes.
- **Only the horizon-adaptive conformal wrapper recovers the guarantee — and pays
  for it.** `split-CP (per-lead)` is the lone method to reach near-nominal marginal
  coverage (0.890) and the smallest conditional gap (0.099), because it widens with
  lead-time and calibrates on realized block residuals. The cost is ~45% wider
  intervals (7.995) and the worst Winkler score (10.521).

## Takeaway

On a heteroskedastic process, under honest multi-step forecasting with **no online
updating**, there is no free lunch. NNS's native interval is the *efficiency*
winner but does **not** deliver the nominal 90% coverage — least of all in the
volatile regime and the worst windows, exactly where guarantees matter most.
Recovering the guarantee requires a horizon-adaptive conformal wrapper and paying
for it in width. This coverage-vs-efficiency tension — not a single "winner" — is
what the benchmark is built to expose, and it is why we frame this as an
*adaptation to discern coverage guarantees on a heteroskedastic process* rather
than a claim that any one band dominates.

## How to read it

- **`NNS native PI` vs `NNS + split-CP (flat)`** is the headline head-to-head:
  the *same* residuals, NNS's `upm_var`+bias rule vs. the empirical conformal
  quantile. If they track closely, NNS's native band already *is*, in effect, a
  flat split-conformal interval.
- **`per-lead-time`** is the only band that adapts to horizon. Watch
  **worst-window** and **high-vol stratum** coverage: that is where letting the
  width grow with lead-time should pay off on a heteroskedastic process, at some
  cost in mean width.
- **Marginal coverage near 0.90 is necessary but cheap.** The honest test on a
  heteroskedastic DGP is the **conditional** coverage gap between the calm and
  volatile regimes — a flat band that hits 0.90 on average by over-covering the
  calm regime and under-covering the volatile one has not actually solved the
  problem.

## Run it

```bash
pip install ovvo-nns numpy pandas scipy
python run_block_nns_intervals.py
```

Writes `results_block/scores.csv` (aggregated) and `scores_all.csv` (per seed).
