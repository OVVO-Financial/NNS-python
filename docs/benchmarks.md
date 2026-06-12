# Benchmarks

Run with:

```bash
mkdir -p docs/benchmark_reports
uv run pytest -n0 -m benchmark --benchmark-enable \
  --benchmark-json=docs/benchmark_reports/benchmark_latest.json tests/benchmarks/
Rscript scripts/benchmark_realistic_sd_r.R \
  --repeats=3 --max-repeats=1 \
  --output=docs/benchmark_reports/realistic_sd_r_latest.csv
uv run python scripts/update_benchmarks_doc.py docs/benchmark_reports/benchmark_latest.json \
  --realistic-sd-r-csv=docs/benchmark_reports/realistic_sd_r_latest.csv
```

## Results

R baselines use installed R NNS 12.1.

`Python speed vs R` is computed as `R baseline / Python mean`. Values above `1.00x` mean Python is faster; values below `1.00x` mean Python is slower.

| Benchmark | Python mean | R baseline | Python speed vs R |
| --- | ---: | ---: | ---: |
| `lpm small` | 0.011 ms | 0.090 ms | 8.00x |
| `pm matrix scale, 10` | 0.085 ms | 3.600 ms | 42.11x |
| `pm matrix scale, 50` | 0.487 ms | 7.200 ms | 14.79x |
| `pm matrix scale, 100` | 13.430 ms | 21.200 ms | 1.58x |
| `sd efficient set degree 2 scale` | 24.352 ms | 4.400 ms | 0.18x |
| `nns sd cluster 252x50 degree2` | 46.482 ms | 16.600 ms | 0.36x |
| `nns sd cluster 252x50 degree2 dendrogram` | 44.213 ms | 18.667 ms | 0.42x |
| `nns cdf 1000 degree0` | 0.036 ms | 1.100 ms | 30.43x |
| `nns cdf 1000 degree2` | 0.113 ms | 1.250 ms | 11.09x |
| `nns cdf 500x3 degree1` | 47.558 ms | 58.000 ms | 1.22x |
| `nns dep 1000` | 7.409 ms | 8.700 ms | 1.17x |
| `nns dep asym 1000` | 7.342 ms | 9.100 ms | 1.24x |
| `nns copula 1000` | 0.393 ms | 1.900 ms | 4.84x |
| `nns causation 1000` | 15.763 ms | 34.200 ms | 2.17x |
| `nns norm 1000x3` | 0.122 ms | 0.620 ms | 5.09x |
| `nns distance 1000x3` | 0.712 ms | 0.700 ms | 0.98x |
| `nns distance bulk 1000x3 100` | 5.684 ms | 5.950 ms | 1.05x |
| `nns distance class 500x3` | 0.602 ms | 0.570 ms | 0.95x |
| `nns distance bulk class 500x3 50` | 1.387 ms | 1.900 ms | 1.37x |
| `nns diff sin` | 1.258 ms | 3.050 ms | 2.42x |
| `dy dx numeric eval points` | 25.271 ms | 37.350 ms | 1.48x |
| `dy_d`, scalar wrt=1, eval_points=mean, N=2, T_obs=100 | 87.198 ms | 274.800 ms | 3.15x |
| `dy_d`, scalar wrt=1, eval_points=median, N=2, T_obs=100 | 86.289 ms | 260.000 ms | 3.01x |
| `dy_d`, scalar wrt=1, eval_points=last, N=2, T_obs=100 | 93.609 ms | 265.800 ms | 2.84x |
| `dy_d`, scalar wrt=1, eval_points=obs, N=2, T_obs=100 | 89.956 ms | 279.600 ms | 3.11x |
| `dy_d`, scalar wrt=1, eval_points=apd, N=2, T_obs=100 | 700.579 ms | 1117.600 ms | 1.60x |
| `nns anova 100x2` | 7.713 ms | 3.500 ms | 0.45x |
| `nns part 500` | 0.726 ms | 2.450 ms | 3.37x |
| `nns reg 500` | 84.662 ms | 30.400 ms | 0.36x |
| `nns reg 200 confidence interval` | 98.202 ms | 85.200 ms | 0.87x |
| `nns reg 200 smooth` | 18.054 ms | 43.200 ms | 2.39x |
| `nns reg factor predictor 200` | 27.413 ms | 415.400 ms | 15.15x |
| `nns reg factor predictor dimred 120` | 59.522 ms | 35.400 ms | 0.59x |
| `nns reg class 200` | 19.457 ms | 29.800 ms | 1.53x |
| `nns reg class 200 confidence interval` | 34.230 ms | 48.200 ms | 1.41x |
| `nns reg dimred 200x3` | 44.167 ms | 34.400 ms | 0.78x |
| `nns m reg 200x3` | 114.786 ms | 88.600 ms | 0.77x |
| `nns m reg 200x3 confidence interval` | 102.176 ms | 125.000 ms | 1.22x |
| `nns m reg class 200x3` | 54.130 ms | 114.800 ms | 2.12x |
| `nns m reg class 200x3 confidence interval` | 54.759 ms | 123.000 ms | 2.25x |
| `nns stack 100x3` | 300.295 ms | 360.333 ms | 1.20x |
| `nns stack factor predictor 60 method1` | 45.079 ms | 207.333 ms | 4.60x |
| `nns stack mixed factor predictor 60 method2` | 37.416 ms | 118.000 ms | 3.15x |
| `nns stack mixed factor predictor 100x3 method12` | 376.170 ms | 332.333 ms | 0.88x |
| `nns stack 100x3 pred int` | 180.852 ms | 304.000 ms | 1.68x |
| `nns stack 100x3 ts test` | 316.159 ms | 285.333 ms | 0.90x |
| `nns stack class 100x3` | 131.847 ms | 261.000 ms | 1.98x |
| `nns stack class 100x3 pred int` | 139.910 ms | 333.333 ms | 2.38x |
| `nns stack class balance 150x3` | 194.450 ms | 311.667 ms | 1.60x |
| `nns boost 50x3` | 197.738 ms | 3548.000 ms | 17.94x |
| `nns boost 50x3 pred int` | 144.660 ms | 3844.500 ms | 26.58x |
| `nns boost 50x3 ts test` | 152.450 ms | 3510.000 ms | 23.02x |
| `nns boost stochastic 64x11` | 269.218 ms | 3219.500 ms | 11.96x |
| `nns boost stochastic ts test 64x11` | 248.128 ms | 3956.000 ms | 15.94x |
| `nns boost factor predictor 50x2` | 157.353 ms | 3738.000 ms | 23.76x |
| `nns boost multi factor predictor 50x3` | 202.788 ms | 4429.000 ms | 21.84x |
| `nns boost class 50x3` | 176.135 ms | 4333.000 ms | 24.60x |
| `nns boost class 50x3 pred int` | 263.675 ms | 4183.000 ms | 15.86x |
| `nns boost class balance 80x3` | 401.302 ms | 4508.500 ms | 11.23x |
| `nns mode continuous 1000` | 0.467 ms | 0.090 ms | 0.19x |
| `nns seas 1000` | 0.012 ms | 1.250 ms | 104.57x |
| `nns seas 5000` | 0.026 ms | 5.900 ms | 230.05x |
| `nns arma 500 auto nonlin` | 20.021 ms | 334.333 ms | 16.70x |
| `nns arma 500 explicit12 nonlin` | 70.419 ms | 350.333 ms | 4.97x |
| `nns arma 200 explicit4 lin predint` | 169.046 ms | 213.400 ms | 1.26x |
| `nns arma 200 auto nonlin predint` | 181.461 ms | 373.800 ms | 2.06x |
| `nns arma optim 80 small` | 35.850 ms | 544.333 ms | 15.18x |
| `nns_var`, dim_red_method=cor, N=3, T_obs=80, h=3, tau=2 | 834.707 ms | 3778.667 ms | 4.53x |
| `nns_var`, dim_red_method=NNS.dep, N=3, T_obs=80, h=3, tau=2 | 1572.523 ms | 6381.667 ms | 4.06x |
| `nns_var`, dim_red_method=NNS.caus, N=3, T_obs=80, h=3, tau=2 | 3394.805 ms | 9718.667 ms | 2.86x |
| `nns_var`, dim_red_method=all, N=3, T_obs=80, h=3, tau=2 | 4087.817 ms | 9976.333 ms | 2.44x |
| `nns meboot 500 reps100` | 71.581 ms | 98.333 ms | 1.37x |
| `nns meboot 1000 reps100` | 102.197 ms | 147.667 ms | 1.44x |
| `nns mc 500 reps30 by02` | 301.693 ms | 638.000 ms | 2.11x |
| `nns mc 500 reps30 by01` | 631.741 ms | 1334.333 ms | 2.11x |
| `nns ss 1000` | 0.051 ms | 0.260 ms | 5.08x |
| `nns ss 200 ci reps100` | 161.687 ms | 173.667 ms | 1.07x |

## Realistic Finance SD North Stars

These benchmarks use the static daily-return fixture at
`tests/fixtures/finance/sp500_daily_returns_2019_2023.csv`. That finance
fixture is local-only and not tracked in git; the latest recorded run used 1257
daily return rows and 480 clean return columns after dropping
tickers with missing or non-finite returns. Constituent-universe benchmarks exclude
`SPY` and `GSPC`, leaving 478 columns. Market-relative workflows
prefer `GSPC` and fall back to `SPY`; tradable-proxy examples use `SPY`.

Benchmark-column sanity metadata:

- SPY/GSPC correlation: 0.998873
- Mean absolute daily return difference: 0.000372
- Max absolute daily return difference: 0.010417

Python timings come from `pytest-benchmark`. R timings come from
`scripts/benchmark_realistic_sd_r.R` when `--realistic-sd-r-csv` is supplied to
the updater. Rows marked `manual placeholder` use the last manually recorded R
baseline so Python/R comparisons remain visible when R has not been rerun.

Run only the realistic Python benchmarks with:

```bash
PYNNS_OFFLINE=1 uv run pytest -q -n0 -m benchmark --benchmark-enable \
  --benchmark-json=docs/benchmark_reports/realistic_sd_python_latest.json \
  tests/benchmarks/test_stochastic_dominance_realistic.py \
  tests/benchmarks/test_finance_sd_rolling.py \
  tests/benchmarks/test_finance_partial_moment_workflows.py
```

Run matching R baselines with:

```bash
Rscript scripts/benchmark_realistic_sd_r.R \
  --repeats=3 --max-repeats=1 \
  --output=docs/benchmark_reports/realistic_sd_r_latest.csv
```

`Python/R slowdown` is computed as `Python mean / R mean`. Values above `1.00x`
mean Python is slower than R.

| Realistic benchmark | Python mean | R mean | R source | Python/R slowdown |
| --- | ---: | ---: | --- | ---: |
| `nns_sd_cluster`, degree=1, N=50, T_obs=252 | 32.096 ms | 3.000 ms | measured | 10.70x |
| `sd_efficient_set`, degree=1, N=50, T_obs=252 | 27.599 ms | 2.667 ms | measured | 10.35x |
| `nns_sd_cluster`, degree=2, N=50, T_obs=252 | 27.428 ms | 6.000 ms | measured | 4.57x |
| `sd_efficient_set`, degree=2, N=50, T_obs=252 | 17.809 ms | 2.000 ms | measured | 8.90x |
| `nns_sd_cluster`, degree=1, N=100, T_obs=252 | 4.535 ms | 6.333 ms | measured | 0.72x |
| `sd_efficient_set`, degree=1, N=100, T_obs=252 | 9.349 ms | 6.000 ms | measured | 1.56x |
| `nns_sd_cluster`, degree=2, N=100, T_obs=252 | 19.124 ms | 19.000 ms | measured | 1.01x |
| `sd_efficient_set`, degree=2, N=100, T_obs=252 | 9.151 ms | 4.667 ms | measured | 1.96x |
| `nns_sd_cluster`, degree=2, N=250, T_obs=252 | 76.885 ms | 59.333 ms | measured | 1.30x |
| `sd_efficient_set`, degree=2, N=250, T_obs=252 | 31.741 ms | 15.000 ms | measured | 2.12x |
| `nns_sd_cluster`, degree=2, N=478, T_obs=252 | 326.310 ms | 194.333 ms | measured | 1.68x |
| `sd_efficient_set`, degree=2, N=478, T_obs=252 | 106.190 ms | 37.000 ms | measured | 2.87x |
| `sd_efficient_set`, degree=2, N=100, T_obs=1257 | 32.352 ms | 21.333 ms | measured | 1.52x |
| `nns_sd_cluster`, degree=2, N=250, T_obs=1257 | 296.694 ms | 209.333 ms | measured | 1.42x |
| `sd_efficient_set`, degree=2, N=250, T_obs=1257 | 192.960 ms | 70.667 ms | measured | 2.73x |
| `nns_sd_cluster`, degree=2, N=478, T_obs=1257 | 992.481 ms | 663.000 ms | measured | 1.50x |
| `sd_efficient_set`, degree=2, N=478, T_obs=1257 | 979.089 ms | 194.000 ms | measured | 5.05x |

Additional realistic finance workflow benchmarks:

| Benchmark | Python mean | R mean | R source | Python/R slowdown | Summary metadata |
| --- | ---: | ---: | --- | ---: | --- |
| Lower/upper constituent dispersion ratio, N=100, T_obs=252 | 0.138 ms | n/a | n/a | n/a | n/a |
| Magnificent Seven downside stress components with SPY | 0.406 ms | n/a | n/a | n/a | n/a |
| Magnificent Seven market-downside stress components | 10.824 ms | 47.000 ms | measured | 0.23x | downside obs: 172; stress R2: 0.7852; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Market-relative daily dispersion, full fixture | 11.549 ms | 37.667 ms | measured | 0.31x | signal len: 1257; finite: 1257; next-day corr: 0.06635; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Market-relative rolling dispersion signal, 252d | 11.956 ms | 37.667 ms | measured | 0.32x | signal len: 1006; finite: 1006; next-day corr: 0.03746; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Market-relative rolling dispersion signal, 63d | 9.073 ms | 39.333 ms | measured | 0.23x | signal len: 1195; finite: 1195; next-day corr: 0.02139; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Partial-moment covariance workflow, 1257d-degree1-mean | 30.235 ms | 1.587 s | measured | 0.02x | rows: 1257; cols: 478; matrix N: 478 |
| Partial-moment covariance workflow, 252d-degree1-mean | 17.619 ms | 296.333 ms | measured | 0.06x | rows: 252; cols: 478; matrix N: 478 |
| Partial-moment covariance workflow, 252d-degree2-zero | 25.084 ms | 302.000 ms | measured | 0.08x | rows: 252; cols: 478; matrix N: 478 |
| Rolling SD cluster, 252-day monthly, degree=2, n100 | 785.532 ms | 787.000 ms | measured | 1.00x | windows: 48; avg set: 14.29; avg clusters: 8.375 |
| Rolling SD cluster, 252-day monthly, degree=2, nmax | 10.759 s | 9.873 s | measured | 1.09x | windows: 48; avg set: 29.48; avg clusters: 13.65 |
| Rolling SD cluster, 252-day quarterly, degree=1 | 1.809 s | 1.226 s | measured | 1.48x | windows: 16; avg set: 468.5; avg clusters: 1.812 |
| Rolling SD cluster, 756-day quarterly, degree=2 | 4.327 s | 4.182 s | measured | 1.03x | windows: 9; avg set: 33.11; avg clusters: 11.89 |
| Rolling SD efficient set, 252-day monthly, degree=2, n100 | 296.756 ms | 259.667 ms | measured | 1.14x | windows: 48; avg set: 14.29; avg turnover: 0.4598 |
| Rolling SD efficient set, 252-day monthly, degree=2, nmax | 3.701 s | 2.400 s | measured | 1.54x | windows: 48; avg set: 29.48; avg turnover: 0.5228 |
| Rolling SD efficient set, 252-day quarterly, degree 1 vs 2 | 3.100 s | 1.931 s | measured | 1.61x | windows: 16; avg d1 set: 468.5; avg d2 set: 29.56 |
| Rolling SD efficient set, 252-day quarterly, degree=1 | 1.722 s | 1.259 s | measured | 1.37x | windows: 16; avg set: 468.5; avg turnover: 0.03102 |

Interpretation:

- Large degree-1 discrete SD uses an exact order-statistic dominance
  matrix: one empirical sample FSD-dominates another iff every sorted
  order statistic is at least as large, with at least one strict
  improvement.
- Guarded prefix-pair evaluation skips curve work for min/mean/identical
  impossible pairs, and the standalone efficient-set path only checks
  already-kept candidates for degree 2/3 and degree-1 continuous cases.
- The implementation deliberately follows R's C++ SD algorithmic structure:
  sorted columns, prefix sums, pair-threshold dominance checks, exact guards, and
  no tolerance-based shortcuts.
- Full-fixture PyNNS runs are feasible for research iteration, but R's C++ SD
  core remains materially faster on the largest cluster cases.
