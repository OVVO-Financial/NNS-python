"""
run_block_nns_intervals.py
================================================================================
Apples-to-apples prediction-interval comparison, NNS point forecast held FIXED.

Question answered
-----------------
At each origin t, given only data through t, NNS.ARMA.optim emits a single
h-step block forecast (h = implied_h = current_train*(1-0.9)/0.9, NO online
updating inside the block). We then build several prediction intervals on the
SAME NNS point forecast and the SAME NNS residuals, and ask:

    Does NNS's native prediction interval beat conformalizing NNS's own
    residuals — and does letting the band widen with lead-time help?

Because the point model is identical across every interval method, this isolates
interval construction with zero protocol mismatch. (Contrast issue #57, where the
NNS optimizer also saw the scored block — fixed here: the model is selected on a
strictly historical validation tail, and the scored block is never shown to it.)

Interval methods compared (all on the identical NNS block point forecast):
  * NNS native PI            — results ± pi_width  (flat width, NNS's own rule)
  * NNS + split-CP (flat)    — empirical 0.9 quantile of NNS validation residuals
  * NNS + split-CP (per-lead)— quantile per lead-time k, pooled across past blocks
                               (widens with horizon; data-starved -> fallback early)
  * NNS + Gaussian (flat)    — z * std(NNS validation residuals)

Run:
    pip install ovvo-nns numpy pandas scipy matplotlib
    python run_block_nns_intervals.py
"""
from __future__ import annotations

import os
import math
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

try:
    import nns as NNS
except ImportError:
    raise SystemExit("ovvo-nns is required. Run: pip install ovvo-nns")

# ── Constants ────────────────────────────────────────────────────────────────
ALPHA       = 0.10
TARGET_COV  = 1.0 - ALPHA
N_LAGS      = 12
CAL_END     = 1000          # first NNS origin = N_LAGS + CAL_END
WINDOW      = 100
N_SEEDS     = 10
TRAINING_FRAC = 0.90        # implied_h = current_train*(1-frac)/frac
VAL_FRAC    = 0.10          # historical validation-tail fraction for model selection
CP_MIN_POOL = 8             # min per-lead samples before trusting per-lead CP
LOCAL_WIN   = 2             # +/- lead-time window to thicken per-lead pools

os.makedirs("results_block", exist_ok=True)

_MSE = lambda predicted, actual: np.mean((predicted - actual) ** 2)

# ── DGP ──────────────────────────────────────────────────────────────────────
def make_timeseries(T: int = 3500, seed: int = 0, heavy_tail: bool = False) -> dict:
    rng = np.random.default_rng(seed + 1)
    tt    = np.arange(1, T + 1, dtype=float)
    level = 0.002 * tt + 1.50 * np.sin(2*np.pi*tt/50) + 0.75 * np.sin(2*np.pi*tt/200)
    sigma = np.ones(T)
    sigma[(tt > 900)  & (tt <= 1400)] = 2.5
    sigma[(tt > 1900) & (tt <= 2450)] = 0.55
    sigma[(tt > 2800)]                = 1.8
    eps = rng.standard_t(5, size=T) / math.sqrt(5/3) if heavy_tail else rng.standard_normal(T)
    y   = np.empty(T)
    y[0] = level[0] + sigma[0] * eps[0]
    for i in range(1, T):
        y[i] = level[i] + 0.55 * (y[i-1] - level[i-1]) + sigma[i] * eps[i]
    return {"y": y, "level": level, "sigma": sigma}

# ── Scoring helpers ──────────────────────────────────────────────────────────
def z_alpha(alpha: float = ALPHA) -> float:
    return float(stats.norm.ppf(1 - alpha / 2))

def coverage(lo, hi, y) -> float:
    lo, hi, y = map(np.asarray, (lo, hi, y))
    return float(np.mean((y >= lo) & (y <= hi)))

def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))

def rolling_coverage(lo, hi, y, window: int = WINDOW) -> np.ndarray:
    lo, hi, y = map(np.asarray, (lo, hi, y))
    n = len(y)
    if n < window:
        return np.array([])
    return np.array([coverage(lo[i:i+window], hi[i:i+window], y[i:i+window])
                     for i in range(n - window + 1)])

def worst_window_coverage(lo, hi, y, window: int = WINDOW) -> float:
    rc = rolling_coverage(lo, hi, y, window)
    return float(np.min(rc)) if len(rc) > 0 else float("nan")

def interval_score(lo, hi, y, alpha: float = ALPHA) -> float:
    lo, hi, y = map(np.asarray, (lo, hi, y))
    return float(np.mean((hi - lo)
                         + (2/alpha) * np.maximum(lo - y, 0)
                         + (2/alpha) * np.maximum(y - hi, 0)))

def coverage_by_stratum(lo, hi, y, sigma, k: int = 4) -> list[float]:
    lo, hi, y, sigma = map(np.asarray, (lo, hi, y, sigma))
    ranks = stats.rankdata(sigma, method="ordinal")
    grp   = np.ceil(ranks / len(ranks) * k).astype(int).clip(1, k)
    return [coverage(lo[grp == j], hi[grp == j], y[grp == j]) for j in range(1, k+1)]

def crps_gaussian(mu, sigma, y) -> float:
    sigma = np.maximum(np.asarray(sigma, float), 1e-8)
    z = (np.asarray(y) - np.asarray(mu)) / sigma
    return float(np.mean(sigma * (z * (2*stats.norm.cdf(z) - 1)
                                  + 2*stats.norm.pdf(z) - 1/math.sqrt(math.pi))))

def log_score_gaussian(mu, sigma, y) -> float:
    sigma = np.maximum(np.asarray(sigma, float), 1e-8)
    return float(np.mean(-stats.norm.logpdf(np.asarray(y), loc=np.asarray(mu), scale=sigma)))

def score_method(name, lo, hi, y, sigma, mu_=None, s_=None) -> dict:
    lo_v = np.minimum(np.asarray(lo, float), np.asarray(hi, float))
    hi_v = np.maximum(np.asarray(lo, float), np.asarray(hi, float))
    cbs  = coverage_by_stratum(lo_v, hi_v, y, sigma)
    row = {
        "method":         name,
        "marg_cov":       coverage(lo_v, hi_v, y),
        "worst_win_cov":  worst_window_coverage(lo_v, hi_v, y),
        "cov_lowvol":     cbs[0],
        "cov_hivol":      cbs[-1],
        "cond_cov_gap":   max(abs(c - TARGET_COV) for c in cbs if not math.isnan(c)),
        "width":          mean_width(lo_v, hi_v),
        "interval_score": interval_score(lo_v, hi_v, y),
        "CRPS":           float("nan"),
        "logscore":       float("nan"),
    }
    if mu_ is not None and s_ is not None:
        row["CRPS"]     = crps_gaussian(mu_, s_, y)
        row["logscore"] = log_score_gaussian(mu_, s_, y)
    return row

# ── NNS seasonal period discovery (full-window) ──────────────────────────────
def get_nns_seas_periods(series: np.ndarray, train_n: int) -> list[int]:
    max_period = int(train_n / 3) - 1
    try:
        raw = NNS.nns_seas(series, plot=False).get("periods", np.array([]))
        seen, valid = set(), []
        for p in raw:
            pi = int(p)
            if 1 < pi < max_period and pi not in seen:
                seen.add(pi); valid.append(pi)
        return valid if valid else [2]
    except Exception:
        return [2]

def _emp_q(scores, level=TARGET_COV):
    """Finite-sample conformal quantile of |scores| at coverage `level`."""
    s = np.sort(np.abs(np.asarray(scores, float)))
    n = len(s)
    if n == 0:
        return None
    k = int(np.ceil((n + 1) * level)) - 1
    return float("inf") if k >= n else float(s[k])

# ── Block walk-forward: NNS point fixed, intervals swapped ────────────────────
def run_once(seed: int = 0, heavy_tail: bool = False, verbose: bool = True) -> dict:
    d = make_timeseries(T=3500, seed=seed, heavy_tail=heavy_tail)
    y, sig = d["y"], d["sigma"]
    T_raw = len(y)
    z = z_alpha(ALPHA)

    # accumulators: per interval-method lo/hi, shared point/y/sigma per scored step
    acc_y, acc_sig, acc_pt = [], [], []
    lo_acc = defaultdict(list); hi_acc = defaultdict(list); sig_acc = defaultdict(list)
    per_lead_pool: dict[int, list] = defaultdict(list)     # realized |resid| by lead-time

    current_train = N_LAGS + CAL_END
    t0 = time.time(); n_chunks = 0

    while current_train < T_raw:
        remaining = T_raw - current_train
        implied_h = max(1, int(current_train * (1 - TRAINING_FRAC) / TRAINING_FRAC))
        h = min(remaining, implied_h)
        train_slice = y[:current_train]
        val_h = max(1, int(round(VAL_FRAC * current_train)))
        periods = get_nns_seas_periods(train_slice, current_train)

        try:
            fit = NNS.nns_arma_optim(
                variable=train_slice, h=h, training_set=current_train - val_h,
                seasonal_factor=periods, negative_values=True,
                obj_fn=_MSE, objective="min", linear_approximation=True,
                pred_int=TARGET_COV, print_trace=False, plot=False,
            )
        except Exception as exc:
            if verbose:
                print(f"  [seed {seed} chunk {n_chunks+1} failed: {exc}] – skipping")
            current_train += h
            continue

        point     = np.asarray(fit["results"], float)            # bias-corrected block point
        nat_lo    = np.asarray(fit["lower.pred.int"], float)
        nat_hi    = np.asarray(fit["upper.pred.int"], float)
        val_err   = np.asarray(fit["errors"], float)             # validation-tail residuals
        bias_shift = float(fit["bias.shift"])
        # residuals of the *reported* (bias-corrected) forecast on the validation tail
        cal_resid = val_err + bias_shift

        yb  = y[current_train:current_train + h]
        sgb = sig[current_train:current_train + h]
        n_chunks += 1

        # ── interval method 1: NNS native PI ──────────────────────────────
        lo_acc["NNS native PI"].extend(nat_lo)
        hi_acc["NNS native PI"].extend(nat_hi)
        nat_hw = np.maximum((nat_hi - nat_lo) / 2.0, 1e-8)
        sig_acc["NNS native PI"].extend(nat_hw / z)

        # ── interval method 2: split-CP (flat) on NNS validation residuals ─
        q_flat = _emp_q(cal_resid, TARGET_COV)
        q_flat = nat_hw.mean() if (q_flat is None or not np.isfinite(q_flat)) else q_flat
        lo_acc["NNS + split-CP (flat)"].extend(point - q_flat)
        hi_acc["NNS + split-CP (flat)"].extend(point + q_flat)
        sig_acc["NNS + split-CP (flat)"].extend(np.full(h, max(q_flat / z, 1e-8)))

        # ── interval method 3: split-CP (per-lead-time) ───────────────────
        pl_lo = np.empty(h); pl_hi = np.empty(h); pl_s = np.empty(h)
        for k in range(h):
            pool = []
            for kk in range(max(0, k - LOCAL_WIN), k + LOCAL_WIN + 1):
                pool.extend(per_lead_pool.get(kk, []))
            if len(pool) >= CP_MIN_POOL:
                qk = _emp_q(pool, TARGET_COV)
                if qk is None or not np.isfinite(qk):
                    qk = q_flat
            else:
                qk = q_flat                                       # fallback while pool is thin
            pl_lo[k] = point[k] - qk; pl_hi[k] = point[k] + qk; pl_s[k] = max(qk / z, 1e-8)
        lo_acc["NNS + split-CP (per-lead)"].extend(pl_lo)
        hi_acc["NNS + split-CP (per-lead)"].extend(pl_hi)
        sig_acc["NNS + split-CP (per-lead)"].extend(pl_s)

        # ── interval method 4: Gaussian (flat) ────────────────────────────
        s_flat = max(float(np.std(cal_resid)), 1e-8)
        lo_acc["NNS + Gaussian (flat)"].extend(point - z * s_flat)
        hi_acc["NNS + Gaussian (flat)"].extend(point + z * s_flat)
        sig_acc["NNS + Gaussian (flat)"].extend(np.full(h, s_flat))

        # shared truth / point
        acc_y.extend(yb); acc_sig.extend(sgb); acc_pt.extend(point)

        # update per-lead pools with realized residuals AFTER scoring (now data <= next t)
        realized = np.abs(yb - point)
        for k in range(h):
            per_lead_pool[k].append(realized[k])

        current_train += h

    y_arr = np.asarray(acc_y); sig_arr = np.asarray(acc_sig); pt_arr = np.asarray(acc_pt)
    rows = []
    for name in lo_acc:
        lo = np.asarray(lo_acc[name]); hi = np.asarray(hi_acc[name])
        s_ = np.asarray(sig_acc[name])
        rows.append(score_method(name, lo, hi, y_arr, sig_arr, mu_=pt_arr, s_=s_))

    if verbose:
        print(f"  [seed {seed}] {n_chunks} blocks, "
              f"point MAE={np.mean(np.abs(pt_arr - y_arr)):.3f}, {time.time()-t0:.0f}s")

    return {"scores": pd.DataFrame(rows), "lo": lo_acc, "hi": hi_acc,
            "y": y_arr, "sigma": sig_arr, "point": pt_arr}

# ── Aggregate over seeds ──────────────────────────────────────────────────────
def run_all() -> dict:
    all_scores = []
    for s in range(N_SEEDS):
        print(f"\n=== seed {s} ===")
        all_scores.append(run_once(seed=s, verbose=True)["scores"])

    scores_dt = pd.concat(all_scores, ignore_index=True)
    metric_cols = ["marg_cov", "worst_win_cov", "cov_lowvol", "cov_hivol",
                   "cond_cov_gap", "width", "interval_score", "CRPS", "logscore"]
    agg = (scores_dt.groupby("method", sort=False)[metric_cols]
           .mean().reset_index().sort_values("interval_score"))

    scores_dt.to_csv("results_block/scores_all.csv", index=False)
    agg.to_csv("results_block/scores.csv", index=False)

    print(f"\n=== BLOCK NNS INTERVAL COMPARISON  "
          f"(point=NNS fixed; mean over {N_SEEDS} seeds, alpha={ALPHA}, target={TARGET_COV}) ===\n")
    print(agg.round(3).to_string(index=False))
    print("\nWrote results_block/scores.csv and scores_all.csv")
    return {"scores": scores_dt, "summary": agg}


if __name__ == "__main__":
    run_all()
