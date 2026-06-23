"""
run_point_duel.py
================================================================================
Point-model duel under one honest protocol.

Every method makes the SAME commitment NNS makes: at each origin t, given only
data through t, forecast an h-step block (h = implied_h = t*(1-0.9)/0.9) with NO
online updating inside the block. No method gets to peek at a realized value to
predict the next one. This is the apples-to-apples that the original benchmark
lacked (where the baselines forecast h=1 with true lags and refreshed every step).

Point forecasters
-----------------
  * NNS block        — NNS.ARMA.optim, model selected on a strictly historical
                       validation tail (leak-free), native block forecast.
  * Ridge (recursive)— ridge on N_LAGS lags, fit on all data <= t, then projected
                       h steps RECURSIVELY (its own predictions become the lags;
                       no true intermediate values). This removes the h=1 crutch.
  * Persistence      — last observed value carried forward h steps (floor).

Intervals (secondary, same protocol)
------------------------------------
Each point method is wrapped with the SAME per-lead-time split-conformal band,
calibrated on that method's own realized block residuals pooled across past
origins (flat fallback while thin). NNS additionally reports its native PI.

Outputs a POINT table (MAE / RMSE, the headline) and an INTERVAL table.

Run:
    pip install ovvo-nns numpy pandas scipy scikit-learn
    python run_point_duel.py
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

try:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Constants ────────────────────────────────────────────────────────────────
ALPHA       = 0.10
TARGET_COV  = 1.0 - ALPHA
N_LAGS      = 12
CAL_END     = 1000
WINDOW      = 100
N_SEEDS     = 10
TRAINING_FRAC = 0.90
VAL_FRAC    = 0.10
CP_MIN_POOL = 8
LOCAL_WIN   = 2

os.makedirs("results_duel", exist_ok=True)
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

# ── Lag features ─────────────────────────────────────────────────────────────
def lag_features(y: np.ndarray, n_lags: int = N_LAGS):
    n  = len(y)
    yy = y[n_lags:]
    X  = np.column_stack([y[n_lags - k: n - k] for k in range(1, n_lags + 1)])
    return X, yy

# ── Point forecasters (all produce an h-step block from the end of y[:t]) ────
def forecast_ridge_recursive(y_hist: np.ndarray, h: int, n_lags: int = N_LAGS) -> np.ndarray:
    """Fit ridge on all data <= t, then project h steps recursively."""
    X, yy = lag_features(y_hist, n_lags)
    if HAS_SKLEARN:
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X, yy)
        predict = lambda lags: float(model.predict(lags.reshape(1, -1))[0])
    else:
        Xtr = np.column_stack([X, np.ones(len(X))])
        coef, *_ = np.linalg.lstsq(Xtr, yy, rcond=None)
        predict = lambda lags: float(np.append(lags, 1.0) @ coef)
    hist = list(y_hist[-n_lags:])
    out = []
    for _ in range(h):
        lags = np.array(hist[-1::-1][:n_lags])   # [y_{t}, y_{t-1}, ..., y_{t-n_lags+1}]
        yhat = predict(lags)
        out.append(yhat)
        hist.append(yhat)                         # feed prediction back (no true value)
    return np.asarray(out)

def forecast_persistence(y_hist: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, float(y_hist[-1]))

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

def forecast_nns_block(y_hist: np.ndarray, h: int):
    """Returns (point, native_lo, native_hi, cal_resid) — leak-free block forecast."""
    n = len(y_hist)
    val_h = max(1, int(round(VAL_FRAC * n)))
    periods = get_nns_seas_periods(y_hist, n)
    fit = NNS.nns_arma_optim(
        variable=y_hist, h=h, training_set=n - val_h, seasonal_factor=periods,
        negative_values=True, obj_fn=_MSE, objective="min",
        linear_approximation=True, pred_int=TARGET_COV, print_trace=False, plot=False,
    )
    point  = np.asarray(fit["results"], float)
    nat_lo = np.asarray(fit["lower.pred.int"], float)
    nat_hi = np.asarray(fit["upper.pred.int"], float)
    cal_resid = np.asarray(fit["errors"], float) + float(fit["bias.shift"])
    return point, nat_lo, nat_hi, cal_resid

# ── Scoring helpers ──────────────────────────────────────────────────────────
def z_alpha(alpha: float = ALPHA) -> float:
    return float(stats.norm.ppf(1 - alpha / 2))

def coverage(lo, hi, y) -> float:
    lo, hi, y = map(np.asarray, (lo, hi, y))
    return float(np.mean((y >= lo) & (y <= hi)))

def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))

def rolling_coverage(lo, hi, y, window: int = WINDOW) -> np.ndarray:
    lo, hi, y = map(np.asarray, (lo, hi, y)); n = len(y)
    if n < window:
        return np.array([])
    return np.array([coverage(lo[i:i+window], hi[i:i+window], y[i:i+window])
                     for i in range(n - window + 1)])

def worst_window_coverage(lo, hi, y, window: int = WINDOW) -> float:
    rc = rolling_coverage(lo, hi, y, window)
    return float(np.min(rc)) if len(rc) > 0 else float("nan")

def interval_score(lo, hi, y, alpha: float = ALPHA) -> float:
    lo, hi, y = map(np.asarray, (lo, hi, y))
    return float(np.mean((hi - lo) + (2/alpha)*np.maximum(lo - y, 0)
                         + (2/alpha)*np.maximum(y - hi, 0)))

def _emp_q(scores, level=TARGET_COV):
    s = np.sort(np.abs(np.asarray(scores, float))); n = len(s)
    if n == 0:
        return None
    k = int(np.ceil((n + 1) * level)) - 1
    return float("inf") if k >= n else float(s[k])

# ── One seed ──────────────────────────────────────────────────────────────────
POINT_METHODS = ["NNS block", "Ridge (recursive)", "Persistence"]

def run_once(seed: int = 0, heavy_tail: bool = False, verbose: bool = True) -> dict:
    d = make_timeseries(T=3500, seed=seed, heavy_tail=heavy_tail)
    y, sig = d["y"], d["sigma"]
    T_raw = len(y); z = z_alpha(ALPHA)

    pt_pred  = defaultdict(list)            # point forecasts by method
    acc_y, acc_sig = [], []
    lo_acc = defaultdict(list); hi_acc = defaultdict(list)
    pools  = {m: defaultdict(list) for m in POINT_METHODS}   # per-method per-lead resid pool
    nat_lo_all, nat_hi_all = [], []        # NNS native PI

    current_train = N_LAGS + CAL_END
    t0 = time.time(); n_chunks = 0

    while current_train < T_raw:
        remaining = T_raw - current_train
        implied_h = max(1, int(current_train * (1 - TRAINING_FRAC) / TRAINING_FRAC))
        h = min(remaining, implied_h)
        y_hist = y[:current_train]
        yb  = y[current_train:current_train + h]
        sgb = sig[current_train:current_train + h]

        try:
            nns_pt, nat_lo, nat_hi, _ = forecast_nns_block(y_hist, h)
        except Exception as exc:
            if verbose:
                print(f"  [seed {seed} chunk {n_chunks+1} NNS failed: {exc}] skip")
            current_train += h
            continue

        preds = {
            "NNS block":         nns_pt,
            "Ridge (recursive)": forecast_ridge_recursive(y_hist, h),
            "Persistence":       forecast_persistence(y_hist, h),
        }
        n_chunks += 1

        # per-lead split-CP on each method's own past realized residuals
        for m in POINT_METHODS:
            p = preds[m]
            lo = np.empty(h); hi = np.empty(h)
            for k in range(h):
                pool = []
                for kk in range(max(0, k - LOCAL_WIN), k + LOCAL_WIN + 1):
                    pool.extend(pools[m].get(kk, []))
                qk = _emp_q(pool, TARGET_COV) if len(pool) >= CP_MIN_POOL else None
                if qk is None or not np.isfinite(qk):
                    # fallback: flat quantile of whatever realized resid we have, else inf->wide
                    flat_pool = [v for vs in pools[m].values() for v in vs]
                    qf = _emp_q(flat_pool, TARGET_COV) if flat_pool else None
                    qk = qf if (qf is not None and np.isfinite(qf)) else float(np.std(p) + 1e-6)
                lo[k] = p[k] - qk; hi[k] = p[k] + qk
            lo_acc[f"{m} + CP"].extend(lo); hi_acc[f"{m} + CP"].extend(hi)

        nat_lo_all.extend(nat_lo); nat_hi_all.extend(nat_hi)
        for m in POINT_METHODS:
            pt_pred[m].extend(preds[m])
        acc_y.extend(yb); acc_sig.extend(sgb)

        # update pools AFTER scoring
        for m in POINT_METHODS:
            r = np.abs(yb - preds[m])
            for k in range(h):
                pools[m][k].append(r[k])

        current_train += h

    y_arr = np.asarray(acc_y); sig_arr = np.asarray(acc_sig)

    # POINT metrics
    point_rows = []
    for m in POINT_METHODS:
        p = np.asarray(pt_pred[m])
        err = p - y_arr
        point_rows.append({
            "method": m,
            "MAE":    float(np.mean(np.abs(err))),
            "RMSE":   float(np.sqrt(np.mean(err**2))),
            "median_AE": float(np.median(np.abs(err))),
        })

    # INTERVAL metrics
    int_rows = []
    def _introw(name, lo, hi):
        lo = np.asarray(lo); hi = np.asarray(hi)
        return {"method": name, "marg_cov": coverage(lo, hi, y_arr),
                "worst_win_cov": worst_window_coverage(lo, hi, y_arr),
                "width": mean_width(lo, hi), "interval_score": interval_score(lo, hi, y_arr)}
    int_rows.append(_introw("NNS native PI", nat_lo_all, nat_hi_all))
    for name in lo_acc:
        int_rows.append(_introw(name, lo_acc[name], hi_acc[name]))

    if verbose:
        mae = {m: np.mean(np.abs(np.asarray(pt_pred[m]) - y_arr)) for m in POINT_METHODS}
        print(f"  [seed {seed}] {n_chunks} blocks  MAE: "
              + "  ".join(f"{m}={mae[m]:.3f}" for m in POINT_METHODS)
              + f"  ({time.time()-t0:.0f}s)")

    return {"point": pd.DataFrame(point_rows), "interval": pd.DataFrame(int_rows)}

# ── Aggregate ────────────────────────────────────────────────────────────────
def run_all() -> dict:
    pts, ints = [], []
    for s in range(N_SEEDS):
        print(f"\n=== seed {s} ===")
        r = run_once(seed=s, verbose=True)
        pts.append(r["point"]); ints.append(r["interval"])

    point_dt = pd.concat(pts, ignore_index=True)
    int_dt   = pd.concat(ints, ignore_index=True)
    point_agg = point_dt.groupby("method", sort=False).mean().reset_index().sort_values("RMSE")
    int_agg   = int_dt.groupby("method", sort=False).mean().reset_index().sort_values("interval_score")

    point_dt.to_csv("results_duel/point_all.csv", index=False)
    int_dt.to_csv("results_duel/interval_all.csv", index=False)
    point_agg.to_csv("results_duel/point.csv", index=False)
    int_agg.to_csv("results_duel/interval.csv", index=False)

    print(f"\n=== POINT-MODEL DUEL  (block h-step, no online updating; "
          f"mean over {N_SEEDS} seeds) ===\n")
    print(point_agg.round(3).to_string(index=False))
    print(f"\n=== INTERVALS (same protocol, per-method CP + NNS native) ===\n")
    print(int_agg.round(3).to_string(index=False))
    print("\nWrote results_duel/{point,interval}{,_all}.csv")
    return {"point": point_agg, "interval": int_agg}


if __name__ == "__main__":
    run_all()
