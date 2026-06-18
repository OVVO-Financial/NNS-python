"""
run_conformal.py
================================================================================
Authentic Python port of the R NNS ARMA Conformal Benchmark.

Fixes:
- Removed all text-truncation slices ([:3] and [:4]) from periods and weights.
- Implements dynamic LIVE reporting per chunk so you see obj.fn and weights instantly.
- Preserves native significance order, MSE obj_fn, and linear_approximation=True.
"""

from __future__ import annotations

import os
import math
import warnings
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Dynamic NNS Function Resolution ──────────────────────────────────────────
if hasattr(NNS, "lpm_var"):
    _lpm_var_fn = NNS.lpm_var
elif hasattr(NNS, "LPM_VaR"):
    _lpm_var_fn = NNS.LPM_VaR
else:
    _lpm_var_fn = getattr(NNS, "lpm_var", getattr(NNS, "LPM_VaR", None))

if _lpm_var_fn is None:
    raise AttributeError("Could not locate lpm_var or LPM_VaR inside the installed nns package.")

# ── Constants ────────────────────────────────────────────────────────────────

ALPHA       = 0.10
TARGET_COV  = 1.0 - ALPHA
N_LAGS      = 12
FIT_END     = 700
CAL_END     = 1000
WINDOW      = 100
N_SEEDS     = 10

TRAINING_FRAC = 0.90
MAX_H         = 250

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ── Data-generating process ──────────────────────────────────────────────────

def make_timeseries(T: int = 3500, seed: int = 0, heavy_tail: bool = False) -> dict:
    """Non-linear, heteroskedastic AR(1) DGP with piecewise volatility regimes."""
    rng = np.random.default_rng(seed + 1)
    tt    = np.arange(1, T + 1, dtype=float)
    level = 0.002 * tt + 1.50 * np.sin(2 * np.pi * tt / 50) + 0.75 * np.sin(2 * np.pi * tt / 200)
    sigma = np.ones(T)
    sigma[(tt > 900)  & (tt <= 1400)] = 2.5
    sigma[(tt > 1900) & (tt <= 2450)] = 0.55
    sigma[(tt > 2800)]                = 1.8
    eps = rng.standard_t(5, size=T) / math.sqrt(5 / 3) if heavy_tail else rng.standard_normal(T)
    y   = np.empty(T)
    y[0] = level[0] + sigma[0] * eps[0]
    for i in range(1, T):
        y[i] = level[i] + 0.55 * (y[i - 1] - level[i - 1]) + sigma[i] * eps[i]
    return {"y": y, "level": level, "sigma": sigma}


def true_conditional_mean(d: dict, raw_idx: np.ndarray) -> np.ndarray:
    return d["level"][raw_idx] + 0.55 * (d["y"][raw_idx - 1] - d["level"][raw_idx - 1])

# ── Lag-feature matrix & ridge baseline ─────────────────────────────────────

def lag_features(y: np.ndarray, n_lags: int = N_LAGS):
    n  = len(y)
    yy = y[n_lags:]
    X  = np.column_stack([y[n_lags - k: n - k] for k in range(1, n_lags + 1)])
    return X, yy


def ridge_forecast(X: np.ndarray, yy: np.ndarray, fit_end: int) -> np.ndarray:
    if HAS_SKLEARN:
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[:fit_end], yy[:fit_end])
        return model.predict(X)
    Xtr = np.column_stack([X[:fit_end], np.ones(fit_end)])
    coef, *_ = np.linalg.lstsq(Xtr, yy[:fit_end], rcond=None)
    Xall = np.column_stack([X, np.ones(len(X))])
    return Xall @ coef

# ── Scoring helpers ──────────────────────────────────────────────────────────

def z_alpha(alpha: float = ALPHA) -> float:
    return float(stats.norm.ppf(1 - alpha / 2))


def coverage(lo, hi, y) -> float:
    lo, hi, y = map(np.asarray, (lo, hi, y))
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def frac_infinite(lo, hi) -> float:
    lo, hi = np.asarray(lo), np.asarray(hi)
    return float(np.mean(~np.isfinite(lo) | ~np.isfinite(hi)))


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
                         + (2 / alpha) * np.maximum(lo - y, 0)
                         + (2 / alpha) * np.maximum(y - hi, 0)))


def coverage_by_stratum(lo, hi, y, sigma, k: int = 4) -> list[float]:
    lo, hi, y, sigma = map(np.asarray, (lo, hi, y, sigma))
    ranks = stats.rankdata(sigma, method="ordinal")
    grp   = np.ceil(ranks / len(ranks) * k).astype(int).clip(1, k)
    return [coverage(lo[grp == j], hi[grp == j], y[grp == j]) for j in range(1, k + 1)]


def gaussian_interval(mu, sigma, alpha: float = ALPHA):
    sigma = np.maximum(np.asarray(sigma, float), 1e-8)
    z     = z_alpha(alpha)
    return mu - z * sigma, mu + z * sigma


def crps_gaussian(mu, sigma, y) -> float:
    sigma = np.maximum(np.asarray(sigma, float), 1e-8)
    z     = (np.asarray(y) - np.asarray(mu)) / sigma
    return float(np.mean(sigma * (z * (2 * stats.norm.cdf(z) - 1)
                                  + 2 * stats.norm.pdf(z)
                                  - 1 / math.sqrt(math.pi))))


def log_score_gaussian(mu, sigma, y) -> float:
    sigma = np.maximum(np.asarray(sigma, float), 1e-8)
    return float(np.mean(-stats.norm.logpdf(np.asarray(y),
                                            loc=np.asarray(mu), scale=sigma)))


def score_method(name: str, family: str, lo, hi, y_te, sig_te,
                 mu_=None, s_=None, precomputed_crps=None, precomputed_logscore=None) -> dict:
    lo_v = np.minimum(np.asarray(lo, float), np.asarray(hi, float))
    hi_v = np.maximum(np.asarray(lo, float), np.asarray(hi, float))
    cbs  = coverage_by_stratum(lo_v, hi_v, y_te, sig_te)
    row  = {
        "method":         name,
        "family":         family,
        "marg_cov":       coverage(lo_v, hi_v, y_te),
        "worst_win_cov":  worst_window_coverage(lo_v, hi_v, y_te),
        "cov_lowvol":     cbs[0],
        "cov_hivol":      cbs[-1],
        "cond_cov_gap":   max(abs(c - TARGET_COV) for c in cbs if not math.isnan(c)),
        "width":          mean_width(lo_v, hi_v),
        "frac_inf":       frac_infinite(lo_v, hi_v),
        "interval_score": interval_score(lo_v, hi_v, y_te),
        "CRPS":           float("nan"),
        "logscore":       float("nan"),
    }
    
    if precomputed_crps is not None and precomputed_logscore is not None:
        row["CRPS"]     = float(np.mean(precomputed_crps))
        row["logscore"] = float(np.mean(precomputed_logscore))
    elif mu_ is not None and s_ is not None:
        row["CRPS"]     = crps_gaussian(mu_, s_, y_te)
        row["logscore"] = log_score_gaussian(mu_, s_, y_te)
    return row

# ── Conformal baselines ─────────────────────────────────────────────────────

def fixed_split_cp(mu_te, resid_cal, alpha: float = ALPHA):
    scores = np.sort(np.abs(resid_cal))
    k      = int(np.ceil((len(scores) + 1) * (1 - alpha))) - 1
    q      = float("inf") if k >= len(scores) else scores[k]
    return mu_te - q, mu_te + q


def aci(mu_te, y_te, alpha: float = ALPHA, gamma: float = 0.03, warm=None):
    n            = len(y_te)
    lo, hi       = np.empty(n), np.empty(n)
    alpha_t      = alpha
    hist_scores  = list(np.abs(warm)) if warm is not None else []
    for t in range(n):
        k   = int(np.ceil((len(hist_scores) + 1) * (1 - alpha_t))) - 1
        q_t = float("inf") if not hist_scores or k >= len(hist_scores) else sorted(hist_scores)[k]
        lo[t], hi[t] = mu_te[t] - q_t, mu_te[t] + q_t
        err_t    = int(y_te[t] < lo[t] or y_te[t] > hi[t])
        alpha_t  = np.clip(alpha_t + gamma * (alpha - err_t), 0.001, 0.999)
        hist_scores.append(abs(y_te[t] - mu_te[t]))
    return lo, hi


def agaci(mu_te, y_te, alpha: float = ALPHA, warm=None,
          gammas=(0.001, 0.005, 0.01, 0.02, 0.05, 0.10)):
    experts = [aci(mu_te, y_te, alpha, g, warm) for g in gammas]
    lo = np.mean([e[0] for e in experts], axis=0)
    hi = np.mean([e[1] for e in experts], axis=0)
    return lo, hi


def conformal_pid(mu_te, y_te, alpha: float = ALPHA, warm=None,
                  Kp=0.1, Ki=0.01, Kd=0.001):
    n           = len(y_te)
    lo, hi      = np.empty(n), np.empty(n)
    hist_scores = list(np.abs(warm)) if warm is not None else []
    err_prev, integral = 0.0, 0.0
    for t in range(n):
        k   = int(np.ceil((len(hist_scores) + 1) * (1 - alpha))) - 1
        q_t = float("inf") if not hist_scores or k >= len(hist_scores) else sorted(hist_scores)[k]
        lo[t], hi[t] = mu_te[t] - q_t, mu_te[t] + q_t
        err_t    = int(y_te[t] < lo[t] or y_te[t] > hi[t]) - alpha
        integral += err_t
        delta    = Kp * err_t + Ki * integral + Kd * (err_t - err_prev)
        err_prev = err_t
        hist_scores.append(abs(y_te[t] - mu_te[t]) * max(1e-6, 1 + delta))
    return lo, hi


def nexcp(mu_te, y_te, alpha: float = ALPHA, warm=None, decay: float = 0.99):
    n            = len(y_te)
    lo, hi       = np.empty(n), np.empty(n)
    hist_scores  = list(np.abs(warm)) if warm is not None else []
    hist_weights = [decay ** (len(warm) - 1 - i) for i in range(len(warm))] if warm is not None else []
    for t in range(n):
        if not hist_scores:
            q_t = float("inf")
        else:
            w_arr   = np.array(hist_weights)
            w_norm  = w_arr / w_arr.sum()
            ord_idx = np.argsort(hist_scores)
            cum_w   = np.cumsum(w_norm[ord_idx])
            idx     = np.where(cum_w >= (1 - alpha))[0]
            q_t     = float("inf") if len(idx) == 0 else hist_scores[ord_idx[idx[0]]]
        lo[t], hi[t] = mu_te[t] - q_t, mu_te[t] + q_t
        hist_scores.append(abs(y_te[t] - mu_te[t]))
        hist_weights = [w * decay for w in hist_weights] + [1.0]
    return lo, hi

# ── Probabilistic baselines ──────────────────────────────────────────────────

def ewma_vol(mu_te, y_te, alpha: float = ALPHA, warm=None, lam: float = 0.94):
    warm_resid  = (np.asarray(warm) if warm is not None else np.array([]))
    te_resid    = np.asarray(y_te) - np.asarray(mu_te)
    all_resid   = np.concatenate([warm_resid, te_resid])
    var_vec     = np.empty(len(all_resid))
    var_vec[0]  = all_resid[0] ** 2
    for i in range(1, len(all_resid)):
        var_vec[i] = lam * var_vec[i - 1] + (1 - lam) * all_resid[i - 1] ** 2
    sig_te = np.maximum(np.sqrt(var_vec[len(warm_resid):]), 1e-6)
    z      = z_alpha(alpha)
    return mu_te - z * sig_te, mu_te + z * sig_te, np.asarray(mu_te), sig_te


def recal_const(mu_te, resid_cal, alpha: float = ALPHA):
    s  = float(np.std(resid_cal))
    z  = z_alpha(alpha)
    s_arr = np.full(len(mu_te), s)
    return mu_te - z * s, mu_te + z * s, np.asarray(mu_te), s_arr

# ── NNS seasonal period discovery ───────────────────────────────────────────

def get_nns_seas_periods(series: np.ndarray, train_n: int) -> list[int]:
    max_period = int(train_n / 3) - 1
    try:
        seas = NNS.nns_seas(series, plot=False)
        raw  = seas.get("periods", np.array([]))
        
        seen = set()
        valid = []
        for p in raw:
            p_int = int(p)
            if 1 < p_int < max_period and p_int not in seen:
                seen.add(p_int)
                valid.append(p_int)
                
        return valid if valid else [2]
    except Exception:
        return [2]

# ── Authentic Nonparametric NNS ARMA walk-forward ───────────────────────────

def run_nns_walkforward(d: dict, seed: int, training_frac: float = TRAINING_FRAC, max_h: int = MAX_H):
    y     = d["y"]
    T_raw = len(y)

    all_pred      = []
    all_lo        = []
    all_hi        = []
    all_y         = []
    all_sig       = []
    all_errors    = []   
    all_crps      = []   
    all_logscores = []   
    chunks        = []

    current_train = N_LAGS + CAL_END
    chunk_id      = 0

    while current_train < T_raw:
        chunk_id   += 1
        remaining   = T_raw - current_train
        implied_h   = max(1, int(current_train * (1 - training_frac) / training_frac))
        h_i         = min(max_h, remaining, implied_h)
        end_i       = current_train + h_i
        train_slice = y[:current_train]

        periods = get_nns_seas_periods(train_slice, current_train)

        try:
            fit = NNS.nns_arma_optim(
                variable         = y[:end_i],
                training_set     = current_train,
                seasonal_factor  = periods,
                negative_values  = True,
                obj_fn           = lambda predicted, actual: np.mean((predicted - actual) ** 2),
                objective        = "min",
                linear_approximation = True,
                pred_int         = TARGET_COV,
                print_trace      = False,
                plot             = False,
            )
        except Exception as exc:
            print(f"  [NNS chunk {chunk_id} failed: {exc}] – skipping")
            current_train = end_i
            continue

        pred_raw = np.asarray(fit["results"],       float)
        lo_raw   = np.asarray(fit["lower.pred.int"], float)
        hi_raw   = np.asarray(fit["upper.pred.int"], float)
        errors_c = np.asarray(fit["errors"],         float)

        lo = np.minimum(lo_raw, hi_raw)
        hi = np.maximum(lo_raw, hi_raw)
        pred_idx = np.arange(current_train, end_i)
        y_true_chunk = y[pred_idx]

        # Extract complete historical component weights vector (no text slicing)
        raw_weights = fit.get("weights", fit.get("opt.weights", "N/A"))
        if isinstance(raw_weights, (np.ndarray, list)) and len(raw_weights) > 0:
            weights_str = "[" + ", ".join([f"{float(w):.3f}" for w in raw_weights]) + "]"
        else:
            weights_str = str(raw_weights)

        # ── LIVE REAL-TIME OPTIMIZATION METRIC COUT ──
        print(f"  ⚡ [SEED {seed} | CHUNK {chunk_id}] Complete Optimization Output:")
        print(f"     -> Target Train Size: {current_train} steps")
        print(f"     -> Discovered Frequencies (All): {periods}")
        print(f"     -> Calculated Parameter Weights: {weights_str}")
        print(f"     -> In-Sample Objective (MSE):   {fit.get('obj.fn', 'N/A'):.6f}\n")

        # ── Step-by-Step True Nonparametric Scoring via LPM.VaR ──
        chunk_crps = []
        chunk_logscores = []
        percentiles = np.linspace(0.001, 0.999, 100)
        
        for t in range(h_i):
            lower_t = lo[t]
            point_t = pred_raw[t]
            upper_t = hi[t]
            support_t = np.array([lower_t, point_t, upper_t])
            
            step_samples = np.array([_lpm_var_fn(p, 2, support_t) for p in percentiles])
            y_t = y_true_chunk[t]
            
            # Nonparametric Empirical CRPS calculation
            s_sorted = np.sort(step_samples)
            M = len(s_sorted)
            term1 = np.abs(step_samples - y_t).mean()
            idx = np.arange(1, M + 1)
            weights = 2 * idx - M - 1
            term2 = np.sum(weights * s_sorted) / (M ** 2)
            chunk_crps.append(term1 - term2)
            
            # Nonparametric Log-score evaluation via KDE
            try:
                kde = stats.gaussian_kde(step_samples)
                chunk_logscores.append(-float(kde.logpdf(y_t)[0]))
            except Exception:
                std = max(np.std(step_samples), 1e-6)
                chunk_logscores.append(-float(stats.norm.logpdf(y_t, loc=point_t, scale=std)))

        all_pred.extend(pred_raw)
        all_lo.extend(lo)
        all_hi.extend(hi)
        all_y.extend(y_true_chunk)
        all_sig.extend(d["sigma"][pred_idx])
        all_errors.extend(errors_c)
        all_crps.extend(chunk_crps)
        all_logscores.extend(chunk_logscores)

        chunks.append({
            "seed":        seed,
            "chunk":       chunk_id,
            "train_end":   current_train,
            "h":           h_i,
            "periods":     str(periods),
            "weights":     weights_str,
            "obj_fn":      float(fit.get("obj.fn", float("nan"))),
            "bias_shift":  float(fit.get("bias.shift", float("nan"))),
        })

        current_train = end_i

    return {
        "pred":     np.array(all_pred),
        "lo":       np.array(all_lo),
        "hi":       np.array(all_hi),
        "y":        np.array(all_y),
        "sigma":    np.array(all_sig),
        "errors":   np.array(all_errors),
        "CRPS":     np.array(all_crps),
        "logscore": np.array(all_logscores),
        "chunks":   pd.DataFrame(chunks),
    }

# ── Full per-seed run ────────────────────────────────────────────────────────

def run_once(seed: int = 0, heavy_tail: bool = False):
    d = make_timeseries(T=3500, seed=seed, heavy_tail=heavy_tail)
    X, yy = lag_features(d["y"], N_LAGS)
    mu    = ridge_forecast(X, yy, FIT_END)

    sig_all  = d["sigma"][N_LAGS:]
    resid    = yy - mu
    te_slice = slice(CAL_END, len(yy))
    mu_te    = mu[te_slice]
    y_te     = yy[te_slice]
    sig_te   = sig_all[te_slice]

    raw_te_idx  = np.arange(CAL_END, len(yy)) + N_LAGS
    true_mu_te  = true_conditional_mean(d, raw_te_idx)

    resid_cal = resid[FIT_END:CAL_END]
    warm      = resid[:CAL_END]

    methods: dict[str, tuple] = {}

    methods["fixed split (CP)"] = (*fixed_split_cp(mu_te, resid_cal), None, None, "cp")
    methods["ACI"] = (*aci(mu_te, y_te, warm=warm), None, None, "cp")
    methods["AgACI"] = (*agaci(mu_te, y_te, warm=warm), None, None, "cp")
    methods["conformal PID"] = (*conformal_pid(mu_te, y_te, warm=warm), None, None, "cp")
    methods["NexCP (weighted)"] = (*nexcp(mu_te, y_te, warm=warm), None, None, "cp")

    or_lo, or_hi = gaussian_interval(true_mu_te, sig_te)
    methods["oracle (true μ,σ)"] = (or_lo, or_hi, true_mu_te, sig_te, "oracle")
    
    os_lo, os_hi = gaussian_interval(mu_te, sig_te)
    methods["true σ on est. μ"] = (os_lo, os_hi, mu_te, sig_te, "oracle")

    ew_lo, ew_hi, ew_mu, ew_s = ewma_vol(mu_te, y_te, warm=warm)
    methods["EWMA-vol Gaussian"] = (ew_lo, ew_hi, ew_mu, ew_s, "prob")
    
    rc_lo, rc_hi, rc_mu, rc_s = recal_const(mu_te, resid_cal)
    methods["static Gaussian (recal)"] = (rc_lo, rc_hi, rc_mu, rc_s, "prob")

    nns_wf = run_nns_walkforward(d, seed=seed)
    methods["NNS.ARMA.optim"] = (nns_wf["lo"], nns_wf["hi"], None, None, "nns")

    rows = []
    for name, (lo, hi, mu_, s_, fam) in methods.items():
        is_nns = (fam == "nns")
        v_y   = nns_wf["y"]     if is_nns else y_te
        v_sig = nns_wf["sigma"] if is_nns else sig_te
        
        p_crps = nns_wf["CRPS"]     if is_nns else None
        p_log  = nns_wf["logscore"] if is_nns else None
        
        rows.append(score_method(name, fam, lo, hi, v_y, v_sig, mu_, s_,
                                 precomputed_crps=p_crps, precomputed_logscore=p_log))

    return {
        "scores":  pd.DataFrame(rows),
        "methods": methods,
        "y_te":    y_te,
        "sig_te":  sig_te,
        "mu_te":   mu_te,
        "true_mu_te": true_mu_te,
        "nns_wf":  nns_wf,
    }

# ── Figures ───────────────────────────────────────────────────────────────────

def make_figures(keep: dict, agg: pd.DataFrame) -> None:
    if not HAS_MPL:
        print("[skip figures: matplotlib not installed]")
        return

    methods = keep["methods"]
    y_te    = keep["y_te"]
    sig_te  = keep["sig_te"]
    nns_wf  = keep["nns_wf"]
    z       = z_alpha(ALPHA)
    t_vec   = np.arange(len(y_te))

    sel   = ["fixed split (CP)", "ACI", "conformal PID", "NNS.ARMA.optim", "EWMA-vol Gaussian"]
    cols  = ["#1f4ed8", "#dc2626", "#16a34a", "#7e22ce", "#15803d"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for nm, c in zip(sel, cols):
        if nm not in methods:
            continue
        lo, hi, *_ = methods[nm]
        fam        = methods[nm][4]
        y_v        = nns_wf["y"] if fam == "nns" else y_te
        rc         = rolling_coverage(lo, hi, y_v)
        ax.plot(np.arange(len(rc)), rc, lw=1.4, color=c, label=nm)
    ax.axhline(TARGET_COV, lw=1, ls="--", color="k", label="target 0.90")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel(f"test step  (rolling window = {WINDOW})")
    ax.set_ylabel("coverage")
    ax.set_title("Rolling coverage under drift")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig("figures/ts_coverage.png", dpi=130)
    plt.close(fig)

    fam_cols = {"cp": "#1f4ed8", "prob": "#15803d", "oracle": "#c2410c", "nns": "#7e22ce"}
    fig, ax  = plt.subplots(figsize=(9.5, 6.5))
    for _, r in agg.iterrows():
        c = fam_cols.get(r["family"], "#666")
        ax.scatter(r["worst_win_cov"], r["interval_score"], c=c, s=55, zorder=3)
        ax.annotate(r["method"], (r["worst_win_cov"], r["interval_score"]),
                    fontsize=6.5, color=c, textcoords="offset points", xytext=(3, 3))
    ax.axvline(TARGET_COV, lw=1, ls="--", color="k")
    ax.set_xlabel("worst rolling-window coverage  →  conditional coverage")
    ax.set_ylabel("interval (Winkler) score  ↓  better")
    ax.set_title("Efficiency vs. worst-case coverage")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0], marker="o", ls="", color=fam_cols[k],
                               label={"cp":"conformal","prob":"probabilistic",
                                      "oracle":"oracle","nns":"NNS"}[k])
                       for k in fam_cols], loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/ts_plane.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_vec, np.clip(2 * z * sig_te, 0, 30), color="k", lw=1.3, label="oracle width 2·z·σ_t")
    for nm, c, fam in [
        ("NNS.ARMA.optim", "#7e22ce", "nns"),
        ("EWMA-vol Gaussian",            "#15803d", "prob"),
        ("fixed split (CP)",             "#1f4ed8", "cp"),
    ]:
        if nm in methods:
            lo, hi, *_ = methods[nm]
            ax.plot(np.arange(len(lo)), np.clip(hi - lo, 0, 30), lw=1.1, color=c, label=nm)
    ax.set_xlabel("test step")
    ax.set_ylabel("interval width")
    ax.set_title("Does the interval width track volatility?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/ts_width.png", dpi=130)
    plt.close(fig)

    nns_y  = nns_wf["y"]
    nns_mu = nns_wf["pred"]
    nns_lo = nns_wf["lo"]
    nns_hi = nns_wf["hi"]
    errors = nns_wf["errors"]
    oos_errors = nns_y - nns_mu

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    clip = min(400, len(nns_y))
    ax0  = fig.add_subplot(gs[0, :2])
    ax0.fill_between(np.arange(clip), nns_lo[:clip], nns_hi[:clip],
                     alpha=0.25, color="#7e22ce", label=f"{int(TARGET_COV*100)}% PI")
    ax0.plot(np.arange(clip), nns_y[:clip],   color="k",       lw=0.8,  label="actual")
    ax0.plot(np.arange(clip), nns_mu[:clip],  color="#7e22ce", lw=1.0,  label="NNS forecast")
    ax0.set_title("NNS.ARMA.optim walk-forward: forecast + prediction interval")
    ax0.set_xlabel("test step")
    ax0.legend(fontsize=8)

    ax1  = fig.add_subplot(gs[0, 2])
    ax1.plot(np.arange(clip), (nns_hi - nns_lo)[:clip], color="#7e22ce", lw=0.9)
    ax1.set_title("NNS Prediction Interval Width")
    ax1.set_xlabel("test step")
    ax1.set_ylabel("width")

    ax2  = fig.add_subplot(gs[1, 0])
    ax2.hist(oos_errors, bins=40, density=True, color="#7e22ce", alpha=0.55, edgecolor="white", lw=0.4)
    xe = np.linspace(oos_errors.min(), oos_errors.max(), 200)
    kde_oos = stats.gaussian_kde(oos_errors)
    ax2.plot(xe, kde_oos.pdf(xe), "k-", lw=1.2, label="OOS Error KDE")
    ax2.set_title("Out-of-Sample Prediction Errors")
    ax2.set_xlabel("y – ŷ")
    ax2.legend(fontsize=8)

    ax3  = fig.add_subplot(gs[1, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(oos_errors, dist="norm")
    ax3.scatter(osm, osr, s=5, color="#7e22ce", alpha=0.5)
    ax3.plot(osm, slope * np.array(osm) + intercept, "k--", lw=1.2)
    ax3.set_title("Normal Q-Q Plot of OOS Errors")
    ax3.set_xlabel("Theoretical quantiles")
    ax3.set_ylabel("Sample quantiles")

    ax4  = fig.add_subplot(gs[1, 2])
    ax4.hist(errors, bins=50, density=True, color="#1f4ed8", alpha=0.55, edgecolor="white", lw=0.4)
    fit_mu, fit_s = np.mean(errors), np.std(errors)
    x_e  = np.linspace(errors.min(), errors.max(), 300)
    ax4.plot(x_e, stats.norm.pdf(x_e, fit_mu, fit_s), "k--", lw=1.2, label=f"N({fit_mu:.2f}, {fit_s:.2f})")
    ax4.set_title("NNS In-Sample Errors (pooled)")
    ax4.set_xlabel("error")
    ax4.legend(fontsize=8)

    fig.suptitle("NNS.ARMA.optim — Authentic Nonparametric Diagnostics", fontsize=12)
    fig.savefig("figures/ts_distributions.png", dpi=130)
    plt.close(fig)
    print("  figures/ts_distributions.png")

# ── Aggregate over seeds & print results ─────────────────────────────────────

def run_all() -> dict:
    all_scores: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    res_keep   = None

    for seed in range(N_SEEDS):
        print(f"\n=== seed {seed} ===")
        res = run_once(seed=seed, heavy_tail=False)
        all_scores.append(res["scores"])
        all_diagnostics.append(res["nns_wf"]["chunks"])
        if seed == 0:
            res_keep = res

    scores_dt = pd.concat(all_scores, ignore_index=True)
    metric_cols = ["marg_cov", "worst_win_cov", "cov_lowvol", "cov_hivol",
                   "cond_cov_gap", "width", "frac_inf",
                   "interval_score", "CRPS", "logscore"]
    agg = (scores_dt
            .groupby(["method", "family"], sort=False)[metric_cols]
            .mean()
            .reset_index()
            .sort_values("interval_score"))

    col_order = ["method", "family"] + metric_cols
    agg = agg[[c for c in col_order if c in agg.columns]]

    scores_dt.to_csv("results/ts_results_all.csv", index=False)
    agg.to_csv("results/ts_results.csv", index=False)

    # ── FINAL COMPREHENSIVE RECAP DICTIONARY BLOCK ──
    print("\n" + "="*115)
    print("                      NNS OPTIMIZATION DIAGNOSTICS REPORT (COMPLETE RECORD)")
    print("="*115)
    diag_df = pd.concat(all_diagnostics, ignore_index=True)
    report_cols = ["seed", "chunk", "train_end", "obj_fn", "periods", "weights"]
    
    # Configure pandas formatting bounds so long values are visible
    with pd.option_context('display.max_colwidth', None, 'display.width', 1000):
        print(diag_df[report_cols].to_string(index=False, max_rows=100))
    print("="*115 + "\n")

    print(f"=== TIME-SERIES BENCHMARK  "
          f"(mean over {N_SEEDS} seeds, alpha={ALPHA}, target cov={TARGET_COV}) ===\n")
    print(agg.round(3).to_string(index=False))

    print("\nWrote:")
    print("  results/ts_results.csv")
    print("  results/ts_results_all.csv")

    if res_keep is not None:
        make_figures(res_keep, agg)

    return {"scores": scores_dt, "summary": agg}


if __name__ == "__main__":
    run_all()
