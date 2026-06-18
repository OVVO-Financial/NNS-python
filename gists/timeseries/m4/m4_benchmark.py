"""NNS.ARMA on the M4 competition (Hourly + Weekly + Daily subsets).

The M4 competition (Makridakis et al., 2018-2020) is the recognised benchmark for
univariate forecasting: 100,000 series scored by OWA -- the average of sMAPE and
MASE, each normalised by the competition's Naive2 baseline (so Naive2 == 1.000).

This script runs the three *seasonal / longer-series* subsets, where a seasonal
ARMA method is best suited:

    Hourly   414 series   m=24   h=48
    Weekly   359 series   m=1    h=13     (M4 treats Weekly as non-seasonal)
    Daily   4227 series   m=1    h=14     (M4 treats Daily  as non-seasonal)

For every series it discovers seasonal periods with nns_seas, forecasts h steps
with NNS.ARMA.optim, and scores sMAPE / MASE / OWA against M4's own Naive2. No
neural network, no training loop.

Usage:
    python m4_benchmark.py                 # all three subsets, all series
    python m4_benchmark.py --freq Hourly   # one subset
    python m4_benchmark.py --limit 50      # first 50 series per subset (smoke test)
    python m4_benchmark.py --workers 4

Results are checkpointed to results/m4_<freq>.csv (resumable) and summarised to
results/m4_summary.csv.
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import nns as NNS

BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
CACHE = "m4_data"
RESULTS = "results"

# M4 official seasonality (m) and forecast horizon (h) per frequency.
SUBSETS = {
    "Hourly": {"m": 24, "h": 48},
    "Weekly": {"m": 1, "h": 13},
    "Daily": {"m": 1, "h": 14},
}


# ── Data ──────────────────────────────────────────────────────────────────────

def _download(rel: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    dest = os.path.join(CACHE, os.path.basename(rel))
    if not os.path.exists(dest):
        urlretrieve(f"{BASE}/{rel}", dest)
    return dest


def load_subset(freq: str) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    train = pd.read_csv(_download(f"Train/{freq}-train.csv"))
    test = pd.read_csv(_download(f"Test/{freq}-test.csv"))
    ids = train.iloc[:, 0].astype(str).to_numpy()
    y_train = [row[~np.isnan(row)] for row in train.iloc[:, 1:].to_numpy(dtype=float)]
    y_test = test.iloc[:, 1:].to_numpy(dtype=float)
    return ids, y_train, y_test


# ── Metrics ─────────────────────────────────────────────────────────────────

def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(forecast)
    diff = np.abs(actual - forecast)
    return float(200.0 * np.mean(np.where(denom == 0, 0.0, diff / denom)))


def mase(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray, m: int) -> float:
    m = m if len(train) > m else 1
    scale = np.mean(np.abs(train[m:] - train[:-m]))
    if scale == 0:
        scale = 1e-8
    return float(np.mean(np.abs(actual - forecast)) / scale)


def _is_seasonal(train: np.ndarray, m: int) -> bool:
    """M4's 90% autocorrelation seasonality test at lag m."""
    n = len(train)
    if m <= 1 or n < 3 * m:
        return False
    x = train - train.mean()
    acf = np.array([np.sum(x[k:] * x[:-k]) / np.sum(x**2) for k in range(1, m + 1)])
    limit = 1.645 * np.sqrt((1.0 + 2.0 * np.sum(acf[:-1] ** 2)) / n)
    return bool(np.abs(acf[-1]) > limit)


def naive2_forecast(train: np.ndarray, h: int, m: int) -> np.ndarray:
    """M4 Naive2: seasonally-adjusted naive (==Naive1 when non-seasonal)."""
    if not _is_seasonal(train, m):
        return np.repeat(train[-1], h)
    # Multiplicative seasonal indices via ratio-to-moving-average.
    n = len(train)
    ma = pd.Series(train).rolling(m, center=True).mean()
    if m % 2 == 0:
        ma = ma.rolling(2).mean().shift(-1)
    ratio = train / ma.to_numpy()
    idx = np.array([np.nanmean(ratio[i::m]) for i in range(m)])
    idx *= m / idx.sum()
    seas_train = np.tile(idx, n // m + 1)[:n]
    deseason = train / seas_train
    fut_seas = np.tile(idx, h // m + 1)[:h]
    return deseason[-1] * fut_seas


# ── Per-series NNS forecast ──────────────────────────────────────────────────

def nns_forecast(train: np.ndarray, h: int, m: int) -> np.ndarray:
    # Use M4's declared seasonality as the nns_seas modulo where it exists
    # (Hourly=24); leave it to auto-detect on the non-seasonal subsets.
    modulo = m if m > 1 else None
    seas = NNS.nns_seas(train, modulo=modulo, plot=False)
    periods = np.asarray(seas.get("periods", []), dtype=np.int64)
    # Cap to the same limit nns_arma_optim enforces (period < train_n / denom).
    train_n = int(0.8 * len(train))
    limit = train_n / min(4, max(3, round(train_n / 100)))
    periods = np.unique(periods[(periods > 1) & (periods < limit)])[:25]
    if periods.size == 0:
        periods = np.array([m if 1 < m < limit else 2], dtype=np.int64)
    fit = NNS.nns_arma_optim(  # default objective; let it search lin/nonlin/both
        variable=train,
        h=h,
        seasonal_factor=periods,
        negative_values=True,
        print_trace=False,
        plot=False,
    )
    return np.asarray(fit["results"], dtype=np.float64)


def _score_one(args: tuple) -> dict:
    sid, train, test, m = args
    h = len(test)
    try:
        fc = nns_forecast(train, h, m)
        ok = True
    except Exception:  # noqa: BLE001 -- a failed series falls back to Naive2
        fc = naive2_forecast(train, h, m)
        ok = False
    n2 = naive2_forecast(train, h, m)
    return {
        "id": sid,
        "nns_smape": smape(test, fc),
        "nns_mase": mase(test, fc, train, m),
        "naive2_smape": smape(test, n2),
        "naive2_mase": mase(test, n2, train, m),
        "nns_ok": ok,
    }


# ── Run one frequency ────────────────────────────────────────────────────────

def run_freq(freq: str, limit: int | None, workers: int) -> pd.DataFrame:
    m, h = SUBSETS[freq]["m"], SUBSETS[freq]["h"]
    ids, y_train, y_test = load_subset(freq)
    if limit:
        ids, y_train, y_test = ids[:limit], y_train[:limit], y_test[:limit]

    os.makedirs(RESULTS, exist_ok=True)
    ckpt = os.path.join(RESULTS, f"m4_{freq}.csv")
    done = set(pd.read_csv(ckpt)["id"]) if os.path.exists(ckpt) else set()

    tasks = [
        (sid, y_train[i], y_test[i][: h], m)
        for i, sid in enumerate(ids)
        if sid not in done
    ]
    print(f"[{freq}] {len(tasks)} series to run ({len(done)} cached), m={m}, h={h}")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_score_one, t): t[0] for t in tasks}
        for n_done, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if n_done % 50 == 0 or n_done == len(tasks):
                pd.DataFrame(rows).to_csv(
                    ckpt, mode="a", header=not os.path.exists(ckpt), index=False
                )
                print(f"  [{freq}] {n_done}/{len(tasks)}")
                rows = []
    if rows:
        pd.DataFrame(rows).to_csv(ckpt, mode="a", header=not os.path.exists(ckpt), index=False)
    return pd.read_csv(ckpt)


def summarise(freq: str, df: pd.DataFrame) -> dict:
    nns_s, nns_m = df["nns_smape"].mean(), df["nns_mase"].mean()
    n2_s, n2_m = df["naive2_smape"].mean(), df["naive2_mase"].mean()
    owa = 0.5 * (nns_s / n2_s + nns_m / n2_m)
    return {
        "freq": freq,
        "series": len(df),
        "nns_fail": int((~df["nns_ok"]).sum()),
        "nns_sMAPE": round(nns_s, 3),
        "nns_MASE": round(nns_m, 3),
        "naive2_sMAPE": round(n2_s, 3),
        "naive2_MASE": round(n2_m, 3),
        "NNS_OWA": round(owa, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", choices=list(SUBSETS), help="run a single subset")
    ap.add_argument("--limit", type=int, default=None, help="first N series per subset")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    args = ap.parse_args()

    freqs = [args.freq] if args.freq else list(SUBSETS)
    summary = [summarise(f, run_freq(f, args.limit, args.workers)) for f in freqs]

    os.makedirs(RESULTS, exist_ok=True)
    out = pd.DataFrame(summary)
    out.to_csv(os.path.join(RESULTS, "m4_summary.csv"), index=False)
    print("\n=== M4 BENCHMARK (NNS.ARMA.optim) ===\n")
    print(out.to_string(index=False))
    print("\nOWA < 1.000 beats the Naive2 baseline; lower is better.")


if __name__ == "__main__":
    main()
