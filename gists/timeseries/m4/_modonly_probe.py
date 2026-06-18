"""Local experimentation harness for NNS.ARMA on the M4 seasonal subsets.

Self-contained: downloads the M4 data on first run (cached in ./m4_data), then
scores NNS forecasts vs the Naive (last-value) baseline on a sample of series.

Tweak the CONFIG block, run `python _modonly_probe.py`, compare OWA/sMAPE, then
revert to m4_benchmark.py for the full committed setup.
"""

import warnings
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import nns as NNS

# ── CONFIG — edit and re-run ────────────────────────────────────────────────
N_SERIES = 30          # series per subset (None = all)
MODULO_FROM_M = True   # use M4 seasonality (Hourly=24) as nns_seas modulo; else MODULO below
MODULO = {"Hourly": 24, "Daily": 7, "Weekly": 12}
MOD_ONLY = True        # nns_seas mod_only
USE_OPTIM = True       # True: nns_arma_optim (searches lin/nonlin/both); False: plain nns_arma
TRAINING_SET = None    # int to pass an explicit training_set to the optimizer; None = default
OBJ_FN = None          # e.g. lambda p, a: np.mean(np.abs(p - a)); None = optimizer default
MAX_PERIODS = 25
SUBSETS = {"Hourly": (24, 48), "Weekly": (1, 13), "Daily": (1, 14)}  # name -> (m, h)
# ─────────────────────────────────────────────────────────────────────────────

BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"


def fetch(freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import os
    os.makedirs("m4_data", exist_ok=True)
    paths = {}
    for split in ("Train", "Test"):
        dest = f"m4_data/{freq}-{split.lower()}.csv"
        if not os.path.exists(dest):
            print(f"  downloading {freq}-{split.lower()}.csv ...")
            urlretrieve(f"{BASE}/{split}/{freq}-{split.lower()}.csv", dest)
        paths[split] = dest
    return pd.read_csv(paths["Train"]), pd.read_csv(paths["Test"])


def smape(a, f):
    d = np.abs(a) + np.abs(f)
    return 200 * np.mean(np.where(d == 0, 0, np.abs(a - f) / d))


def mase(a, f, tr, m=1):
    s = np.mean(np.abs(tr[m:] - tr[:-m])) or 1e-8
    return np.mean(np.abs(a - f)) / s


def periods_for(y, freq):
    modulo = (SUBSETS[freq][0] if MODULO_FROM_M else MODULO[freq])
    modulo = modulo if modulo and modulo > 1 else None
    tn = int(0.8 * len(y))
    lim = tn / min(4, max(3, round(tn / 100)))
    p = np.asarray(NNS.nns_seas(y, modulo=modulo, mod_only=MOD_ONLY).get("periods", []), int)
    p = np.unique(p[(p > 1) & (p < lim)])[:MAX_PERIODS]
    return p if p.size else np.array([2])


def forecast(y, h, freq):
    sf = periods_for(y, freq)
    kw = dict(variable=y, h=h, seasonal_factor=sf, negative_values=True, print_trace=False)
    if OBJ_FN is not None:
        kw.update(obj_fn=OBJ_FN, objective="min")
    if USE_OPTIM:
        if TRAINING_SET is not None:
            kw["training_set"] = TRAINING_SET
        return np.asarray(NNS.nns_arma_optim(**kw)["results"], float)
    kw.pop("print_trace", None)
    out = NNS.nns_arma(variable=y, h=h, seasonal_factor=sf, negative_values=True)
    return np.asarray(out["results"] if isinstance(out, dict) else out, float)


def run(freq):
    m, h = SUBSETS[freq]
    tr, te = fetch(freq)
    n = len(tr) if N_SERIES is None else min(N_SERIES, len(tr))
    ns, n2s, nm, n2m, fail = [], [], [], [], 0
    for i in range(n):
        y = tr.iloc[i, 1:].dropna().astype(float).values
        test = te.iloc[i, 1:].dropna().astype(float).values[:h]
        try:
            fc = forecast(y, h, freq)
        except Exception:
            fc = np.repeat(y[-1], h); fail += 1
        n2 = np.repeat(y[-1], h)
        ns.append(smape(test, fc)); nm.append(mase(test, fc, y))
        n2s.append(smape(test, n2)); n2m.append(mase(test, n2, y))
    owa = 0.5 * (np.mean(ns) / np.mean(n2s) + np.mean(nm) / np.mean(n2m))
    print(f"{freq:7s} n={n:<4} OWA={owa:.3f}  sMAPE={np.mean(ns):.2f}  MASE={np.mean(nm):.2f}  fail={fail}")


if __name__ == "__main__":
    print(f"config: optim={USE_OPTIM} mod_only={MOD_ONLY} modulo_from_m={MODULO_FROM_M} "
          f"obj_fn={'custom' if OBJ_FN else 'default'} training_set={TRAINING_SET}")
    for freq in SUBSETS:
        run(freq)
    print("\nOWA < 1.000 beats the Naive baseline; lower is better.")
