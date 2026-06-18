"""Compare nns_seas mod_only=True vs False (logical modulo, default optim objective).

Run from a dir containing m4_data/<freq>-train.csv and -test.csv.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import nns as NNS


def smape(a, f):
    d = np.abs(a) + np.abs(f)
    return 200 * np.mean(np.where(d == 0, 0, np.abs(a - f) / d))


def mase(a, f, tr, m=1):
    s = np.mean(np.abs(tr[m:] - tr[:-m])) or 1e-8
    return np.mean(np.abs(a - f)) / s


def periods_for(y, modulo, mod_only):
    tn = int(0.8 * len(y))
    lim = tn / min(4, max(3, round(tn / 100)))
    p = np.asarray(NNS.nns_seas(y, modulo=modulo, mod_only=mod_only).get("periods", []), int)
    p = np.unique(p[(p > 1) & (p < lim)])[:25]
    return p if p.size else np.array([modulo if modulo < lim else 2])


def run(freq, h, modulo, mod_only, n=30):
    tr = pd.read_csv(f"m4_data/{freq}-train.csv")
    te = pd.read_csv(f"m4_data/{freq}-test.csv")
    ns, n2s, nm, n2m = [], [], [], []
    for i in range(n):
        y = tr.iloc[i, 1:].dropna().astype(float).values
        test = te.iloc[i, 1:].dropna().astype(float).values[:h]
        try:
            fc = np.asarray(
                NNS.nns_arma_optim(
                    variable=y, h=h, seasonal_factor=periods_for(y, modulo, mod_only),
                    print_trace=False,
                )["results"],
                float,
            )
        except Exception:
            fc = np.repeat(y[-1], h)
        n2 = np.repeat(y[-1], h)
        ns.append(smape(test, fc)); nm.append(mase(test, fc, y))
        n2s.append(smape(test, n2)); n2m.append(mase(test, n2, y))
    owa = 0.5 * (np.mean(ns) / np.mean(n2s) + np.mean(nm) / np.mean(n2m))
    return owa, float(np.mean(ns))


if __name__ == "__main__":
    for freq, h, mod in [("Hourly", 48, 24), ("Daily", 14, 7), ("Weekly", 13, 12)]:
        for mo in (True, False):
            owa, sm = run(freq, h, mod, mo)
            print(f"{freq:7s} mod={mod:<2} mod_only={str(mo):5s}  OWA={owa:.3f}  sMAPE={sm:.2f}")
