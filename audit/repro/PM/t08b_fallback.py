"""Pure-Python fallbacks (used when native kernel missing) vs native kernel."""
import sys
import numpy as np, pandas as pd, nns
ct = sys.modules["nns.central_tendencies"] if "nns.central_tendencies" in sys.modules else __import__("nns.central_tendencies", fromlist=["x"])
core = sys.modules["nns.core"]
ret = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)
rng = np.random.default_rng(2)
cases = {"norm": rng.normal(size=500), "bimodal": np.concatenate([rng.normal(-3, 1, 300), rng.normal(3, 1, 200)]),
         "nvda": ret["NVDA"].to_numpy(), "ints": rng.integers(0, 10, 200).astype(float)}
nat = {k: (ct.nns_mode(x, False, False), ct.nns_mode(x, False, True), ct.nns_mode(x, True, False), ct.nns_gravity(x)) for k, x in cases.items()}
orig = ct.native_fn
ct.native_fn = lambda name: None
for k, x in cases.items():
    fb = (ct.nns_mode(x, False, False), ct.nns_mode(x, False, True), ct.nns_mode(x, True, False), ct.nns_gravity(x))
    for lab, a, b in zip(["mode", "mode multi", "mode discrete", "gravity"], nat[k], fb):
        a, b = np.atleast_1d(a), np.atleast_1d(b)
        same = a.shape == b.shape and np.allclose(a, b)
        print(f"{k:8s} {lab:14s} native={np.round(a[:4],5)}{'...' if a.size>4 else ''} (n={a.size}) fallback={np.round(b[:4],5)} (n={b.size}) {'OK' if same else 'DIFF'}")
ct.native_fn = orig
