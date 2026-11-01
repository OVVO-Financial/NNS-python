import numpy as np, pandas as pd, importlib, nns
import nns._native as nat
r = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv").dropna()
x = r["SPY"].to_numpy()[:1500]; y = r["TLT"].to_numpy()[:1500]
disc = np.random.default_rng(2).integers(0, 5, 400).astype(float); disc2 = np.random.default_rng(3).integers(0, 5, 400).astype(float)
p = pd.read_csv("/home/user/audit/data/prices_daily.csv")["SPY"].dropna().to_numpy()
tz = np.linspace(p.min(), p.max(), 100)
calls = {
 "upm deg8 100 targets (SPY prices)": lambda: nns.upm(8, tz, p),
 "lpm deg2 50 targets": lambda: nns.lpm(2, np.quantile(x, np.linspace(0, 1, 50)), x),
 "nns_gravity": lambda: nns.nns_gravity(x),
 "nns_mode": lambda: nns.nns_mode(x),
 "nns_mode discrete": lambda: nns.nns_mode(disc, discrete=True),
 "nns_dep cont": lambda: nns.nns_dep(x, y)["Dependence"],
 "nns_dep disc": lambda: nns.nns_dep(disc, disc2)["Dependence"],
 "nns_copula": lambda: nns.nns_copula(np.column_stack([x, y])),
 "nns_part quadrants": lambda: np.asarray(nns.nns_part(x, y)["dt"]["quadrant"]),
 "pm_matrix cov": lambda: nns.pm_matrix(1, 1, None, np.column_stack([x, y]), pop_adj=True)["cov.matrix"] if True else None,
 "co_lpm deg(1,2)": lambda: nns.co_lpm(1, x, y, 0.0, 0.0, degree_y=2) if 'degree_y' in nns.co_lpm.__code__.co_varnames else nns.co_lpm(1, x, y, 0.0, 0.0),
 "nns_ss": lambda: nns.nns_ss(x, y),
}
native = {}
for k, f in calls.items():
    try: native[k] = f()
    except Exception as e: native[k] = f"ERR {type(e).__name__}: {e}"
real = nat.native_fn
nat.native_fn = lambda name: None
for mod in list(__import__("sys").modules):
    m = __import__("sys").modules[mod]
    if mod.startswith("nns") and hasattr(m, "native_fn"): m.native_fn = nat.native_fn
for k, f in calls.items():
    try: fb = f()
    except Exception as e: fb = f"ERR {type(e).__name__}: {e}"
    a = native[k]
    try:
        if isinstance(a, dict): a = np.array(list(a.values()), dtype=float); fb = np.array(list(fb.values()), dtype=float)
        if np.asarray(a).dtype.kind in "US": same = np.array_equal(a, fb); print(f"{k:38s} identical={same}")
        else:
            d = np.max(np.abs(np.asarray(a, float) - np.asarray(fb, float)))
            print(f"{k:38s} max|native-fallback| = {d:.3g}")
    except Exception as e:
        print(f"{k:38s} native={str(a)[:60]} fallback={str(fb)[:60]}")
