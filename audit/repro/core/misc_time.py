import numpy as np, pandas as pd, time, nns
from nns.stochastic_dominance import sd_efficient_set
r = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv").dropna().drop(columns=["Date"]).to_numpy()
p = pd.read_csv("/home/user/audit/data/prices_daily.csv")["SPY"].dropna().to_numpy()
def T(lbl, f, reps=1):
    f(); t=time.perf_counter()
    for _ in range(reps): out=f()
    print(f"{lbl}: {(time.perf_counter()-t)/reps:.4f}s", flush=True); return out
T("py nns_dep matrix 11 assets n=2000", lambda: __import__("nns.dependence").dependence.nns_dep_matrix(r[:2000]) if hasattr(__import__("nns.dependence").dependence,"nns_dep_matrix") else None)
T("py nns_part mean noise n=4000", lambda: nns.nns_part(r[:4000,0], r[:4000,1], noise_reduction="mean"))
T("py nns_part off n=4000", lambda: nns.nns_part(r[:4000,0], r[:4000,1]))
T("py nns_gravity n=1e5", lambda: nns.nns_gravity(np.tile(r[:,0], 30)[:100000]))
T("py nns_mode n=1e5", lambda: nns.nns_mode(np.tile(r[:,0], 30)[:100000]))
