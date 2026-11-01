"""Prefix-power backend (used for >32 targets, integer degree <= 32) at higher degrees on NVDA returns."""
import sys; sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, pandas as pd, nns
from rrun import rcall, arr
x = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)["NVDA"].to_numpy()
t = np.sort(x)
for d in [4, 6, 8, 12]:
    v = np.asarray(nns.lpm(d, t, x)); e = np.concatenate([nns.lpm(d, t[i:i+32], x) for i in range(0, len(t), 32)])
    m = e > 0; rel = np.abs(v - e)[m] / e[m]
    print(f"degree {d}: max rel err {rel.max():.3g}, median rel err {np.median(rel):.3g}, #negative LPM {np.sum(v < 0)}, #rel err >1%: {np.sum(rel > 0.01)} of {m.sum()}")
r = rcall("t <- sort(x); v <- LPM(8, t, x); list(neg = sum(v < 0), worst = min(v))", x=x)
print("R degree 8, sorted targets:", r)
