"""LPM.VaR / UPM.VaR: inverse-of-ratio checks and A/B vs R."""
import sys, warnings
sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, pandas as pd
import nns
from rrun import rcall, arr

ret = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)
px = pd.read_csv("/home/user/audit/data/prices_daily.csv", index_col=0)
btc = px["BTC-USD"].dropna().pct_change().dropna().to_numpy()
rng = np.random.default_rng(3)
cases = {"nvda": ret["NVDA"].to_numpy(), "btc": btc, "norm50": rng.normal(size=50),
         "ties": np.array([1., 1, 2, 2, 2, 3, 5, 5, 9]), "two": np.array([0., 1.]), "spy_px": px["SPY"].dropna().to_numpy()}
ps = [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
print("== inverse property: lpm_ratio(d, lpm_var(p, d, x), x) == p")
for name, x in cases.items():
    for d in [0, 1, 2, 3, 1.5]:
        errs = []
        for p in ps:
            v = nns.lpm_var(p, d, x)
            back = nns.lpm_ratio(d, v, x)
            u = nns.upm_var(p, d, x)
            backu = nns.upm_ratio(d, u, x)
            if d > 0 and 0 < p < 1:
                errs.append(max(abs(back - p), abs(backu - p)))
        if errs and max(errs) > 1e-6:
            print(f"{name:7s} d={d}: max |ratio(VaR(p)) - p| = {max(errs):.3g}")
print("(only failures printed)")

print("== A/B vs R")
for name, x in cases.items():
    r = rcall("lapply(ds, function(d) list(l=unname(sapply(ps, function(p) LPM.VaR(p, d, x))), u=unname(sapply(ps, function(p) UPM.VaR(p, d, x)))))",
              x=x, ds=[0, 1, 2, 3, 4, 1.5, 0.5], ps=ps)
    for d, rr in zip([0, 1, 2, 3, 4, 1.5, 0.5], r):
        pl = np.array([nns.lpm_var(p, d, x) for p in ps]); pu = np.array([nns.upm_var(p, d, x) for p in ps])
        dl = np.max(np.abs(pl - arr(rr["l"]))); du = np.max(np.abs(pu - arr(rr["u"])))
        scale = np.ptp(x)
        if max(dl, du) > 1e-9 * scale:
            print(f"MISMATCH {name:7s} d={d}: max|dL|={dl:.3g} max|dU|={du:.3g} (range {scale:.3g})")
            if d in (1.5, 0.5):
                print("   py L:", np.round(pl, 6)); print("   R  L:", np.round(arr(rr['l']), 6))
print("== degree-0 tie/quantile convention: LPM.VaR(p,0,x) == quantile type 7, but LPM.ratio(0, VaR, x) >= p?")
x = cases["ties"]
for p in [0.05, 0.2, 0.25, 0.5, 0.9]:
    v = nns.lpm_var(p, 0, x)
    print(p, v, "LPM.ratio(0,v)=", nns.lpm_ratio(0, v, x))
print("== optimize-based fractional degree: accuracy of ratio inversion (R and Python)")
x = cases["nvda"]
for d in [0.5, 1.5, 2.5]:
    for p in [0.01, 0.05, 0.5]:
        v = nns.lpm_var(p, d, x)
        rv = rcall("LPM.VaR(p, d, x)", p=p, d=d, x=x)
        print(f"d={d} p={p}: py var={v:.6g} ratio={nns.lpm_ratio(d, v, x):.6g} | R var={rv:.6g} ratio={nns.lpm_ratio(d, rv, x):.6g}")
