"""NNS.moments vs scipy, R; classical helpers."""
import sys, warnings
sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, pandas as pd, scipy.stats as st
import nns
from rrun import rcall, arr

ret = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)
px = pd.read_csv("/home/user/audit/data/prices_daily.csv", index_col=0)
rng = np.random.default_rng(5)
cases = {"nvda": ret["NVDA"].to_numpy(), "uso": ret["USO"].to_numpy(), "spy_px": px["SPY"].dropna().to_numpy(),
         "small5": np.array([1., 2, 4, 8, 16]), "lvl1e8": 1e8 + rng.normal(size=500)}
for name, x in cases.items():
    for pop in [True, False]:
        m = nns.nns_moments(x, population=pop)
        r = rcall("NNS.moments(x, population=p)", x=x, p=pop)
        sk = st.skew(x, bias=pop); ku = st.kurtosis(x, bias=pop)
        var = np.var(x, ddof=0 if pop else 1)
        print(f"{name:7s} pop={pop!s:5s} py: var={m['variance']:.10g} skew={m['skewness']:.10g} kurt={m['kurtosis']:.10g}")
        print(f"{'':7s} {'':9s} R : var={r['variance']:.10g} skew={r['skewness']:.10g} kurt={r['kurtosis']:.10g}")
        print(f"{'':7s} {'':9s} sp: var={var:.10g} skew={sk:.10g} kurt={ku:.10g}  mean py-R={m['mean']-r['mean']:.3g}")
print("== classical helpers")
x = cases["nvda"]
print("var_pm ddof=1:", nns.var_pm(x, ddof=1), np.var(x, ddof=1))
print("skew_pm vs scipy:", nns.skew_pm(x), st.skew(x))
print("kurt_pm vs scipy:", nns.kurt_pm(x), st.kurtosis(x))
print("mean_pm level data:", nns.mean_pm(cases["lvl1e8"]) - np.mean(cases["lvl1e8"]))
print("var_pm level data 1e8+N(0,1) (true ~1):", nns.var_pm(cases["lvl1e8"]), np.var(cases["lvl1e8"]))
for f in ["var_pm", "skew_pm", "kurt_pm"]:
    try:
        print(f, "const:", getattr(nns, f)(np.ones(4)))
    except Exception as e:
        print(f, "const RAISES", type(e).__name__, e)
print("ecdf_pm:", nns.ecdf_pm(np.array([3., 1, 2, 2])))
