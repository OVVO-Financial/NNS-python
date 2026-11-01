import sys, warnings
sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, pandas as pd, nns
from rrun import rcall, arr
ret = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)
v = ret["NVDA"].to_numpy()[:1000]
print("ties in NVDA[:1000]:", len(v) - len(np.unique(v)))
try:
    r = rcall("o <- NNS.CDF(v, degree=0, type='hazard', plot=FALSE); o$Function[[2]][1:5]", v=v)
    print("R:", r)
except RuntimeError as e:
    print("R error:", str(e).strip().splitlines()[-2:])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        o = nns.nns_cdf(v, degree=0, type="hazard")
        h = o["Function"]["h(x)"]
        print("python hazard: n nan", np.sum(np.isnan(h)), "n inf", np.sum(np.isinf(h)), "first", h[:5])
    except Exception as e:
        print("python error:", type(e).__name__, e)
    print("warnings:", [str(x.message)[:80] for x in w][:3])
# simple ties
x = np.array([1., 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
try:
    print("R ties:", rcall("NNS.CDF(x, type='hazard', plot=FALSE)$Function[[2]]", x=x))
except RuntimeError as e:
    print("R error ties:", str(e).strip().splitlines()[-2:])
try:
    print("py ties:", nns.nns_cdf(x, type="hazard")["Function"]["h(x)"])
except Exception as e:
    print("python error ties:", type(e).__name__, e)
