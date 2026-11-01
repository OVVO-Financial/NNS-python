"""Vectorised (>32 targets) LPM/UPM use a binomial prefix-power expansion; check its accuracy
against the direct scan (<=32 targets path) on real price-level and return data, Python and R."""
import sys
sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, pandas as pd
import nns
from rrun import rcall, arr

px = pd.read_csv("/home/user/audit/data/prices_daily.csv", index_col=0)
ret = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)

def direct(fn, d, t, x):
    # <=32 targets -> exact direct scan path
    return np.concatenate([np.atleast_1d(fn(d, t[i:i + 32], x)) for i in range(0, len(t), 32)])

def report(label, x, ds=(1, 2, 3, 4)):
    t = np.sort(x)  # the NNS.CDF / LPM.ratio(degree, sort(x), x) use case
    for d in ds:
        for nm, fn in (("lpm", nns.lpm), ("upm", nns.upm)):
            v = np.asarray(fn(d, t, x)); e = direct(fn, d, t, x)
            mask = e > 0
            rel = np.abs(v - e)[mask] / e[mask]
            zero_wrong = np.sum((e == 0) & (v != 0))
            print(f"{label:10s} d={d} {nm}: max rel err {rel.max():.3g}  #rel>1e-6: {np.sum(rel>1e-6)}  "
                  f"#neg: {np.sum(v<0)}  #(true 0, got !=0): {zero_wrong}")
        rv = np.asarray(nns.lpm_ratio(d, t, x))
        re_ = direct(nns.lpm_ratio, d, t, x)
        print(f"{label:10s} d={d} lpm_ratio: max abs err {np.nanmax(np.abs(rv-re_)):.3g}, outside [0,1]: {np.sum((rv<0)|(rv>1))}, non-monotone steps: {np.sum(np.diff(rv) < -1e-15)}")

spy = px["SPY"].dropna().to_numpy()
btc_px = px["BTC-USD"].dropna().to_numpy()
nvda = ret["NVDA"].to_numpy()
report("SPY price", spy)
report("BTC price", btc_px)
report("NVDA ret", nvda)

# R: same comparison (R 13.3 has the same prefix backend for >32 targets)
r = rcall("""
t <- sort(x)
dir <- function(f, d) unlist(lapply(split(t, ceiling(seq_along(t)/32)), function(tt) f(d, tt, x)))
lapply(c(2,3,4), function(d) { v <- UPM(d, t, x); e <- dir(UPM, d); m <- e > 0;
  list(maxrel = max(abs(v-e)[m]/e[m]), nneg = sum(v<0), ratio_out = sum(LPM.ratio(d,t,x) < 0 | LPM.ratio(d,t,x) > 1)) })
""", x=spy)
print("R SPY price UPM d=2,3,4:", r)
# demonstrate the R path discontinuity at 32 vs 33 targets
t33 = np.sort(spy)[-33:]
r2 = rcall("list(a = UPM(4, t[2:33], x), b = UPM(4, t, x)[2:33])", x=spy, t=t33)
a, b = arr(r2["a"]), arr(r2["b"])
print("R UPM(4, top-32 targets) direct vs same targets inside a 33-target call: max rel diff",
      np.max(np.abs(a - b) / np.abs(a)))
print("example: target", t33[-5], "direct", a[-5], "via 33-target call", b[-5])
print("python same:", nns.upm(4, t33[1:], spy)[-5], np.asarray(nns.upm(4, t33, spy))[1:][-5])
