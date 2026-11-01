"""lpm_var has no vector-percentile support: a Python loop over 100 percentiles vs one vectorised R call."""
import sys, time; sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, nns
from rrun import rcall
rng = np.random.default_rng(0)
x = rng.standard_t(3, size=100_000) * 0.01
ps = np.linspace(0.005, 0.995, 100)
t0 = time.perf_counter(); v = np.array([nns.lpm_var(p, 1, x) for p in ps]); tp = time.perf_counter() - t0
r = rcall("t0 <- proc.time()[[3]]; v <- LPM.VaR(ps, 1, x); list(t = proc.time()[[3]] - t0, v = v)", x=x, ps=ps)
print(f"100 percentiles, n=1e5, degree 1: python loop {tp*1e3:.0f} ms, R vectorised {r['t']*1e3:.0f} ms, max|diff| {np.max(np.abs(v - np.array(r['v']))):.2g}")
