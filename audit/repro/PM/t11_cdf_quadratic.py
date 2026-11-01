"""Univariate nns_cdf falls back to an (n x n) dense matrix for fractional degree, degree>=4, or any NaN."""
import sys, time, resource, subprocess, numpy as np
sys.path.insert(0, "/home/user/audit/work/PM")
import nns
from rrun import rcall
n = int(sys.argv[1]); deg = float(sys.argv[2]); nan = sys.argv[3] == "nan"
rng = np.random.default_rng(1)
x = rng.standard_t(3, size=n)
if nan: x[5] = np.nan
t0 = time.perf_counter(); o = nns.nns_cdf(x, deg); tp = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
xr = [v if np.isfinite(v) else "NA" for v in x]
r = rcall("t0 <- proc.time()[[3]]; o <- NNS.CDF(x, deg, plot=FALSE); list(t=proc.time()[[3]] - t0, f=o$Function[[2]])", x=xr, deg=deg)
f = np.array(r["f"], dtype=float)
print(f"n={n} degree={deg} nan={nan}: python {tp:.2f}s peak RSS {peak:.0f} MB | R {r['t']:.3f}s | max|diff| {np.nanmax(np.abs(f - o['Function']['CDF'])):.2g}")
