"""Multivariate NNS.CDF perf/memory: python builds (n, n, d) tensors; native clpm_nd_batch exists but is unused."""
import sys, time, resource, numpy as np
sys.path.insert(0, "/home/user/audit/work/PM")
import nns
from nns._native import native_fn
from rrun import rcall
n = int(sys.argv[1]); d = int(sys.argv[2]); deg = float(sys.argv[3])
rng = np.random.default_rng(1)
B = rng.normal(size=(n, d))
t0 = time.perf_counter(); o = nns.nns_cdf(B, deg); tp = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
batch = native_fn("clpm_nd_batch")
t0 = time.perf_counter()
nb = np.asarray(batch(np.ascontiguousarray(B.ravel(order="F")), n, d, np.ascontiguousarray(B.ravel(order="F")), n, deg, True))
tb = time.perf_counter() - t0
r = rcall("t0 <- proc.time()[[3]]; o <- NNS.CDF(B, deg, plot=FALSE); list(t=proc.time()[[3]] - t0, f=o$Function$CDF)", B=B, deg=deg)
print(f"n={n} d={d} degree={deg}: python nns_cdf {tp:.2f}s (peak RSS {peak:.0f} MB) | native clpm_nd_batch {tb:.3f}s | R {r['t']:.2f}s")
print("  native batch (row-major flatten) == python CDF:", np.allclose(nb, o["Function"]["CDF"]), " R == python:", np.allclose(np.array(r['f']), o["Function"]["CDF"]))
