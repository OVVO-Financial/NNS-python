import time, warnings, numpy as np, pandas as pd
import nns
warnings.simplefilter("ignore")
def tm(f, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
    return best
pe = np.linspace(-3.5, 3.5, 100)
for n in (1000, 10000, 100000):
    d = pd.read_csv(f"uni_{n}.csv"); x, y = d.x.values, d.y.values
    print(f"uni n={n} default: {tm(lambda: nns.nns_reg(x, y, point_est=pe)):.3f}s")
    print(f"uni n={n} smooth: {tm(lambda: nns.nns_reg(x, y, point_est=pe, smooth=True)):.3f}s")
    if n <= 10000: print(f"uni n={n} order=max: {tm(lambda: nns.nns_reg(x, y, point_est=pe, order='max')):.3f}s")
    print(f"uni n={n} ci: {tm(lambda: nns.nns_reg(x, y, point_est=pe, confidence_interval=.95)):.3f}s")
d = pd.read_csv("mv_10000.csv"); X = d.iloc[:, :5].values; y = d.y.values; pe5 = X[:200]
print(f"mv n=1e4 p=5 default: {tm(lambda: nns.nns_reg(X, y, point_est=pe5), 2):.3f}s")
print(f"mv n=1e4 p=5 n.best=1: {tm(lambda: nns.nns_reg(X, y, point_est=pe5, n_best=1), 2):.3f}s")
print(f"mv n=1e4 p=5 order=max: {tm(lambda: nns.nns_reg(X, y, point_est=pe5, order='max'), 2):.3f}s")
for m in ("cor", "NNS.dep", "all"):
    print(f"mv n=1e4 p=5 dimred {m}: {tm(lambda: nns.nns_reg(X, y, point_est=pe5, dim_red_method=m), 2):.3f}s")
r = nns.nns_reg(X, y, point_est=pe5); print("mv R2", r["R2"], "n.best", r["n.best"], "RPM rows", len(r["RPM"]["y.hat"]))
