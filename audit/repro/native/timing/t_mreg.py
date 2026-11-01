import sys, time, io, contextlib, importlib.util, os
import numpy as np, pandas as pd, nns
import matplotlib; matplotlib.use("Agg")
def best(fn, k=3):
    ts = []
    for _ in range(k):
        t = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t)
    return min(ts), r
rng = np.random.default_rng(42)
X = rng.normal(size=(10_000, 5)); y = X[:, 0] + np.sin(X[:, 1]) + X[:, 2] * X[:, 3] + 0.1 * rng.normal(size=10_000)
pe = rng.normal(size=(200, 5))
t, r = best(lambda: nns.nns_m_reg(X, y, point_est=pe), 2)
print(f"nns_m_reg n=1e4 p=5 (point_est 200): {t:.2f}s  R2={r['R2']!r}")
np.save(sys.argv[1] + "_mreg_yhat.npy", np.r_[r["Fitted.xy"]["y.hat"], r["Point.est"]])
b = pd.read_csv("/home/user/audit/data/boston.csv")
yb = b["medv"].to_numpy(float); Xb = b.drop(columns=["medv"]).to_numpy(float)
tr = np.arange(400); te = np.arange(400, 506)
t, r = best(lambda: nns.nns_stack(Xb[tr], yb[tr], ivs_test=Xb[te], method=1, folds=1, ncores=1))
print(f"nns_stack Boston method=1 folds=1: {t:.2f}s  OBJfn.reg={r['OBJfn.reg']!r}")
np.save(sys.argv[1] + "_stack.npy", np.asarray(r["reg"], float))
t, r = best(lambda: nns.nns_stack(Xb[tr], yb[tr], ivs_test=Xb[te], method=1, ncores=1), 1)
print(f"nns_stack Boston method=1 folds=5: {t:.2f}s  OBJfn.reg={r['OBJfn.reg']!r}")
spec = importlib.util.spec_from_file_location("v07", "/home/user/wt/native/examples/vignettes/07_clustering_and_regression.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
buf = io.StringIO()
def run07():
    with contextlib.redirect_stdout(buf): mod.main()
t, _ = best(run07, 1)
print(f"vignette 07 main(): {t:.2f}s")
open(sys.argv[1] + "_v07.txt", "w").write(buf.getvalue())
