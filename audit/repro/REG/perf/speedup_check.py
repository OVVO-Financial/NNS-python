import sys, time, warnings, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/audit/work/REG/perf"); warnings.simplefilter("ignore")
import nns, speedup_patch
d = pd.read_csv("mv_10000.csv"); X = d.iloc[:, :5].values; y = d.y.values; pe5 = X[:200]
b = pd.read_csv("/home/user/audit/data/boston.csv"); bx = b.drop(columns="medv").values; by = b.medv.values
cases = [("mv1e4 default", X, y, {}), ("mv1e4 order=max", X, y, {"order": "max"}), ("boston nr=median", bx, by, {"noise_reduction": "median"}),
         ("boston FACTOR", bx, by, {"dist": "FACTOR"}), ("boston L2", bx, by, {"dist": "L2"}), ("boston L1 o2", bx, by, {"dist": "L1", "order": 2})]
for name, XX, yy, kw in cases:
    t = time.perf_counter(); a = nns.nns_reg(XX, yy, point_est=XX[:50], **kw); t1 = time.perf_counter() - t
    speedup_patch.install()
    t = time.perf_counter(); c = nns.nns_reg(XX, yy, point_est=XX[:50], **kw); t2 = time.perf_counter() - t
    import importlib, nns._reg_engine as eng; eng._mreg_distances = speedup_patch._orig_dist; eng._mreg_build_rpm = speedup_patch._orig_rpm
    same = np.array_equal(a["Fitted.xy"]["y.hat"], c["Fitted.xy"]["y.hat"]) and np.array_equal(a["Point.est"], c["Point.est"]) and all(np.array_equal(a["RPM"][k], c["RPM"][k]) for k in a["RPM"])
    print(f"{name:18s} current {t1:6.2f}s  patched {t2:6.2f}s  identical={same}")
