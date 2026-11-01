import numpy as np, time, json, subprocess
from nns._reg_engine import _mreg_predict_path, _mreg_predict
rng = np.random.default_rng(7)
n, p, m, kmax = 1500, 3, 600, 30
rpm_x = rng.normal(size=(n, p)); yhat = rpm_x @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=n)
xt = rng.normal(size=(m, p))
mins, maxs = rpm_x.min(0), rpm_x.max(0)
np.savetxt("mp_rpm.csv", rpm_x, delimiter=","); np.savetxt("mp_y.csv", yhat, delimiter=","); np.savetxt("mp_xt.csv", xt, delimiter=",")
t=time.perf_counter(); P = _mreg_predict_path(xt, rpm_x, yhat, kmax, "NNS", mins, maxs); tp=time.perf_counter()-t
t=time.perf_counter(); S = _mreg_predict(xt, rpm_x, yhat, 12, "NNS", mins, maxs, False); ts=time.perf_counter()-t
np.savetxt("mp_py_path.csv", P, delimiter=","); np.savetxt("mp_py_single.csv", S, delimiter=",")
print(f"python path {tp:.3f}s single(k=12) {ts:.3f}s")
