import time, numpy as np
from nns._reg_engine import _mreg_distances
rng = np.random.default_rng(0)
rpm = rng.standard_normal((9711, 5)); xt = rng.standard_normal((412, 5))
mins, maxs = rpm.min(0), rpm.max(0)
def dist2d(rpm_x, xtest, mins, maxs):
    rng_ = maxs - mins
    out = np.zeros((xtest.shape[0], rpm_x.shape[0])); tmp = np.empty_like(out); sq = np.empty_like(out)
    for j in range(rpm_x.shape[1]):
        np.subtract(xtest[:, j, None], rpm_x[None, :, j], out=tmp); tmp /= rng_[j]
        np.multiply(tmp, tmp, out=sq); np.abs(tmp, out=tmp); tmp += sq; out += tmp
    return out
for rep in range(3):
    t = time.perf_counter(); a = _mreg_distances(rpm, xt, "NNS", mins, maxs); t1 = time.perf_counter() - t
    t = time.perf_counter(); b = dist2d(rpm, xt, mins, maxs); t2 = time.perf_counter() - t
    print(f"current {t1:.3f}s  column-loop {t2:.3f}s  identical={np.array_equal(a, b)} maxdiff={np.abs(a-b).max():.1e}")
