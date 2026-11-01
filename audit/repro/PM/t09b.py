import time, numpy as np, nns, sys
from nns._native import native_fn
rng = np.random.default_rng(0)
x = rng.standard_t(3, size=10000) * 0.01
tv = np.quantile(x, np.linspace(0.001, 0.999, 1000))
lv = native_fn("lpm_v")
def t(f, r=200):
    f(); t0 = time.perf_counter()
    for _ in range(r): f()
    return (time.perf_counter() - t0) / r * 1e3
print("nns.lpm(2,tv,x)  ms", t(lambda: nns.lpm(2, tv, x)))
print("native lpm_v      ms", t(lambda: lv(2.0, tv, x)))
print("native lpm overload ms", t(lambda: native_fn("lpm")(2.0, tv, x)))
print("isfinite checks   ms", t(lambda: (np.all(np.isfinite(x)), np.all(np.isfinite(tv)))))
print("np.sort(x)        ms", t(lambda: np.sort(x)))
print("lpm(1,tv,x)  ms", t(lambda: nns.lpm(1, tv, x)), " lpm(0,tv,x) ms", t(lambda: nns.lpm(0, tv, x)))
print("lpm(3,tv,x)  ms", t(lambda: nns.lpm(3, tv, x)), " lpm(4,tv,x) ms", t(lambda: nns.lpm(4, tv, x)))
