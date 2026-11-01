"""lpm/upm with any NaN in x silently drop to an (m x n) dense numpy fallback."""
import time, resource, numpy as np, nns
rng = np.random.default_rng(0)
x = rng.normal(size=100_000); tv = np.linspace(-2, 2, 1000)
t0 = time.perf_counter(); a = nns.lpm(1, tv, x); t1 = time.perf_counter() - t0
x[10] = np.nan
t0 = time.perf_counter(); b = nns.lpm(1, tv, x); t2 = time.perf_counter() - t0
t0 = time.perf_counter(); c = nns.lpm(0, tv, x); t3 = time.perf_counter() - t0
print(f"finite: {t1*1e3:.1f} ms; one NaN, degree 1: {t2*1e3:.1f} ms (result all NaN: {np.all(np.isnan(b))}); degree 0: {t3*1e3:.1f} ms")
print("peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
print("degree-0 with NaN: lpm(0,0,x)+upm(0,0,x) =", nns.lpm(0, 0.0, x) + nns.upm(0, 0.0, x), " (silently != 1; NaN counted in n)")
print("lpm_ratio(0, 0, x) =", nns.lpm_ratio(0, 0.0, x), " lpm_ratio(1, 0, x) =", nns.lpm_ratio(1, 0.0, x))
