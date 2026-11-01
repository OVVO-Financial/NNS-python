"""pm_matrix: native fused loop vs the module's own numpy/BLAS fallback (same results)."""
import sys, time, numpy as np, nns
pmm = sys.modules["nns.pm_matrix"]
rng = np.random.default_rng(0)
for shape in [(1_000_000, 11), (10_000, 100), (100_000, 50)]:
    X = rng.normal(size=shape)
    t0 = time.perf_counter(); a = nns.pm_matrix(1, 1, None, X, True); tn = time.perf_counter() - t0
    orig = pmm.native_fn; pmm.native_fn = lambda name: None
    t0 = time.perf_counter(); b = nns.pm_matrix(1, 1, None, X, True); tb = time.perf_counter() - t0
    pmm.native_fn = orig
    same = max(float(np.max(np.abs(a[k] - b[k]) / np.max(np.abs(a[k])))) for k in a)
    print(f"{shape}: native {tn*1e3:.0f} ms, numpy/BLAS fallback {tb*1e3:.0f} ms, max rel diff={same:.2g}")
