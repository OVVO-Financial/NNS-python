import numpy as np, time, nns, resource
from nns._native import native_fn
rng = np.random.default_rng(1)
x = np.concatenate([5 + 1e-6*rng.normal(size=1000), [6.0]])
np.savetxt("grav_x.csv", x)
t=time.perf_counter(); g = nns.nns_gravity(x); t1=time.perf_counter()-t
t=time.perf_counter(); ge = native_fn("gravity_exact")(x); t2=time.perf_counter()-t
print(f"nns_gravity (nns::gravity) {g!r} {t1:.2f}s ; gravity_exact (binding, 1e6 bin cap) {ge!r} {t2:.3f}s")
print("peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
print("mode discrete big:", nns.nns_mode(np.array([3e9,3e9,3e9,1,2]), discrete=True))
print("mode(0,0,1,1):", nns.nns_mode(np.array([0,0,1,1.])), nns.nns_mode(np.array([0,0,1,1.]), multi=True))
