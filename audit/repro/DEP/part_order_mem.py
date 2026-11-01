import numpy as np, nns, time, tracemalloc
rng=np.random.default_rng(0); x=rng.normal(size=1000); y=rng.normal(size=1000)
for order in [None, 10, 2000, 20000]:
    tracemalloc.start(); t=time.perf_counter()
    r=nns.nns_part(x,y,order=order,noise_reduction="mean")
    dt=time.perf_counter()-t; cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"order={order}: depth={r['order']} time={dt:.3f}s peak_mem={peak/1e6:.1f}MB")
