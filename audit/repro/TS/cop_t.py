import numpy as np, time
from nns.copula import _copula
from nns._native import native_fn
rng=np.random.default_rng(0)
f=native_fn("copula_nd")
for n in [100,1000]:
    m=rng.random((n,3)); t=np.mean(m,0)
    flat=np.ascontiguousarray(np.ravel(m,order='F'))
    t0=time.perf_counter(); [f(flat,n,3,t,True) for _ in range(50)]; print(n,'native %.3f ms'%((time.perf_counter()-t0)/50*1000))
