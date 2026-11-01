import numpy as np, time, os
from nns._native import native_fn
from nns.dependence import nns_dep
rng=np.random.default_rng(0)
f=native_fn("copula_nd")
for n in [50,125,1000]:
    m=rng.random((n,3)); t=np.mean(m,0); flat=np.ascontiguousarray(np.ravel(m,order='F'))
    best=min((lambda: (lambda t0: ([f(flat,n,3,t,True) for _ in range(200)], time.perf_counter()-t0)[1])(time.perf_counter()))() for _ in range(5))
    bd=min((lambda: (lambda t0: ([nns_dep(m[:,0],m[:,2],asym=True) for _ in range(200)], time.perf_counter()-t0)[1])(time.perf_counter()))() for _ in range(3))
    print(n,'copula native best %.3f ms/call'%(best/200*1000), ' nns_dep %.3f ms/call'%(bd/200*1000))
