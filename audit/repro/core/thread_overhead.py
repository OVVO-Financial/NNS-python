import numpy as np, time, os
from nns._native import native_fn as nf
x=np.random.default_rng(0).normal(size=200); y=x[::-1].copy()
tx=np.array([0.0,0.1]); ty=np.array([0.0,0.1]); f=nf('co_lpm_v'); s=nf('co_lpm'); N=20000
t=time.perf_counter(); [f(1.0,1.0,x,y,tx,ty) for _ in range(N)]; a=(time.perf_counter()-t)/N
t=time.perf_counter(); [(s(1.0,1.0,x,y,0.0,0.0), s(1.0,1.0,x,y,0.1,0.1)) for _ in range(N)]; b=(time.perf_counter()-t)/N
print(f'co_lpm_v(2 targets, n=200): {a*1e6:.1f} us/call ; 2x scalar co_lpm: {b*1e6:.1f} us/call ; cores={os.cpu_count()}')
