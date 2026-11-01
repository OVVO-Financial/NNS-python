import numpy as np, time, resource, sys, nns
k=int(sys.argv[1]); n=int(sys.argv[2]); d=int(sys.argv[3])
rng=np.random.default_rng(0); X=rng.standard_t(4,(n,k))*0.01+rng.normal(0.0003,0.0003,k)
np.save(f"big_{k}_{n}.npy", X) if d==1 else None
t=time.perf_counter(); e=nns.sd_efficient_set(X,d); el=time.perf_counter()-t
print(f"py k={k} n={n} deg={d}: {el*1000:.0f} ms, |eff|={len(e)}, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
