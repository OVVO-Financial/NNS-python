import numpy as np, pandas as pd, time, nns, cProfile, pstats
d=pd.read_csv("/home/user/audit/work/DEP/time_100000.csv"); x=d.x.values; y=d.y.values
t=time.perf_counter(); r=nns.nns_causation(x,y,tau=0); print("py caus0 n=1e5", time.perf_counter()-t, r)
pr=cProfile.Profile(); pr.enable(); nns.nns_causation(x[:20000],y[:20000],tau=1); pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(12)
