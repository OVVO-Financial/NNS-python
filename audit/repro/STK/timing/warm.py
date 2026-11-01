import time; t=time.perf_counter()
import numpy as np, pandas as pd, warnings; warnings.simplefilter("ignore")
from nns import nns_stack; print("import nns secs %.2f"%(time.perf_counter()-t))
t=time.perf_counter(); nns_stack(np.random.rand(40,3), np.random.rand(40)); print("first tiny call secs %.2f"%(time.perf_counter()-t))
for n,p in [(500,5),(500,10),(2000,5),(2000,10),(10000,5)]:
    X=pd.read_csv(f"X_{n}_{p}.csv").to_numpy(); y=pd.read_csv(f"y_{n}_{p}.csv").iloc[:,0].to_numpy(); m=int(0.8*n)
    ts=[]
    for _ in range(2):
        t=time.perf_counter(); nns_stack(X[:m],y[:m],X[m:]); ts.append(time.perf_counter()-t)
    print(f"warm stack n={n} p={p}: {min(ts):.2f}s")
