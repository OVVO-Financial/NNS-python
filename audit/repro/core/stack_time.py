import numpy as np, pandas as pd, time, nns
b = pd.read_csv("/home/user/audit/data/boston.csv")
y = b["medv"].to_numpy(float); X = b.drop(columns=["medv"]).to_numpy(float)
tr = np.arange(400); te = np.arange(400, 506)
for method in (1, 2):
    t=time.perf_counter()
    r = nns.nns_stack(X[tr], y[tr], ivs_test=X[te], method=method, folds=1, ncores=1)
    print(f"python nns_stack method={method}: {time.perf_counter()-t:.2f}s")
