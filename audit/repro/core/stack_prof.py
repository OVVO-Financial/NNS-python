import numpy as np, pandas as pd, cProfile, pstats, nns
b = pd.read_csv("/home/user/audit/data/boston.csv")
y = b["medv"].to_numpy(float); X = b.drop(columns=["medv"]).to_numpy(float)
nns.nns_stack(X[:400], y[:400], ivs_test=X[400:], method=1, folds=1, ncores=1)
cProfile.run("nns.nns_stack(X[:400], y[:400], ivs_test=X[400:], method=1, folds=1, ncores=1)", "stack.prof")
pstats.Stats("stack.prof").sort_stats("cumulative").print_stats(18)
