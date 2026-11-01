import cProfile, pstats, sys, warnings, numpy as np, pandas as pd
import nns
warnings.simplefilter("ignore")
d = pd.read_csv("mv_10000.csv"); X = d.iloc[:, :5].values; y = d.y.values; pe5 = X[:200]
kw = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
cProfile.run("nns.nns_reg(X, y, point_est=pe5, **kw)", "mv.prof")
pstats.Stats("mv.prof").sort_stats("tottime").print_stats(14)
