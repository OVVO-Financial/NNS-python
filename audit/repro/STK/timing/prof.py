import sys, cProfile, pstats, time, numpy as np, pandas as pd, warnings; warnings.simplefilter("ignore")
from nns import nns_stack, nns_boost
n,p,what=sys.argv[1],sys.argv[2],sys.argv[3]; X=pd.read_csv(f"X_{n}_{p}.csv").to_numpy(); y=pd.read_csv(f"y_{n}_{p}.csv").iloc[:,0].to_numpy(); m=int(0.8*len(y))
nns_stack(X[:50],y[:50],X[50:60])  # warm
f=(lambda: nns_stack(X[:m],y[:m],X[m:])) if what=="stack" else (lambda: nns_boost(X[:m],y[:m],X[m:],epochs=50,learner_trials=50))
t0=time.perf_counter(); f(); print("warm secs", time.perf_counter()-t0)
pr=cProfile.Profile(); pr.enable(); f(); pr.disable()
st=pstats.Stats(pr); st.sort_stats("tottime").print_stats(18)
