import time, numpy as np, pandas as pd, warnings; warnings.simplefilter("ignore")
from nns import nns_boost, nns_stack
X=pd.read_csv("X_10000_5.csv").to_numpy(); y=pd.read_csv("y_10000_5.csv").iloc[:,0].to_numpy(); nns_stack(X[:50],y[:50])
t=time.perf_counter(); b=nns_boost(X[:8000],y[:8000],X[8000:],epochs=20,learner_trials=20); el=time.perf_counter()-t
print(f"PY boost n=10000 p=5 (20/20) secs={el:.2f} maxdiff_vs_R={np.abs(np.loadtxt('r_b10k.txt')-b['results']).max():.3g}")
