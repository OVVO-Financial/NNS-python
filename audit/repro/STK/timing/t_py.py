import sys, time, numpy as np, pandas as pd, warnings; warnings.simplefilter("ignore")
from nns import nns_stack, nns_boost
n,p=sys.argv[1],sys.argv[2]; X=pd.read_csv(f"X_{n}_{p}.csv").to_numpy(); y=pd.read_csv(f"y_{n}_{p}.csv").iloc[:,0].to_numpy()
m=int(np.floor(0.8*len(y))); t0=time.perf_counter(); r=nns_stack(X[:m],y[:m],X[m:]); el=time.perf_counter()-t0
rs=np.loadtxt(f"r_stack_{n}_{p}.txt") if __import__("os").path.exists(f"r_stack_{n}_{p}.txt") else None
print(f"PY n={n} p={p} stack secs={el:.2f} nbest={r['NNS.reg.n.best']} w={r['weights']['reg']:.2f} maxdiff_vs_R={np.abs(rs-r['stack']).max() if rs is not None else 'NA'}")
if len(sys.argv)>3:
    t0=time.perf_counter(); b=nns_boost(X[:m],y[:m],X[m:],epochs=50,learner_trials=50); el=time.perf_counter()-t0
    rb=np.loadtxt(f"r_boost_{n}_{p}.txt"); print(f"PY boost(50/50) secs={el:.2f} maxdiff_vs_R={np.abs(rb-b['results']).max():.3g}")
