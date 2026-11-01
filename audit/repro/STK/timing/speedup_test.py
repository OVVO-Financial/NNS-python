import time, numpy as np, pandas as pd, warnings; warnings.simplefilter("ignore")
import nns._reg_engine as E
from nns import nns_boost, nns_stack
X=pd.read_csv("X_2000_5.csv").to_numpy(); y=pd.read_csv("y_2000_5.csv").iloc[:,0].to_numpy(); m=1600
nns_stack(X[:50],y[:50],X[50:60])
t=time.perf_counter(); b0=nns_boost(X[:m],y[:m],X[m:],epochs=50,learner_trials=50); t0=time.perf_counter()-t
orig_build=E._mreg_build_rpm; orig_dist=E._mreg_distances
def fast_dist(rpm_x, xtest, dist, mins, maxs):
    if dist not in ("NNS","L1","L2"): return orig_dist(rpm_x, xtest, dist, mins, maxs)
    ranges=maxs-mins; active=np.flatnonzero(np.isfinite(ranges)&(ranges>0))
    acc=np.zeros((xtest.shape[0], rpm_x.shape[0]))
    for j in active:
        z=(xtest[:,j][:,None]-rpm_x[:,j][None,:])/ranges[j]
        acc += (np.abs(z)+z*z) if dist=="NNS" else (np.abs(z) if dist=="L1" else z*z)
    return np.sqrt(acc) if dist=="L2" else acc
def fast_build(x, yv, ids, noise_reduction, is_class):
    order=np.argsort(ids, kind="stable")
    if np.unique(ids).size==ids.size: return x[order].astype(np.float64), yv[order].astype(np.float64)
    ids_s=ids[order]; xs=x[order]; ys=yv[order]
    _, starts=np.unique(ids_s, return_index=True); starts=np.sort(starts); ends=np.append(starts[1:], ids_s.size)
    sizes=ends-starts
    rows=xs[starts].astype(np.float64).copy(); yh=ys[starts].astype(np.float64).copy()
    for g in np.flatnonzero(sizes>1):
        s,e=starts[g],ends[g]
        rows[g]=[E._reduce_value(xs[s:e,j], noise_reduction, False) for j in range(x.shape[1])]
        yh[g]=E._reduce_value(ys[s:e], noise_reduction, is_class)
    return rows, yh
E._mreg_build_rpm=fast_build; E._mreg_distances=fast_dist
t=time.perf_counter(); b1=nns_boost(X[:m],y[:m],X[m:],epochs=50,learner_trials=50); t1=time.perf_counter()-t
print(f"boost n=2000 p=5: current {t0:.2f}s, with column-accum distances + singleton RPM shortcut {t1:.2f}s; results identical: {np.array_equal(b0['results'], b1['results'])}")
import cProfile, pstats
pr=cProfile.Profile(); pr.enable(); nns_boost(X[:m],y[:m],X[m:],epochs=50,learner_trials=50); pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(12)
