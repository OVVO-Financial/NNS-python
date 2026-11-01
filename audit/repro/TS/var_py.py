import sys, json, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from nns import nns_var
k,h,tau,drm = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
P=pd.read_csv('/home/user/audit/work/TS/data/panel_monthly.csv').iloc[:, :k]
R=json.load(open(sys.argv[5]))
t0=time.time(); r=nns_var(P, h=h, tau=tau, dim_red_method=drm, status=False); tp=time.time()-t0
for key in ['univariate','multivariate','ensemble']:
    a=np.array(R[key]); b=np.asarray(r[key])
    print(key, 'maxabs diff', np.max(np.abs(a-b)), 'allclose', np.allclose(a,b,atol=1e-8))
rvp=[[v for v in r['relevant_variables'][:,i] if v is not None] for i in range(k)]
rvr=[R['rv'][c] for c in R['rv']]
print('relevant eq', [sorted(a)==sorted(b) for a,b in zip(rvp,rvr)])
print('R rv', rvr); print('Py rv', rvp)
print(f"time R {R['time'][0]:.2f}s Py {tp:.2f}s")
