import numpy as np, pandas as pd, warnings, inspect, textwrap
warnings.filterwarnings('ignore')
import nns.var as V
from nns import nns_var
src = inspect.getsource(V._var_multivariate_stack_stage)
assert "lagged_iv[-h:, relevant]," in src
fixed_src = src.replace("def _var_multivariate_stack_stage(", "def _fixed_stage(").replace(
    "lagged_iv[-h:, relevant],", "np.delete(lagged, i, axis=1)[-h:, relevant],")
ns = dict(vars(V)); exec(fixed_src, ns); fixed_stage = ns['_fixed_stage']
P=pd.read_csv('/home/user/audit/work/TS/data/panel_monthly.csv').iloc[:, :3]
X=P.values; n=len(X); h=6
origins=[n-h-12*k for k in range(5)][::-1]
E={}
for o in origins:
    tr, te = X[:o], X[o:o+h]
    r = nns_var(pd.DataFrame(tr, columns=P.columns), h=h, tau=1, status=False)
    fx = fixed_stage(r['interpolated_and_extrapolated'], r['univariate'], h=h, tau=1, names=r['names'])['multivariate']
    cands = {'zero':np.zeros_like(te), 'hist_mean':np.tile(tr.mean(0),(h,1)), 'univariate':r['univariate'],
             'multivariate(as shipped)':r['multivariate'], 'ensemble(as shipped)':r['ensemble'],
             'multivariate(fixed test rows)':fx, 'ensemble(fixed)':0.5*fx+0.5*r['univariate'],
             'last_h_observed (replay)':tr[-h:]}
    for k,v in cands.items(): E.setdefault(k,[]).append(np.sqrt(np.mean((v-te)**2)))
    print('origin',o,'corr(shipped multivariate, last h observed)=%.3f'%np.corrcoef(r['multivariate'].ravel(), tr[-h:].ravel())[0,1])
print(f'\nRMSE (all 3 series pooled) over origins {origins}, h={h}, tau=1:')
for k,v in sorted(E.items(), key=lambda kv: np.mean(kv[1])): print(f'  {k:32s} {np.mean(v):.5f}  {np.round(v,4)}')
