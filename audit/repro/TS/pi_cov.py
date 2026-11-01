import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from nns import nns_arma, nns_arma_optim
d='/home/user/audit/work/TS/data/'
S={'AirPassengers':pd.read_csv(d+'ap.csv')['y'].values.astype(float),'VIX_monthly':pd.read_csv(d+'vix_monthly.csv')['y'].values,
   'SPY_ret':pd.read_csv(d+'panel_monthly.csv')['SPY'].values}
h=12
for name,x in S.items():
    n=len(x); origins=list(range(n-h-60, n-h+1, 6))
    cov_a=[]; w_a=[]; cov_o=[]; w_o=[]
    for o in origins:
        tr,te=x[:o],x[o:o+h]
        r=nns_arma(tr,h=h,seasonal_factor=12,method='nonlin',pred_int=0.95,random_seed=o)
        lo,hi=r['Lower 95% pred.int'],r['Upper 95% pred.int']; cov_a.append(np.mean((te>=lo)&(te<=hi))); w_a.append(np.mean(hi-lo))
        q=nns_arma_optim(tr,h=h,seasonal_factor=list(range(1,25)),print_trace=False)
        cov_o.append(np.mean((te>=q['lower.pred.int'])&(te<=q['upper.pred.int']))); w_o.append(np.mean(q['upper.pred.int']-q['lower.pred.int']))
    print(f'{name}: {len(origins)} origins, h={h}. nominal 95%: nns_arma(pred_int) coverage {np.mean(cov_a):.2f} (mean width {np.mean(w_a):.3g});  nns_arma_optim PI coverage {np.mean(cov_o):.2f} (width {np.mean(w_o):.3g}); series sd {x.std():.3g}')
