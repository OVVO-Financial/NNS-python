import numpy as np, pandas as pd, warnings
from nns import nns_arma
x=pd.read_csv('data/ap.csv')['y'].values
for kw in [dict(h=1, seasonal_factor=12, method='nonlin'), dict(h=12, seasonal_factor=12, method='lin'),
           dict(h=12, seasonal_factor=[12,24], method='lin'), dict(h=12, seasonal_factor=12, method='nonlin'), dict(h=12, seasonal_factor=12, method='means')]:
    try:
        r=nns_arma(x, pred_int=.95, random_seed=1, **kw); print(kw, {k:np.round(v[:4],1) for k,v in r.items()})
    except Exception as e: print(kw,'ERR',type(e).__name__,e)
