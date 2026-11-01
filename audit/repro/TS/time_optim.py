import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
from nns import nns_arma_optim, nns_arma
for f in ['vix_d1000','vix_d3000']:
    x=pd.read_csv(f'data/{f}.csv')['y'].values
    t=time.time(); r=nns_arma_optim(x,h=20,seasonal_factor=list(range(1,25)),print_trace=False); print(f,'optim Py',round(time.time()-t,2),'s periods',r['periods'],r['method'],r['obj.fn'])
    t=time.time(); a=nns_arma(x,h=50,seasonal_factor=[5,21],method='nonlin'); print(f,'ARMA nonlin h=50 Py',round(time.time()-t,2),'s',a[:3])
