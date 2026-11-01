import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
from nns import nns_arma, nns_arma_optim
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
d='/home/user/audit/work/TS/data/'
S={'AirPassengers':pd.read_csv(d+'ap.csv')['y'].values.astype(float),
   'VIX_monthly':pd.read_csv(d+'vix_monthly.csv')['y'].values,
   'SPY_monthly_px':pd.read_csv(d+'spy_monthly.csv')['y'].values}
h=12
def models(x, name):
    out={}
    out['naive']=np.repeat(x[-1],h)
    out['snaive12']=np.tile(x[-12:], 2)[:h]
    out['mean']=np.repeat(x.mean(),h)
    try:
        if name=='AirPassengers':
            out['HW_mult']=ExponentialSmoothing(x,trend='add',seasonal='mul',seasonal_periods=12).fit().forecast(h)
        out['ARIMA(1,1,1)']=ARIMA(x,order=(1,1,1)).fit().forecast(h)
        if name!='SPY_monthly_px':
            out['ARIMA(1,0,1)']=ARIMA(x,order=(1,0,1)).fit().forecast(h)
    except Exception as e: print('sm err',e)
    out['NNS.ARMA default']=nns_arma(x,h=h)
    out['NNS.ARMA sf=12 nonlin']=nns_arma(x,h=h,seasonal_factor=12)
    out['NNS.ARMA sf=12 lin']=nns_arma(x,h=h,seasonal_factor=12,method='lin')
    o=nns_arma_optim(x,h=h,seasonal_factor=list(range(1,25)),print_trace=False)
    out['NNS.ARMA.optim 1:24']=o['results']
    out['_optim_cfg']=(list(o['periods']),o['method'])
    return out
for name,x in S.items():
    n=len(x); origins=[n-h-12*k for k in range(4)][::-1] if name!='AirPassengers' else [n-36,n-24,n-12]
    errs={}; cfgs=[]
    for o in origins:
        tr, te = x[:o], x[o:o+h]
        M=models(tr,name); cfgs.append(M.pop('_optim_cfg'))
        for k,v in M.items(): errs.setdefault(k,[]).append(np.mean(np.abs(np.asarray(v)[:len(te)]-te)))
    print(f'\n== {name}: n={n}, origins={origins}, h={h}; MAE averaged over origins'); 
    for k,v in sorted(errs.items(), key=lambda kv: np.mean(kv[1])): print(f'  {k:24s} {np.mean(v):9.3f}   per-origin {np.round(v,2)}')
    print('  optim chose', cfgs)
