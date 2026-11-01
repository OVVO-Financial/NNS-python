import json, time, numpy as np, pandas as pd, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
from nns import nns_meboot
R=json.load(open('/home/user/audit/work/TS/meb_r.json'))
d='/home/user/audit/work/TS/data/'
S={'ap':pd.read_csv(d+'ap.csv')['y'].values.astype(float),'vix':pd.read_csv(d+'vix_monthly.csv')['y'].values,'spyd':pd.read_csv(d+'spy_daily_ret_1000.csv')['y'].values}
def stats(x,b):
    Rm=b['replicates']; t=np.arange(len(x))
    sp=[spearmanr(c,x)[0] for c in Rm.T]; pe=[np.corrcoef(c,x)[0,1] for c in Rm.T]; sl=[np.polyfit(t,c,1)[0] for c in Rm.T]
    return dict(sp_mean=np.mean(sp), sp_sd=np.std(sp,ddof=1), pe_mean=np.mean(pe), colmean_mean=Rm.mean(0).mean(), colmean_sd=Rm.mean(0).std(ddof=1),
        colsd_mean=Rm.std(0,ddof=1).mean(), slope_mean=np.mean(sl), ens_mean=b['ensemble'].mean(), ens_sd=b['ensemble'].std(ddof=1), min=Rm.min(), max=Rm.max(),
        q01=np.quantile(Rm,.01), q99=np.quantile(Rm,.99))
keys=['sp_mean','sp_sd','pe_mean','colmean_mean','colmean_sd','colsd_mean','slope_mean','ens_sd','min','max','q01','q99']
def show(tag, r, p, extra=''):
    print(f'{tag:14s}'+extra+' | '.join(f"{k}: R {r[k]:.4g} Py {p[k]:.4g}" for k in keys))
for nm,x in S.items():
    det=R[nm+' det']; b=nns_meboot(x,reps=300,rho=0,random_seed=1)
    print(nm,'deterministic:', {k: bool(np.allclose(np.atleast_1d(b[k]), np.atleast_1d(det[k]))) for k in ['xx','z','dv','dvtrim','xmin','xmax','desintxb','ordxx']})
    for rho in [-1,-0.5,0,0.5,1]:
        t0=time.time(); b=nns_meboot(x,reps=300,rho=rho,random_seed=2); tp=time.time()-t0
        r=R[f'{nm} {rho:g}']; show(f'{nm} rho={rho}', r, stats(x,b), f't R {r["time"]:.2f}s Py {tp:.2f}s | ')
    for tag,kw in [('nodrift',dict(drift=False)),('td0',dict(target_drift=0.0)),('tds2',dict(target_drift_scale=2.0)),('xmin',dict(xmin=float(x.min()))),('pearson',dict(type='pearson'))]:
        b=nns_meboot(x,reps=300,rho=0.5,random_seed=3,**kw); show(f'{nm} {tag}', R[f'{nm} {tag}'], stats(x,b))
x=S['ap']; t0=time.time(); b=nns_meboot(x,reps=50,rho=0.5,type='nnsdep',random_seed=4); tp=time.time()-t0
show('ap nnsdep', R['ap nnsdep'], stats(x,b), f't R {R["ap nnsdep"]["time"]:.2f}s Py {tp:.2f}s | ')
