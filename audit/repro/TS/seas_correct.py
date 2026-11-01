import numpy as np, pandas as pd
from nns import nns_seas
t=np.arange(1,145)
ap=pd.read_csv('data/ap.csv')['y'].values.astype(float)
cases={'pure sin P=12 (mean 100, amp 10, no noise)':100+10*np.sin(2*np.pi*t/12),
 'sin P=12 + N(0,1)':100+10*np.sin(2*np.pi*t/12)+np.random.default_rng(0).normal(0,1,144),
 'sin P=7 + N(0,1)':100+10*np.sin(2*np.pi*t/7)+np.random.default_rng(0).normal(0,1,144),
 'AirPassengers raw':ap,
 'AirPassengers log-diff':np.diff(np.log(ap)),
 'AirPassengers detrended ratio (x/MA12)':ap[6:-6]/pd.Series(ap).rolling(12,center=True).mean().values[6:-6]}
for k,x in cases.items():
    r=nns_seas(x); p=r['periods']; rank12 = (np.flatnonzero(p==12)[0]+1) if 12 in p else None
    print(f'{k:45s} n={x.size} best={r["best.period"]:4d} top5={list(p[:5])} rank of 12: {rank12} (of {p.size})')
