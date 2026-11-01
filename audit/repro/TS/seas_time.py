import numpy as np, time
from nns import nns_seas
import nns.seasonality as S
rng=np.random.default_rng(1)
for n in [1000, 10000, 100000]:
    x=100+np.sin(2*np.pi*np.arange(1,n+1)/12)+rng.normal(size=n)
    t=time.time(); r=nns_seas(x); print('Py seas n=',n, round(time.time()-t,3),'s best', r['best.period'])
