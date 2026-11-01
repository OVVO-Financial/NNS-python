import numpy as np
from nns import nns_seas
rng=np.random.default_rng(7); n=1000
for mu in [10, 1, 0.1, 0.01, 0.0005]:
    cnt=[]; best=[]
    for rep in range(10):
        x=rng.normal(mu,0.01 if mu<0.1 else 1.0, n) if mu<0.1 else rng.normal(mu,1.0,n)
        r=nns_seas(x); p=r['periods']; cnt.append(0 if (p.size==1 and r['all.periods']['Coefficient.of.Variation'][0]==r['all.periods']['Variable.Coefficient.of.Variation'][0]) else p.size); best.append(r['best.period'])
    print(f'iid noise mean={mu}, sd={"0.01" if mu<0.1 else "1"}: periods flagged (of {n//2}) mean={np.mean(cnt):.0f}; best periods {best}')
