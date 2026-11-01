import numpy as np, pandas as pd, json
from nns import nns_seas, nns_arma
S=pd.read_csv('data/synth.csv'); wn=S['wn'].values
cases={'wn_mod7': (wn, dict(modulo=7)), 'linear_trend': (np.arange(1,101.)+0.0, {}), 'exp_growth': (1.03**np.arange(100), {}), 'wn_mod5':(wn, dict(modulo=5))}
out={}
for k,(x,kw) in cases.items():
    s=nns_seas(x, **kw); a=nns_arma(x, h=4, **kw)
    print(k, 'seas top', s['periods'][:4], 'cv', np.round(s['all.periods']['Coefficient.of.Variation'][:2],4), 'varcv', round(s['all.periods']['Variable.Coefficient.of.Variation'][0],4), '-> nns_arma', np.round(a,3), 'last obs', np.round(x[-2:],2))
    out[k]=dict(x=list(map(float,x)), kw=kw, py=list(map(float,a)))
json.dump(out, open('zero_cases.json','w'))
