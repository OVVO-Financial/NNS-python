import json, numpy as np
from nns import nns_seas
R = json.load(open('/home/user/audit/work/TS/seas_r.json'))
for nm in ['s12','zero_mean']:
    r=R[nm]; x=np.array(r['x'])
    p3 = nns_seas(x, modulo=[7,12], mod_only=False)
    a=np.array(r['m712']); b=p3['periods']
    print(nm, len(a), len(b), set(a)==set(b), np.nonzero(a!=b)[0][:5], a[np.nonzero(a!=b)[0][:6]], b[np.nonzero(a!=b)[0][:6]])
    print(p3['all.periods']['Coefficient.of.Variation'][np.nonzero(a!=b)[0][:6]])
