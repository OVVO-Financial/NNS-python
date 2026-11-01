import json, numpy as np
from nns import nns_seas
R = json.load(open('/home/user/audit/work/TS/seas_r.json'))
for nm, r in R.items():
    x = np.array(r['x'], float)
    p = nns_seas(x); p2 = nns_seas(x, modulo=[12]); p3 = nns_seas(x, modulo=[7,12], mod_only=False)
    def L(v): return list(np.atleast_1d(v))
    ok1 = L(p['periods'])==L(r['periods']); ok2 = L(p2['periods'])==L(r['m12']); ok3 = L(p3['periods'])==L(r['m712'])
    cvok = np.allclose(p['all.periods']['Coefficient.of.Variation'], r['cv']) if ok1 else None
    print(nm, 'n=',len(x),'best R/Py', r['best'], p['best.period'], 'periods eq',ok1, cvok, 'mod12', ok2, 'mod7,12 noonly', ok3)
    if not ok1: print('  R', L(r['periods'])[:15], '\n  P', L(p['periods'])[:15])
    if not ok2: print('  R12', L(r['m12'])[:15], '\n  P12', L(p2['periods'])[:15])
    if not ok3: print('  R', L(r['m712'])[:20], '\n  P', L(p3['periods'])[:20])
    print('   top periods R:', L(r['periods'])[:8])
