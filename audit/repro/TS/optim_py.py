import json, time, numpy as np, warnings
warnings.filterwarnings('ignore')
from nns import nns_arma_optim
C=json.load(open('/home/user/audit/work/TS/optim_cfg.json')); R=json.load(open('/home/user/audit/work/TS/optim_r.json'))
for c in C:
    r=R[c['name']]; kw={k:v for k,v in c.items() if k not in ('name','x')}
    t0=time.time(); p=nns_arma_optim(np.array(c['x']), **kw); tp=time.time()-t0
    same = (list(np.atleast_1d(p['periods']))==list(np.atleast_1d(r['periods'])) and p['method']==r['method'])
    res_ok = np.allclose(p['results'], r['results'], rtol=1e-6, atol=1e-6)
    pi_ok = np.allclose(p['lower.pred.int'], r['lower.pred.int'], rtol=1e-6, atol=1e-6)
    print(c['name'], 'R:',r['periods'],r['method'],round(r['obj.fn'],5),r['shrink'],r['nns.regress'],round(r['bias.shift'],4),
          '| Py:',list(np.atleast_1d(p['periods'])),p['method'],round(p['obj.fn'],5),p['shrink'],p['nns.regress'],round(p['bias.shift'],4),
          '| sel eq',same,'results eq',res_ok,'PI eq',pi_ok, f"t R {r['time']:.2f}s Py {tp:.2f}s")
    if not res_ok: print('   R res',np.round(r['results'][:5],3),'\n   P res',np.round(p['results'][:5],3))
