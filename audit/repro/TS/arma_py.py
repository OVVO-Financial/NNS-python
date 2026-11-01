import json, time, numpy as np, warnings
warnings.filterwarnings('ignore')
from nns import nns_arma
C=json.load(open('/home/user/audit/work/TS/arma_cfg.json')); R=json.load(open('/home/user/audit/work/TS/arma_r.json'))
nbad=0
for i,(c,r) in enumerate(zip(C['cfgs'],R)):
    x=np.array(C['series'][c['series']])
    kw={k:v for k,v in c.items() if k not in ('series',)}
    t0=time.time()
    try: p=np.asarray(nns_arma(x, **kw),float); err=None
    except Exception as e: p=None; err=f'{type(e).__name__}: {e}'
    tp=time.time()-t0
    ro=r['out']
    if isinstance(ro,str) or (isinstance(ro,list) and ro and isinstance(ro[0],str) and ro[0].startswith('ERR')):
        status = 'R-ERR' + ('' if err else ' (py ok)')
        print(i, c, status, ro if isinstance(ro,str) else ro[0], '| py:', err or p[:4]); nbad+=1; continue
    ro=np.array([np.nan if v=='NA' or v=='NaN' else v for v in np.atleast_1d(ro)],float)
    if err: print(i,c,'PY-ERR',err,'R:',ro[:4]); nbad+=1; continue
    ok = p.shape==ro.shape and np.allclose(p,ro,rtol=1e-7,atol=1e-7,equal_nan=True)
    if not ok:
        nbad+=1; print(i,c,'MISMATCH\n   R:',np.round(ro[:6],4),'\n  Py:',np.round(p[:6],4))
    else:
        flag = ' ALLZERO' if np.all(ro==0) else (' NAN' if np.any(np.isnan(ro)) else '')
        print(i, c['series'], c['method'], c.get('seasonal_factor'), 'OK', flag, f"R {r['t']*1000:.1f}ms Py {tp*1000:.1f}ms")
print('bad',nbad)
