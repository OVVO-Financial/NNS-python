import numpy as np, pandas as pd, nns, warnings, traceback, signal, sys
warnings.simplefilter('ignore')
rng=np.random.default_rng(7)
inputs={
 'len1':np.array([1.5]),'len2':np.array([1.,2.]),'len3':np.array([3.,1.,2.]),
 'const':np.full(30,2.0),'ties':np.repeat([1.,2.,3.],10),'ints':np.arange(30),
 'normal':rng.normal(size=60),'big':1e9+rng.normal(size=60),'tiny':1e-12*rng.normal(size=60),
 'series':pd.Series(rng.normal(size=40)),'bool':rng.random(40)>.5,
}
def y_for(x): 
    x=np.asarray(x,dtype=float); return np.sin(x)+0.1*np.arange(len(x))
fns={
 'nns_dep':lambda x:nns.nns_dep(x,y_for(x)),
 'nns_cor':lambda x:nns.nns_cor(x,y_for(x)) if hasattr(nns,'nns_cor') else None,
 'nns_copula':lambda x:nns.nns_copula(np.column_stack([x,y_for(x)])),
 'nns_reg':lambda x:nns.nns_reg(x,y_for(x)),
 'nns_reg_pe':lambda x:nns.nns_reg(x,y_for(x),point_est=np.asarray(x,float)[:1]),
 'nns_part':lambda x:nns.nns_part(x,y_for(x)),
 'nns_causation':lambda x:nns.nns_causation(x,y_for(x)),
 'nns_moments':lambda x:nns.nns_moments(x),
 'nns_mode':lambda x:nns.nns_mode(x),
 'nns_gravity':lambda x:nns.nns_gravity(x),
 'nns_cdf':lambda x:nns.nns_cdf(x),
 'lpm_var':lambda x:nns.lpm_var(0.05,1,x),
 'fsd':lambda x:nns.fsd(x,np.asarray(x,float)+0.1),
 'ssd':lambda x:nns.ssd(x,np.asarray(x,float)+0.1),
 'nns_anova':lambda x:nns.nns_anova(x,np.asarray(x,float)+0.1),
 'nns_norm':lambda x:nns.nns_norm(np.column_stack([x,y_for(x)])),
 'nns_seas':lambda x:nns.nns_seas(x),
 'nns_arma':lambda x:nns.nns_arma(x,h=2),
 'nns_meboot':lambda x:nns.nns_meboot(x,reps=3),
 'dy_dx':lambda x:nns.dy_dx(x,y_for(x)),
 'nns_ss':lambda x:nns.nns_ss(x,np.asarray(x,float)+0.1),
 'nns_stack':lambda x:nns.nns_stack(np.asarray(x,float).reshape(-1,1),y_for(x),np.asarray(x,float)[:2].reshape(-1,1)),
}
class TO(Exception): pass
def h(*a): raise TO()
signal.signal(signal.SIGALRM,h)
for fn,f in fns.items():
    out=[]
    for k,x in inputs.items():
        signal.alarm(20)
        try:
            r=f(x); out.append((k,'ok'))
        except TO: out.append((k,'TIMEOUT'))
        except Exception as e: out.append((k,f'{type(e).__name__}: {str(e)[:70]}'))
        finally: signal.alarm(0)
    bad=[o for o in out if o[1]!='ok']
    print(fn, 'all ok' if not bad else '')
    for b in bad: print('   ',b)
    sys.stdout.flush()
