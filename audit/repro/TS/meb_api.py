import numpy as np, pandas as pd, traceback
from nns import nns_meboot, nns_mc
x=pd.read_csv('data/ap.csv')['y']
def t(label, f):
    try: r=f(); print(label,'OK', type(r).__name__, (len(r) if isinstance(r,list) else list(r.keys())[:4]) if r is not None else None)
    except Exception as e: print(label,'ERR',type(e).__name__, str(e)[:150])
t('series input', lambda: nns_meboot(x, reps=5, rho=0.5))
t('int dtype', lambda: nns_meboot(x.values.astype(int), reps=5, rho=0.5))
t('target_drift vector', lambda: nns_meboot(x.values, reps=5, rho=0, target_drift=[1,7]))
t('target_drift_scale vector', lambda: nns_meboot(x.values, reps=5, rho=0, target_drift_scale=[0.5,2]))
t('reps=1', lambda: nns_meboot(x.values, reps=1, rho=0.5))
t('n=2', lambda: nns_meboot(np.array([1.,2.]), reps=5, rho=0.5))
t('n=3', lambda: nns_meboot(np.array([1.,2.,4.]), reps=5, rho=0.5))
t('const', lambda: nns_meboot(np.ones(20), reps=5, rho=0.5))
t('linear exact', lambda: nns_meboot(np.arange(20.), reps=5, rho=0.5))
t('rho None', lambda: nns_meboot(x.values, reps=5))
t('2d col', lambda: nns_meboot(x.values.reshape(-1,1), reps=5, rho=0.5))
t('mc by wrong sign', lambda: nns_mc(x.values, reps=3, lower_rho=1, upper_rho=-1, by=0.5))
t('mc default', lambda: nns_mc(x.values, reps=3, by=0.5, random_seed=1))
r=nns_meboot(x.values, reps=5, rho=0.5, xmin=0, force_clt=False); print('xmin+force_clt=False colsd', r['replicates'].std(0,ddof=1))
t('trim .5', lambda: nns_meboot(x.values, reps=3, rho=0, trim=0.5)['dvtrim'])
print('dvtrim trim=.5 py', nns_meboot(x.values, reps=3, rho=0, trim=0.5)['dvtrim'])
