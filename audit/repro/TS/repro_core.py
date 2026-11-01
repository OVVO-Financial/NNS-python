"""Compact Python repros for TS findings (run with the NNS-python venv)."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from nns import nns_arma, nns_arma_optim, nns_seas, nns_meboot, nns_mc
ap = pd.read_csv('/home/user/audit/data/airpassengers.csv')['y'].values.astype(float)
def tryit(label, f):
    try: print(label, np.round(np.asarray(f()), 3) if not isinstance(f(), dict) else 'dict OK')
    except Exception as e: print(label, 'ERR', type(e).__name__, e)
tryit("TS-means: method='means' ->", lambda: nns_arma(ap, h=4, seasonal_factor=12, method='means'))
t = np.arange(1, 241); s = 3*np.sin(2*np.pi*t/12) + np.random.default_rng(1).normal(0, .5, 240)
print('TS-nonlin-clip last obs', np.round(s[-12:], 2)); tryit('   nonlin fcst (negative_values=True):', lambda: nns_arma(s, h=12, seasonal_factor=12, negative_values=True))
tryit('TS-weights w=[1,1]:', lambda: nns_arma(ap, h=3, seasonal_factor=[12, 24], method='lin', weights=[1, 1]))
tryit('TS-weights w=[1] (wrong length, silently broadcast):', lambda: nns_arma(ap, h=3, seasonal_factor=[12, 24], method='lin', weights=[1.0]))
tryit('TS-pi-lin: lin single lag + pred_int (R works):', lambda: nns_arma(ap, h=12, seasonal_factor=12, method='lin', pred_int=0.95))
tryit('TS-pi-h1: h=1 + pred_int:', lambda: nns_arma(ap, h=1, seasonal_factor=12, pred_int=0.95))
tryit('TS-ts>n: training_set=200 > n silently accepted:', lambda: nns_arma(ap, h=2, training_set=200, seasonal_factor=12))
tryit('TS-short: n=3 default:', lambda: nns_arma(np.array([1., 2, 3]), h=2))
tryit('TS-sfNone:', lambda: nns_arma(ap, h=2, seasonal_factor=None))
print('TS-seas-odd: n=13 periods', nns_seas(np.array([1, 2, 3]*4 + [1.]))['periods'], '(R: [3 1])')
print('TS-seas-tie: zero-mean alternating, modulo=[7,12], mod_only=False best',
      nns_seas(np.array([-1, 1]*5, float), modulo=[7, 12], mod_only=False)['best.period'], '(R: 2)')
tryit('TS-meboot-vec-drift:', lambda: nns_meboot(ap, reps=5, rho=0, target_drift=[1, 7]))
tryit('TS-mc-vec-drift:', lambda: nns_mc(ap, reps=5, by=1, target_drift=[1, 7]))
o = nns_arma_optim(ap[:132], h=12, seasonal_factor=[6, 12, 18, 24], linear_approximation=False)
print('TS-noapprox: Python', o['periods'], o['method'], o['obj.fn'], '(R: lin [12] 0.0789 = lin.only result)')
