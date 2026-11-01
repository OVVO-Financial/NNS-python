import numpy as np, pandas as pd
from nns import nns_arma
x=pd.read_csv('data/ap.csv')['y']
def f(l, fn):
    try: r=fn(); print(l, ':', np.round(np.asarray(r)[:4],3))
    except Exception as e: print(l, ': ERR', type(e).__name__, str(e)[:120])
f('series input', lambda: nns_arma(x, h=4, seasonal_factor=12))
f('sfTRUE bpNone', lambda: nns_arma(x.values, h=4, seasonal_factor=True, best_periods=None))
f('sf=1 nonlin', lambda: nns_arma(x.values, h=4, seasonal_factor=1))
f('sfFALSE bpNone', lambda: nns_arma(x.values, h=4, seasonal_factor=False, best_periods=None))
f('h0', lambda: nns_arma(x.values, h=0, seasonal_factor=12))
f('ts>n', lambda: nns_arma(x.values, h=4, training_set=200, seasonal_factor=12))
f('sf 12.5', lambda: nns_arma(x.values, h=4, seasonal_factor=12.5, method='lin'))
f('sf None', lambda: nns_arma(x.values, h=4, seasonal_factor=None))
f('sf np.int64', lambda: nns_arma(x.values, h=4, seasonal_factor=np.int64(12)))
f('weights single 0.5 lin', lambda: nns_arma(x.values, h=4, seasonal_factor=12, weights=0.5, method='lin'))
f('weights single 0.5 nonlin', lambda: nns_arma(x.values, h=4, seasonal_factor=12, weights=0.5, method='nonlin'))
f('modulo no-match', lambda: nns_arma(x.values, h=4, modulo=7))
f('method bogus', lambda: nns_arma(x.values, h=4, method='NONLINEAR'))
f('weights wrong length', lambda: nns_arma(x.values, h=4, seasonal_factor=[12,24], weights=[1.0]))
f('training_set=0', lambda: nns_arma(x.values, h=4, training_set=0, seasonal_factor=12))
