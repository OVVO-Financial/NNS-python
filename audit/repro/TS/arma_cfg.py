import json, pandas as pd
d='/home/user/audit/work/TS/data/'
S=pd.read_csv(d+'synth.csv')
series={'ap':pd.read_csv(d+'ap.csv')['y'].tolist(),'vix':pd.read_csv(d+'vix_monthly.csv')['y'].tolist(),
 's_neg':S['s_neg'].tolist(),'wn':S['wn'].tolist(),'s12_trend':S['s12_trend'].tolist(),'const':[5.0]*60,
 'ret':pd.read_csv(d+'panel_monthly.csv')['SPY'].tolist()}
cfgs=[]
def add(s, **kw): cfgs.append(dict(series=s, **kw))
for m in ['lin','nonlin','both','means']:
    for sf in [True, False, 12, [12,24,36]]:
        add('ap', h=12, training_set=120, seasonal_factor=sf, method=m, best_periods=(2 if sf is False else 1))
for m in ['lin','nonlin','both']:
    add('ap', h=45, training_set=100, seasonal_factor=12, method=m)
    add('ap', h=12, seasonal_factor=[12,24], method=m, weights=[0.5,0.5])
    add('ap', h=12, seasonal_factor=[12,24], method=m, weights=[1,1])
    add('ap', h=12, seasonal_factor=False, best_periods=3, method=m, weights='equal')
    add('ap', h=12, seasonal_factor=12, method=m, shrink=True)
    add('ap', h=12, seasonal_factor=[12,24], method=m, shrink=True)
    add('ap', h=6, seasonal_factor=False, method=m, dynamic=True, best_periods=2)
    add('ap', h=6, seasonal_factor=True, method=m, dynamic=True)
    add('ap', h=12, seasonal_factor=True, method=m, modulo=12)
    for s in ['vix','s_neg','wn','s12_trend','const','ret']:
        add(s, h=12, seasonal_factor=True, method=m)
        add(s, h=12, seasonal_factor=12, method=m)
    add('s_neg', h=12, seasonal_factor=12, method=m, negative_values=True)
    add('ap', h=12, seasonal_factor=12, method=m, negative_values=True)
add('ap', h=24, seasonal_factor=12, method='lin', shrink=True)
add('ap', h=5, seasonal_factor=12, method='lin', shrink=True)
add('ap', h=1, seasonal_factor=12, method='nonlin')
add('ap', h=1, seasonal_factor=12, method='lin')
add('ap', h=12, seasonal_factor=1, method='lin')
add('ap', h=12, seasonal_factor=1, method='nonlin')
add('ap', h=12, seasonal_factor=200, method='nonlin')
json.dump({'series':series,'cfgs':cfgs}, open('/home/user/audit/work/TS/arma_cfg.json','w'))
print(len(cfgs))
