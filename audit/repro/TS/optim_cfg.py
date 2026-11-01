import json, pandas as pd
d='/home/user/audit/work/TS/data/'
ap=pd.read_csv(d+'ap.csv')['y'].tolist(); vix=pd.read_csv(d+'vix_monthly.csv')['y'].tolist()
spy=pd.read_csv(d+'spy_monthly.csv')['y'].tolist(); ret=pd.read_csv(d+'panel_monthly.csv')['SPY'].tolist()
cfgs=[dict(name='ap_doc1', x=ap[:132], training_set=120, seasonal_factor=[12,18,24]),
      dict(name='ap_doc2', x=ap[:132], h=12, seasonal_factor=[12,18,24]),
      dict(name='ap_1_24', x=ap[:132], h=12, seasonal_factor=list(range(1,25))),
      dict(name='ap_lin_only', x=ap[:132], h=12, seasonal_factor=list(range(1,25)), lin_only=True),
      dict(name='ap_noapprox', x=ap[:132], h=12, seasonal_factor=[6,12,18,24], linear_approximation=False),
      dict(name='ap_max', x=ap[:132], h=12, seasonal_factor=[6,12,24], objective='max'),
      dict(name='vix_1_24', x=vix[:-12], h=12, seasonal_factor=list(range(1,25))),
      dict(name='vix_ts', x=vix[:-12], training_set=len(vix[:-12])-12, seasonal_factor=[1,2,3,6,12]),
      dict(name='spy_1_24', x=spy[:-12], h=12, seasonal_factor=list(range(1,25))),
      dict(name='ret_1_12', x=ret[:-12], h=12, seasonal_factor=list(range(1,13))),
]
json.dump(cfgs, open(d+'../optim_cfg.json','w'))
