import json, numpy as np, warnings
from nns import nns_arma, nns_arma_optim
from nns.arma import _default_arma_optim_objective as obj
C={c['name']:c for c in json.load(open('optim_cfg.json'))}
x=np.array(C['ap_doc2']['x']); n=len(x); tr=int(np.floor(.8*n)); act=x[tr:]
lin=nns_arma(x,h=n-tr,training_set=tr,seasonal_factor=[12],method='lin')
non=nns_arma(x,h=n-tr,training_set=tr,seasonal_factor=[12],method='nonlin')
both=nns_arma(x,h=n-tr,training_set=tr,seasonal_factor=[12],method='both')
print('obj avg-of-paths', obj((lin+non)/2, act), ' obj NNS.ARMA both', obj(both, act), 'obj lin',obj(lin,act),'obj nonlin',obj(non,act))
r=nns_arma_optim(x,h=12,seasonal_factor=[12,18,24])
print('reported', r['obj.fn'], 'errors-based obj', obj(r['errors']+act, act))
print('max diff path-avg vs both', np.max(np.abs((lin+non)/2-both)))
