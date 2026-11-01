import json, numpy as np, warnings
warnings.filterwarnings('ignore')
from nns import nns_arma, nns_arma_optim
from nns.arma import _default_arma_optim_objective as obj
C={c['name']:c for c in json.load(open('optim_cfg.json'))}
for nm in ['vix_1_24','ret_1_12']:
    x=np.array(C[nm]['x']); n=len(x); tr=int(np.floor(.8*n)); act=x[tr:]
    r=nns_arma_optim(x,h=12,seasonal_factor=C[nm]['seasonal_factor'])
    p=list(r['periods']); m=r['method']
    sel=nns_arma(x,h=n-tr,training_set=tr,seasonal_factor=p,method=m)
    eq=nns_arma(x,h=n-tr,training_set=tr,seasonal_factor=p,method=m,weights=np.full(len(p),1/len(p)))
    print(nm, 'periods',p,m,'reported obj',round(r['obj.fn'],4),'| obj(default weights)',round(obj(sel,act),4),'| obj(equal weights)',round(obj(eq,act),4),'| weights returned',r['weights'])
