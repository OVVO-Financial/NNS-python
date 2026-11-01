import numpy as np, pandas as pd, time, warnings, cProfile, pstats
warnings.filterwarnings('ignore')
from nns import nns_var
for f in ['daily5_500','daily5_1000']:
    X=pd.read_csv(f'data/{f}.csv')
    t=time.time(); r=nns_var(X,h=5,tau=2,status=False); el=time.time()-t
    R=pd.read_csv(f'var_ens_{f}.csv').values
    print(f,'Py VAR %.2f s'%el, 'ensemble max|diff| vs R', np.max(np.abs(R-r['ensemble'])))
X=pd.read_csv('data/daily5_1000.csv')
cProfile.run('nns_var(X,h=5,tau=2,status=False)','prof_var.out'); pstats.Stats('prof_var.out').sort_stats('tottime').print_stats(12)
