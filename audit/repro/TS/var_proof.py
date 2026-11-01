"""Proof that NNS.VAR 'multivariate' ignores the forecast period entirely (R & Python).
The multivariate stage is evaluated on tail(IV, h) of the *training* lag matrix, i.e. the
last h OBSERVED rows, so replacing the univariate forecasts by garbage changes nothing."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import nns.var as V
from nns import nns_var
P=pd.read_csv('/home/user/audit/work/TS/data/panel_monthly.csv').iloc[:, :3]
h=6
r=nns_var(P, h=h, tau=1, status=False)
base=V._var_multivariate_stack_stage(r['interpolated_and_extrapolated'], r['univariate'], h=h, tau=1, names=r['names'])['multivariate']
garbage=V._var_multivariate_stack_stage(r['interpolated_and_extrapolated'], r['univariate']+1000.0, h=h, tau=1, names=r['names'])['multivariate']
print('multivariate unchanged when univariate forecasts += 1000:', np.array_equal(base, garbage))
print('shipped multivariate:\n', np.round(r['multivariate'],4))
print('last h OBSERVED values (in-sample):\n', np.round(P.values[-h:],4))
