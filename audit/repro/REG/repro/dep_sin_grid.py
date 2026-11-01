import math, numpy as np, pandas as pd
import nns._reg_engine as eng
from nns import nns_dep
from nns.copula import _copula
d = pd.read_csv("/home/user/audit/work/REG/repro/sinxy.csv"); x, y = d.x.values, d.y.values
m = np.column_stack((eng._rescale_01(x), eng._rescale_01(x), eng._rescale_01(y)))
t_np = m.mean(axis=0); t_exact = np.array([math.fsum(c) / c.size for c in m.T])
print("py: np.mean target", [repr(v) for v in t_np], "fsum target", [repr(v) for v in t_exact])
print("py: nns_dep", nns_dep(x, y, asym=True)["Dependence"], " copula(np.mean target)", _copula(m, t_np, True), " copula(fsum target)", _copula(m, t_exact, True))
print("py: _dependence", eng._dependence(x, y), " default order", eng._default_order(x, y))
