import numpy as np, pandas as pd
from nns.anova import _anova_bin
from nns.dependence import _gravity
F=pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv"); c=F.SPY.values[:300]; t=F.QQQ.values[:300]
print("corr SPY/QQQ first 300 days: %.3f"%np.corrcoef(c,t)[0,1])
def run(scheme, seed):
    rng=np.random.default_rng(seed); l=min(c.size,t.size); P=rng.integers(0,l,(l,100))
    cert=[]
    for g in range(100):
        ti = P[:,g] if scheme=="paired" else (P[:,0] if g==0 else P[:,100-g])  # R: treatment col g -> p_{102-g} (1-based)
        cert.append(_anova_bin(c[P[:,g]], t[ti], means_only=False, medians=False)["Certainty"])
    cert=np.array(cert); return _gravity(cert), np.quantile(cert,.025)
for s in ("paired","R-scheme"):
    v=np.array([run(s,k) for k in range(20)]).mean(0); print(s, "robust est %.3f lower %.3f"%tuple(v))
