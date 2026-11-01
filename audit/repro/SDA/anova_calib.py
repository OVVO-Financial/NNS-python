import numpy as np, pandas as pd, nns
from scipy import stats
rng=np.random.default_rng(0)
def sim(n, shift=0.0, scale=1.0, reps=400, **kw):
    c=[]; pt=[]; pks=[]
    for _ in range(reps):
        x=rng.normal(0,1,n); y=rng.normal(shift,scale,n)
        c.append(nns.nns_anova(x,y,confidence_interval=None,**kw)["Certainty"])
        pt.append(stats.ttest_ind(x,y,equal_var=False).pvalue); pks.append(stats.ks_2samp(x,y).pvalue)
    c=np.array(c); pt=np.array(pt); pks=np.array(pks)
    return c, pt, pks
for kw in ({"means_only":True},{"means_only":False}):
    for n in (30,100,1000):
        c,pt,pks=sim(n,**kw)
        print(f"H0 n={n} {kw}: Certainty quantiles 5/25/50/75/95% = {np.round(np.quantile(c,[.05,.25,.5,.75,.95]),3)}, P(C<0.10)={np.mean(c<0.10):.3f}, P(C<0.05)={np.mean(c<.05):.3f}  (t-test P(p<.05)={np.mean(pt<.05):.3f})")
for kw in ({"means_only":True},{"means_only":False}):
    for (sh,sc) in ((0.3,1.0),(0.0,1.5)):
        c,pt,pks=sim(100,sh,sc,**kw)
        print(f"H1 n=100 shift={sh} scale={sc} {kw}: median C={np.median(c):.3f} P(C<.05)={np.mean(c<.05):.3f} | t P(p<.05)={np.mean(pt<.05):.3f} KS P(p<.05)={np.mean(pks<.05):.3f}")
F=pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv").iloc[:,1:]
for a,b in (("SPY","TLT"),("SPY","QQQ"),("NVDA","TLT"),("SPY","USO")):
    print(a,b,"NNS full",round(nns.nns_anova(F[a].values,F[b].values,confidence_interval=None)["Certainty"],3),
          "means.only",round(nns.nns_anova(F[a].values,F[b].values,confidence_interval=None,means_only=True)["Certainty"],3),
          "KS p=%.2e"%stats.ks_2samp(F[a],F[b]).pvalue, "Welch p=%.3f"%stats.ttest_ind(F[a],F[b],equal_var=False).pvalue, "sd ratio %.2f"%(F[a].std()/F[b].std()))
ir=pd.read_csv("/home/user/audit/data/iris.csv"); g=[v["Sepal.Length"].values for _,v in ir.groupby("Species")]
print("iris oneway F p=%.2e kruskal p=%.2e"%(stats.f_oneway(*g).pvalue, stats.kruskal(*g).pvalue), "NNS multi", nns.nns_anova(g)["Certainty"])
print("fin oneway F p=%.3f kruskal p=%.3f"%(stats.f_oneway(*[F[c] for c in F]).pvalue, stats.kruskal(*[F[c] for c in F]).pvalue), "NNS", nns.nns_anova(F.to_numpy())["Certainty"])
