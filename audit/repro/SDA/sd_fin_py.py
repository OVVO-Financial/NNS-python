import json, numpy as np, pandas as pd, nns
from nns import stochastic_dominance as sdm
R = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv").iloc[:,1:]
X = R.to_numpy(); nm = list(R.columns); k = X.shape[1]
r = json.load(open("sd_fin_r.json"))
def mat(f):
    M = np.zeros((k,k),int)
    for i in range(k):
        for j in range(k):
            if i!=j: M[i,j]=f(X[:,i],X[:,j])
    return M
py = {
 "FSD": mat(nns.fsd), "SSD": mat(nns.ssd), "TSD": mat(nns.tsd),
 "FSDc": mat(lambda a,b: 1 if nns.fsd_uni(a,b,type="continuous") and not np.array_equal(np.sort(a),np.sort(b)) else (-1 if nns.fsd_uni(b,a,type="continuous") else 0)),
 "FSDu": mat(nns.fsd_uni), "FSDuc": mat(lambda a,b: nns.fsd_uni(a,b,type="continuous")),
 "SSDu": mat(nns.ssd_uni), "TSDu": mat(nns.tsd_uni)}
for key,M in py.items():
    RM = np.array(r[key])
    print(key, "match" if np.array_equal(RM,M) else f"MISMATCH {np.argwhere(RM!=M).tolist()}", "nonzero:", int((M!=0).sum()))
for d in (1,2,3):
    e = [nm[i] for i in nns.sd_efficient_set(X,d)]
    print("eff",d, e, r[f"eff{d}"], e==r[f"eff{d}"])
e = [nm[i] for i in nns.sd_efficient_set(X,1,type="continuous")]; print("eff1c", e, r["eff1c"], e==r["eff1c"])
for d in (1,2,3):
    for mc in (1,2,3):
        c = nns.nns_sd_cluster(X, degree=d, min_cluster=mc, names=nm)["Clusters"]
        rc = r[f"clu{d}_{mc}"]
        print("clu",d,mc, c==rc, "" if c==rc else (c, rc))
c = nns.nns_sd_cluster(X, degree=1, type="continuous", names=nm)["Clusters"]; print("clu1c", c==r["clu1c"])
dd = nns.nns_sd_cluster(X, degree=2, dendrogram=True, names=nm)["Dendrogram"]
rd = r["dend2"]
print("dend merge", np.array_equal(np.array(rd["merge"]), dd["merge"]), "height", np.allclose(rd["height"], dd["height"]), "order", list(dd["order"])==rd["order"], "labels", list(dd["labels"])==rd["labels"])
Rm = pd.read_csv("/home/user/audit/data/returns_monthly.csv").iloc[:,1:].dropna()
Xm = Rm.to_numpy(); nmm=list(Rm.columns)
for d in (1,2,3):
    e=[nmm[i] for i in nns.sd_efficient_set(Xm,d)]; print("meff",d,e==r[f"meff{d}"], e, r[f"meff{d}"])
e=[nmm[i] for i in nns.sd_efficient_set(Xm,1,type="continuous")]; print("meff1c",e==r["meff1c"], e, r["meff1c"])
