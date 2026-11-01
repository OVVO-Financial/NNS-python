import json, numpy as np, pandas as pd, nns
r=json.load(open("anova_r.json"))
ir=pd.read_csv("/home/user/audit/data/iris.csv"); F=pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv").iloc[:,1:]
sp=[g["Sepal.Length"].to_numpy() for _,g in ir.groupby("Species")]
def num(v): return np.nan if v in ("NaN","NA",None) else float(v)
def cmp(tag, py, rr):
    if isinstance(rr, dict):
        d={k: abs(float(py[k])-num(v)) if not (np.isnan(float(py[k])) and np.isnan(num(v))) else 0.0 for k,v in rr.items()}
        m=max(d.values()); print(f"{tag:22s} maxdiff {m:.2e}", "" if m<1e-9 else d)
    else:
        a=np.array(py,dtype=float); b=np.array(rr,dtype=float).reshape(a.shape)
        m=np.nanmax(np.abs(a-b)); print(f"{tag:22s} maxdiff {m:.2e}  R={np.ravel(b)[:3]} py={np.ravel(a)[:3]}")
for mo in (0,1):
    for md in (0,1):
        tag=f"mo{mo}md{md}"; kw=dict(means_only=bool(mo),medians=bool(md))
        cmp("iris_bin_"+tag, nns.nns_anova(sp[0],sp[1],confidence_interval=None,**kw), r["iris_bin_"+tag])
        cmp("iris_multi_"+tag, [nns.nns_anova(sp,**kw)["Certainty"]], [r["iris_multi_"+tag]])
        cmp("iris_pw_"+tag, nns.nns_anova(sp,pairwise=True,**kw), r["iris_pw_"+tag])
        cmp("fin_bin_"+tag, nns.nns_anova(F.SPY.values,F.TLT.values,confidence_interval=None,**kw), r["fin_bin_"+tag])
        cmp("fin_multi_"+tag, [nns.nns_anova(F.to_numpy(),**kw)["Certainty"]], [r["fin_multi_"+tag]])
        cmp("fin_pw_"+tag, nns.nns_anova(F.to_numpy(),pairwise=True,**kw), r["fin_pw_"+tag])
cmp("same", nns.nns_anova(sp[0],sp[0],confidence_interval=None), r["same"])
cmp("const", nns.nns_anova(np.ones(10),np.ones(10),confidence_interval=None), r["const"])
cmp("const2", nns.nns_anova(np.ones(10),np.arange(1,11.),confidence_interval=None), r["const2"])
ci=np.array([[nns.nns_anova(sp[0],sp[1],random_seed=s)[k] for k in ("Effect_Size_LB","Effect_Size_UB")] for s in range(50)]).mean(0)
print("ci_both py", ci, "R", r["ci_both"])
ci=np.array([[nns.nns_anova(sp[0],sp[1],medians=True,random_seed=s)[k] for k in ("Effect_Size_LB","Effect_Size_UB")] for s in range(30)]).mean(0)
print("ci_med py", ci, "R", r["ci_med"])
rb=nns.nns_anova(F.SPY.values[:300],F.QQQ.values[:300],robust=True,random_seed=3)
print("robust keys py", list(rb)); print("robust keys R ", r["robust_names"])
rs=np.array([[nns.nns_anova(F.SPY.values[:300],F.QQQ.values[:300],robust=True,random_seed=s)[k] for k in ("Robust Certainty Estimate","Lower Bound Robust Certainty","Upper Bound Robust Certainty")] for s in range(30)]).mean(0)
print("robust seeds py", rs, "R", np.mean(np.array(r["robust_seeds"],dtype=float),axis=1))
print("robust left py keys", list(nns.nns_anova(F.SPY.values[:300],F.QQQ.values[:300],robust=True,tails="left",random_seed=3)))
