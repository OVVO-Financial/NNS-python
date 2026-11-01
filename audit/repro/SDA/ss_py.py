import json, numpy as np, pandas as pd, nns, time
r=json.load(open("ss_r.json")); R=pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv").iloc[:,1:]
mx=0
for i in range(R.shape[1]):
    for j in range(R.shape[1]):
        p=nns.nns_ss(R.iloc[:,i].values,R.iloc[:,j].values)
        mx=max(mx, max(abs(p[k]-v) for k,v in zip(["p_gt","p_tie","p_star"], r[f"{i+1} {j+1}"])))
print("finance max abs diff", mx)
a=pd.read_csv("ss_a.csv").a.values; b=pd.read_csv("ss_b.csv").b.values
print("ties", nns.nns_ss(a,b), r["ties"])
print("nan", nns.nns_ss([1,np.nan,3],[2,np.nan]), r["nan"])
print("inf", nns.nns_ss([1,np.inf,3],[-np.inf,2,np.inf]), r["inf"])
ci=nns.nns_ss(R.SPY.values[:500],R.TLT.values[:500],confidence_interval=True,reps=199,random_seed=1)
print("ci py", ci["p_star"], ci["lower"], ci["upper"], ci["boot_vals"].mean(), ci["boot_vals"].std(ddof=1)); print("ci R ", r["ci"])
t=time.perf_counter(); [nns.nns_ss(R.SPY.values,R.TLT.values) for _ in range(50)]; print("t ms py", (time.perf_counter()-t)/50*1000, "R", r["t_ms"])
x=np.random.default_rng(0).normal(size=10**6); y=np.random.default_rng(1).normal(size=10**6)
t=time.perf_counter(); nns.nns_ss(x,y); print("1e6 py ms", (time.perf_counter()-t)*1000, "R", r["t1e6_ms"])
t=time.perf_counter(); nns.nns_ss(R.SPY.values[:1000],R.TLT.values[:1000],confidence_interval=True,reps=999); print("CI 999 reps n=1000 py ms", (time.perf_counter()-t)*1000, "R", r["tci_ms"])
for lab,f in [("np.bool_ CI", lambda: nns.nns_ss([1,2],[0,1],confidence_interval=np.True_)),
              ("2D input", lambda: nns.nns_ss(np.array([[1,2],[3,4]]),[0])),
              ("Series w NaN", lambda: nns.nns_ss(pd.Series([1,None,3]),pd.Series([2.0]))),
              ("reps=2.5", lambda: nns.nns_ss([1,2,3,4,5],[0,1,2,3,3],confidence_interval=True,reps=2.5)),
              ("ci=0.5 list", lambda: nns.nns_ss([1,2,3,4,5],[0,1,2,3,3],confidence_interval=True,reps=5,ci=0.5)["lower"])]:
    try: print(lab, f())
    except Exception as e: print(lab, "EXC", type(e).__name__, e)
