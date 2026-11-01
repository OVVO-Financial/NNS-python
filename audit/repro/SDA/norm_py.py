import json, time, numpy as np, pandas as pd, nns
r=json.load(open("norm_r.json"))
P11=pd.read_csv("prices11.csv",index_col=0); P5=pd.read_csv("prices5.csv",index_col=0)
def rel(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return np.max(np.abs(a-b)/np.maximum(np.abs(b),1e-300))
t=time.perf_counter(); p=nns.nns_norm(P11.to_numpy()); tp=(time.perf_counter()-t)*1000
print("p11 nonlinear rel diff %.2e  time py %.0f ms R %.0f ms"%(rel(p,r["p11_nl"]), tp, r["t11nl"][0]))
print("p11 linear rel diff %.2e"%rel(nns.nns_norm(P11.to_numpy(),linear=True), r["p11_l"]))
t=time.perf_counter(); p=nns.nns_norm(P5.to_numpy()); tp=(time.perf_counter()-t)*1000
print("p5 nonlinear rel diff %.2e time py %.1f R %.1f"%(rel(p,r["p5_nl"]),tp,r["t5nl"][0]))
print("p5 linear rel diff %.2e"%rel(nns.nns_norm(P5.to_numpy(),linear=True), r["p5_l"]))
lu=nns.nns_norm([np.array([1,2,3,4,5,6,7.]),np.array([10,20,30,40,50,60.]),np.array([.5,.6,.7,.8,.9,1,1.1,1.2,1.3])])
print("list uneq", [rel(a,b) for a,b in zip(lu, r["list_uneq"].values() if isinstance(r["list_uneq"],dict) else r["list_uneq"])])
print("list eq rel %.2e"%rel(nns.nns_norm([P5.iloc[:,0].values,P5.iloc[:,1].values,P5.iloc[:,2].values]), r["list_eq"]))
print("zero mean", nns.nns_norm(np.array([[-1,1],[0,2],[1,3.]])).tolist(), r["zero_mean"])
# dep matrix compare
from nns.norm import _scale_factor
d=_scale_factor(P11.to_numpy()); print("dep11 max abs diff %.2e"%np.max(np.abs(d-np.array(r["dep11"]))))
print("R names", r["names11"][:3])
for lab,f in [("DataFrame", lambda: nns.nns_norm(P5)), ("pandas list of Series", lambda: nns.nns_norm([P5.SPY,P5.TLT])),
              ("NaN", lambda: nns.nns_norm(np.array([[1,np.nan],[2,3.]]))), ("1 col", lambda: nns.nns_norm(P5.to_numpy()[:,:1])[:2]),
              ("const col", lambda: nns.nns_norm(np.c_[np.ones(5),np.arange(5.)])) ]:
    try: print(lab, "->", type(f()).__name__, np.asarray(f())[:2] if not isinstance(f(),list) else "list")
    except Exception as e: print(lab,"-> EXC",type(e).__name__,e)
