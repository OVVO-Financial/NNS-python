import json, subprocess, os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import OBJ, D
from nns import nns_boost
def bcase(name, Xtr, ytr, Xte=None, rargs=None, pyargs=None, factor_cols=None, yfactor=False, obj=None, pyconv=None):
    p=os.path.join(D,name); os.makedirs(p,exist_ok=True)
    pd.DataFrame(Xtr).to_csv(f"{p}/xtr.csv",index=False); pd.DataFrame({"y":ytr}).to_csv(f"{p}/ytr.csv",index=False)
    spec=dict(xtrain=f"{p}/xtr.csv", ytrain=f"{p}/ytr.csv", factor_cols=factor_cols or [], yfactor=yfactor, args=dict(rargs or {}))
    if Xte is not None: pd.DataFrame(Xte).to_csv(f"{p}/xte.csv",index=False); spec["xtest"]=f"{p}/xte.csv"
    if obj: spec["args"]["obj.fn"]=OBJ[obj][0]
    json.dump(spec,open(f"{p}/spec.json","w"))
    rr=subprocess.run(["Rscript", os.path.join(os.path.dirname(os.path.abspath(__file__)),"r_boost.R"), f"{p}/spec.json", f"{p}/r.json"],capture_output=True,text=True)
    R=json.load(open(f"{p}/r.json")) if rr.returncode==0 else {"error":rr.stderr[-600:]}
    kw=dict(pyargs or {});
    if obj: kw["obj_fn"]=OBJ[obj][1]
    Xp = pyconv(Xtr) if pyconv else np.asarray(Xtr, float); Xtp = (pyconv(Xte) if pyconv else np.asarray(Xte,float)) if Xte is not None else None
    yp = np.asarray(ytr).astype(str) if yfactor else np.asarray(ytr)
    t0=time.time()
    try: P=nns_boost(Xp, yp, Xtp, **kw); P["secs"]=time.time()-t0
    except Exception as e: P={"error":repr(e)}
    print("===", name)
    if "error" in R or "error" in P: print(" R:", R.get("error","ok"), "\n PY:", P.get("error","ok")); return R,P
    a=np.asarray(R["results"],float); b=np.asarray(P["results"],float)
    print(f"  results maxabsdiff={np.max(np.abs(a-b)):.3g} R[:4]={a[:4]} PY[:4]={b[:4]}")
    print("  R freq:", R["feature.frequency"]); print("  PY freq:", P["feature.frequency"])
    if R.get("pred.int") is not None or P.get("pred.int") is not None:
        print("  R pi[:2]:", str(R.get("pred.int"))[:150]); print("  PY pi:", {k: np.asarray(v)[:2] for k,v in (P.get("pred.int") or {}).items()})
    print(f"  secs R={R['secs']:.2f} PY={P['secs']:.2f}")
    return R,P
