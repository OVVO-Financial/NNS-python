import json, subprocess, sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import py_stack
OBJ = {
 "rmse": ("sqrt(mean((predicted - actual)^2))", lambda p,a: float(np.sqrt(np.mean((p-a)**2)))),
 "mae": ("mean(abs(predicted - actual))", lambda p,a: float(np.mean(np.abs(p-a)))),
 "acc": ("mean(predicted == actual)", lambda p,a: float(np.mean(p==a))),
 "negsse": ("-sum((predicted - actual)^2)", lambda p,a: float(-np.sum((p-a)**2))),
}
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")
os.makedirs(D, exist_ok=True)
def case(name, Xtr, ytr, Xte=None, rargs=None, pyargs=None, factor_cols=None, yfactor=False, obj=None):
    p = os.path.join(D, name); os.makedirs(p, exist_ok=True)
    pd.DataFrame(Xtr).to_csv(f"{p}/xtr.csv", index=False)
    pd.DataFrame({"y": ytr}).to_csv(f"{p}/ytr.csv", index=False)
    spec = dict(xtrain=f"{p}/xtr.csv", ytrain=f"{p}/ytr.csv", factor_cols=factor_cols or [], yfactor=yfactor)
    if Xte is not None:
        pd.DataFrame(Xte).to_csv(f"{p}/xte.csv", index=False); spec["xtest"]=f"{p}/xte.csv"
    rargs = dict(rargs or {}); pyargs = dict(pyargs or {})
    spec["args"]=rargs
    if obj: spec["args"]["obj.fn"]=OBJ[obj][0]
    json.dump(spec, open(f"{p}/spec.json","w"))
    rr = subprocess.run(["Rscript", os.path.join(os.path.dirname(__file__),"r_stack.R"), f"{p}/spec.json", f"{p}/r.json"], capture_output=True, text=True)
    R = json.load(open(f"{p}/r.json")) if rr.returncode==0 else {"error": rr.stderr[-800:]}
    spec["pyargs"]=pyargs
    if obj: pyargs["obj_fn"]=OBJ[obj][1]
    try:
        P = py_stack.run(spec)
    except Exception as e:
        P = {"error": repr(e)}
    return R, P
def num(v):
    try: return np.asarray(v, dtype=float).ravel()
    except Exception: return None
def report(name, R, P, keys=("OBJfn.reg","NNS.reg.n.best","OBJfn.dim.red","NNS.dim.red.threshold","probability.threshold","weights","reg","dim.red","stack","pred.int","secs")):
    print(f"=== {name}")
    if "error" in R or "error" in P:
        print("  R:", R.get("error","ok")[-400:] if "error" in R else "ok"); print("  PY:", P.get("error","ok") if "error" in P else "ok"); return
    for k in keys:
        rv, pv = R.get(k), P.get(k)
        if k=="weights": rv=[rv["reg"],rv["dim.red"]] if rv else None; pv=[pv["reg"],pv["dim.red"]]
        if k in ("pred.int",) and rv is not None:
            rv = np.column_stack([rv[c] for c in rv]) if isinstance(rv, dict) else rv
            pv = np.column_stack([pv[c] for c in pv]) if isinstance(pv, dict) else pv
        a, b = num(rv), num(pv)
        if a is None or b is None or a.size!=b.size:
            print(f"  {k}: R={str(rv)[:80]} PY={str(pv)[:80]}"); continue
        if a.size<=3: print(f"  {k}: R={a} PY={b}")
        else:
            d = np.nanmax(np.abs(a-b)) if a.size else 0
            print(f"  {k}: n={a.size} maxabsdiff={d:.3g} R[:3]={a[:3]} PY[:3]={b[:3]}")
