import json, sys, time, numpy as np, pandas as pd
from nns import nns_stack
def run(spec):
    Xtr = pd.read_csv(spec["xtrain"]); ytr = pd.read_csv(spec["ytrain"]).iloc[:,0].to_numpy()
    Xte = pd.read_csv(spec["xtest"]) if spec.get("xtest") else None
    fc = [j-1 for j in spec.get("factor_cols", [])]
    def conv(df):
        if df is None: return None
        if fc:
            a = df.to_numpy(object)
            for j in fc: a[:, j] = a[:, j].astype(str)
            return a
        return df.to_numpy(float)
    if spec.get("yfactor"): ytr = ytr.astype(str)
    kw = dict(spec.get("pyargs", {}))
    t0=time.time()
    r = nns_stack(conv(Xtr), ytr, conv(Xte), **kw)
    r["secs"]=time.time()-t0
    return r
if __name__=="__main__":
    spec=json.load(open(sys.argv[1])); r=run(spec)
    def enc(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, dict): return {k: enc(v) for k,v in o.items()}
        if isinstance(o, list): return [enc(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o): return None
        return o
    json.dump(enc(r), open(sys.argv[2],"w"))
