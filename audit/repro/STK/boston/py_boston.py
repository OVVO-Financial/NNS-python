import json, time, numpy as np, pandas as pd
from nns import nns_stack
df = pd.read_csv("boston_full.csv"); tr = np.loadtxt("train_idx.txt", dtype=int)
te = np.setdiff1d(np.arange(len(df)), tr)
X = df.iloc[:, :13].to_numpy(float); y = df["medv"].to_numpy(float)
rmse = lambda p,a: float(np.sqrt(np.mean((np.asarray(p)-a)**2)))
def run(**kw):
    t0=time.time()
    r = nns_stack(X[tr], y[tr], X[te], obj_fn=rmse, objective="min", **kw)
    return dict(reg=rmse(r["reg"],y[te]), dim=rmse(r["dim.red"],y[te]), stack=rmse(r["stack"],y[te]),
        nbest=r["NNS.reg.n.best"], thr=r["NNS.dim.red.threshold"], w=r["weights"], objreg=r["OBJfn.reg"],
        objdim=r["OBJfn.dim.red"], vars=r["NNS.dim.red.variables"].tolist(), head=np.asarray(r["stack"][:5]).tolist(), secs=time.time()-t0)
out = dict(default=run(), cv25=run(cv_size=0.25))
R = json.load(open("r_boston.json"))
for k in out:
    for f in out[k]:
        print(k, f, "PY", out[k][f], "R", R[k][f])
