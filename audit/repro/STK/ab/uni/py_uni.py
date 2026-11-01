import numpy as np, pandas as pd
from nns._reg_engine import nns_reg_engine
from nns import nns_reg
x=np.loadtxt("uni/xtr.txt"); y=np.loadtxt("uni/ytr.txt"); xt=np.loadtxt("uni/xte.txt")
a=np.asarray(nns_reg_engine(x,y,point_est=xt,point_only=True,type=None,factor_2_dummy=False)["Point.est"])
full=nns_reg(x,y,point_est=xt)
b=np.asarray(full["Point.est"]); rf=np.loadtxt("uni/r_fast.txt"); rr=np.loadtxt("uni/r_reg.txt")
print("py engine vs R fast:", np.abs(a-rf).max(), " py nns_reg vs R NNS.reg:", np.abs(b-rr).max())
rp=pd.read_csv("uni/r_rp.csv"); prp=full["regression.points"]
print("rp rows R", len(rp), "py", len(prp["x"]) if isinstance(prp,dict) else len(prp))
px=np.asarray(prp["x"]); py_=np.asarray(prp["y"])
if len(px)==len(rp): print("rp x diff", np.abs(px-rp.x).max(), "rp y diff", np.abs(py_-rp.y).max())
else:
    print("R x", rp.x.values[:20]); print("P x", px[:20])
