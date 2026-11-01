# Test: does column-sequential accumulation reproduce R's %*% exactly?
import numpy as np, pandas as pd, nns.stack as S
from nns._rrng import RRNG
X=pd.read_csv("cases/boston_m12_pi/xtr.csv").to_numpy(float); y=pd.read_csv("cases/boston_m12_pi/ytr.csv").iloc[:,0].to_numpy()
captured=[]
orig=S.nns_reg_engine
import nns._reg_engine as E
P0=S._mreg_prepare
def P(x,yy,*a):
    captured.append(x.copy()); return P0(x,yy,*a)
S._mreg_prepare=P
# monkeypatch matmul in project via replacing np.ndarray @? Instead recompute manually below.
r=S.nns_stack(X,y,pd.read_csv("cases/boston_m12_pi/xte.csv").to_numpy(float),obj_fn=lambda p,a: float(np.sqrt(np.mean((p-a)**2))))
for i,c in enumerate(captured[:5],1):
    rt=np.loadtxt(f"dump/rtr{i}.txt")
    print(i, "numpy@ vs R: n mismatched =", int(np.sum(c[:,0]!=rt)))
