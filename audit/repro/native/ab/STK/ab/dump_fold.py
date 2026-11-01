import numpy as np, pandas as pd, nns.stack as S
from nns._reg_engine import _mreg_prepare as P0, _mreg_predict_path as Q0
calls=[]
def P(x,y,*a):
    r=P0(x,y,*a); calls.append(("prep",x.copy(),y.copy(),r)); return r
def Q(xt,rx,ry,k,d,mn,mx):
    r=Q0(xt,rx,ry,k,d,mn,mx); calls.append(("path",xt.copy(),r)); return r
S._mreg_prepare=P; S._mreg_predict_path=Q
X=pd.read_csv("cases/boston_m12_pi/xtr.csv").to_numpy(float); y=pd.read_csv("cases/boston_m12_pi/ytr.csv").iloc[:,0].to_numpy()
Xt=pd.read_csv("cases/boston_m12_pi/xte.csv").to_numpy(float)
r=S.nns_stack(X,y,Xt,obj_fn=lambda p,a: float(np.sqrt(np.mean((p-a)**2))))
c=calls[0]; np.savetxt("dump/f1_x.csv",c[1],delimiter=","); np.savetxt("dump/f1_y.csv",c[2],delimiter=",")
np.savetxt("dump/f1_rpmx.csv",c[3][0],delimiter=","); np.savetxt("dump/f1_rpmy.csv",c[3][1],delimiter=",")
np.savetxt("dump/f1_xt.csv",calls[1][1],delimiter=","); np.savetxt("dump/f1_path.csv",calls[1][2],delimiter=",")
print(c[1].shape, c[3][0].shape, calls[1][2].shape)
k=0
for c in calls:
    if c[0]=="path":
        k+=1; np.savetxt(f"dump/path{k}.csv", c[2], delimiter=","); np.savetxt(f"dump/xt{k}.csv", c[1], delimiter=",")
print("npaths",k)
k=0
for c in calls:
    if c[0]=="prep":
        k+=1; np.savetxt(f"dump/prx{k}.csv", c[3][0], delimiter=","); np.savetxt(f"dump/pry{k}.csv", c[3][1], delimiter=","); np.savetxt(f"dump/ptr{k}.csv", c[1], delimiter=",")
