import numpy as np
from nns import nns_reg, nns_dep
x=np.loadtxt("uni/xtr.txt"); y=np.loadtxt("uni/ytr.txt")
for o in (None,1,2,3,4,5,"max"):
    r=nns_reg(x,y,order=o); print("order",o,"rp",len(np.asarray(r["regression.points"]["x"])))
print("dep:", nns_dep(x,y)["Dependence"])
