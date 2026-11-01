import numpy as np, pandas as pd
from nns.stack import _spearman_signed
X=pd.read_csv("cases/boston_m12_pi/xtr.csv").to_numpy(float); y=pd.read_csv("cases/boston_m12_pi/ytr.csv").iloc[:,0].to_numpy()
mn=X.min(0); rg=X.max(0)-mn; D=(X-mn)/rg
rD=np.loadtxt("dump/rD.txt").reshape(13,-1).T; print("D bit-equal:", np.array_equal(D,rD))
c=_spearman_signed(D,y); rc=np.loadtxt("dump/rcor.txt"); print("cor bit-equal per col:", (c==rc).astype(int), np.abs(c-rc).max())
a=np.zeros(13); o=np.argsort(-np.abs(rc),kind="stable")[:3]; a[o]=rc[o]
rp=np.loadtxt("dump/rproj.txt")
acc=np.zeros(len(y))
for j in range(13): acc=acc+D[:,j]*a[j]
print("np@ equal:", np.sum((D@a/3)!=rp), " sequential equal:", np.sum(acc/3!=rp), "of", len(y))
