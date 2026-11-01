import numpy as np, pandas as pd
from nns._rrng import RRNG
from nns.stack import _spearman_signed
X=pd.read_csv("cases/boston_m12_pi/xtr.csv").to_numpy(float); y=pd.read_csv("cases/boston_m12_pi/ytr.csv").iloc[:,0].to_numpy()
n=len(y); rng=RRNG(123); sh=rng.sample(np.arange(n)); fid=np.zeros(n,int); fid[sh]=np.resize(np.arange(1,6),n)
for b in range(1,6):
    tr=np.flatnonzero(fid!=b); xt=X[tr]; mn=xt.min(0); rg=xt.max(0)-mn; D=(xt-mn)/rg
    c=_spearman_signed(D,y[tr]); o=np.argsort(-np.abs(c),kind="stable")[:2]; a=np.zeros(13); a[o]=c[o]
    v_np=D@a/2
    acc=np.zeros(len(tr))
    for j in range(13): acc=acc+D[:,j]*a[j]
    v_seq=acc/2
    rt=np.loadtxt(f"dump/rtr{b}.txt")
    print(b, len(tr), len(rt), "np@ mism", int(np.sum(v_np!=rt)), np.abs(v_np-rt).max(), " seq mism", int(np.sum(v_seq!=rt)), np.abs(v_seq-rt).max())
