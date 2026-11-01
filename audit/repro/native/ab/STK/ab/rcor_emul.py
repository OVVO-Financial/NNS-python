"""Emulation of R's stats:::C_cor (cov.c, complete-case path, LDOUBLE accumulators)."""
import numpy as np
from nns.stack import _rank_average
LD=np.longdouble
def r_mean(x):
    n=x.size; s=LD(0)
    for v in x: s+=LD(v)
    t=s/LD(n)
    if np.isfinite(float(t)):
        s=LD(0)
        for v in x: s+=LD(v)-t
        t=t+s/LD(n)
    return float(t)
def r_cor(x,y):
    n=x.size; n1=LD(n-1); xm=r_mean(x); ym=r_mean(y)
    dx=x-xm; dy=y-ym  # double
    s=LD(0)
    for p in dx*dy: s+=LD(p)
    cov=float(s/n1)
    sx=LD(0)
    for p in dx*dx: sx+=LD(p)
    sy=LD(0)
    for p in dy*dy: sy+=LD(p)
    sdx=float(np.sqrt(sx/n1)); sdy=float(np.sqrt(sy/n1))
    if sdx==0 or sdy==0: return float("nan")
    return float(min(1.0,max(-1.0, cov/(sdx*sdy))))
def r_spearman(x,y): return r_cor(_rank_average(x), _rank_average(y))
if __name__=="__main__":
    import pandas as pd
    X=pd.read_csv("cases/boston_m12_pi/xtr.csv").to_numpy(float); y=pd.read_csv("cases/boston_m12_pi/ytr.csv").iloc[:,0].to_numpy()
    mn=X.min(0); rg=X.max(0)-mn; D=(X-mn)/rg
    c=np.array([r_spearman(D[:,j],y) for j in range(13)]); rc=np.loadtxt("dump/rcor.txt")
    print("emulated cor bit-equal:", (c==rc).astype(int))
