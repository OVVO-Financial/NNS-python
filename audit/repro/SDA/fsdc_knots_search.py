import numpy as np, nns
rng=np.random.default_rng(11)
def ratio(v,t):
    L=np.mean(np.maximum(t[:,None]-v,0),1); U=np.mean(np.maximum(v-t[:,None],0),1)
    with np.errstate(invalid="ignore"): r=L/(L+U)
    return np.where(L+U>0, r, 0.0)
cnt=0; pos=0; ex=None
for it in range(20000):
    n=int(rng.integers(2,7)); x=np.round(rng.normal(0.5,1,n),2); y=np.round(rng.normal(0,1,n),2)
    if np.ptp(x)==0 or np.ptp(y)==0: continue
    if nns.fsd_uni(x,y,type="continuous"):
        pos+=1
        t=np.linspace(min(x.min(),y.min()), max(x.max(),y.max()), 20001)
        d=ratio(x,t)-ratio(y,t)
        if d.max()>1e-9:
            cnt+=1; ex=ex or (x.tolist(),y.tolist(),float(d.max()),float(t[d.argmax()]))
print("positives",pos,"knot-only false positives",cnt, ex)
