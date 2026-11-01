import numpy as np, nns, json
rng=np.random.default_rng(5)
def L2(v,t): return np.mean(np.maximum(t-v,0)**2)
def exact_tsd(x,y):
    g=np.unique(np.r_[x,y]); worst=-np.inf
    for t in g: worst=max(worst, L2(x,t)-L2(y,t))
    for a,b in zip(g[:-1],g[1:]):
        # D(t) quadratic on (a,b): find vertex numerically via 3 points
        ts=np.array([a,(a+b)/2,b]); ds=np.array([L2(x,t)-L2(y,t) for t in ts])
        A=np.polyfit(ts,ds,2)
        if A[0]<0:
            tv=-A[1]/(2*A[0])
            if a<tv<b: worst=max(worst, L2(x,tv)-L2(y,tv))
    return worst
found=[]
for it in range(20000):
    n=int(rng.integers(2,6)); x=np.round(rng.normal(0.3,1,n),2); y=np.round(rng.normal(0,1.3,n),2)
    if nns.tsd_uni(x,y)==1:
        w=exact_tsd(x,y)
        if w>1e-12: found.append((x.tolist(),y.tolist(),w))
print(len(found)); print(found[:5])
json.dump(found[:20], open("tsd_knots_counterex.json","w"))
