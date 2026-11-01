import numpy as np, pandas as pd, nns, itertools
def tsd_exact_worst(x,y):
    """max over all real t of LPM2_x(t)-LPM2_y(t) (exact, piecewise quadratic)."""
    x=np.sort(x); y=np.sort(y); nx,ny=x.size,y.size
    g=np.unique(np.r_[x,y])
    def coef(v,n,t):
        k=np.searchsorted(v,t,side="right"); c=np.r_[0,np.cumsum(v)]; q=np.r_[0,np.cumsum(v*v)]
        return k/n, c[k]/n, q[k]/n
    ax,sx,qx=coef(x,nx,g); ay,sy,qy=coef(y,ny,g)
    a=ax-ay; b=-2*(sx-sy); c=qx-qy
    D=lambda t,i: a[i]*t*t+b[i]*t+c[i]
    idx=np.arange(g.size)
    worst=np.max(D(g,idx))
    with np.errstate(divide="ignore",invalid="ignore"):
        tv=-b/(2*a)
    ok=(a<0)&(idx<g.size-1)
    ok&= (tv>g)&(tv<np.r_[g[1:],np.inf])
    if ok.any(): worst=max(worst, np.max(D(tv[ok],idx[ok])))
    # scale reference
    return worst
if __name__=="__main__":
    R=pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv").iloc[:,1:]
    X=R.to_numpy(); nm=list(R.columns)
    for i,j in itertools.permutations(range(X.shape[1]),2):
        if nns.tsd_uni(X[:,i],X[:,j]):
            w=tsd_exact_worst(X[:,i],X[:,j])
            print(nm[i],"TSD",nm[j],"worst interior excess %.3e"%w, "FALSE POSITIVE" if w>1e-15 else "")
    # rate among random small samples
    rng=np.random.default_rng(7); pos=fp=0
    for it in range(50000):
        n=int(rng.integers(3,30)); x=rng.normal(0.3,1,n); y=rng.normal(0,1.3,n)
        if nns.tsd_uni(x,y):
            pos+=1; fp+= tsd_exact_worst(x,y)>1e-12
    print("random normal pairs: TSD positives",pos,"false positives",fp)
