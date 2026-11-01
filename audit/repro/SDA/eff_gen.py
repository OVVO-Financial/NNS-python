import numpy as np, json
rng=np.random.default_rng(3); meta=[]
for s in range(40):
    kind = s%4
    ncol = [5,20,80,120][s%4]; nrow=int(rng.integers(5,40))
    if s%3==0: X=rng.integers(0,6,(nrow,ncol)).astype(float)
    elif s%3==1: X=np.round(rng.normal(rng.normal(0,.3,ncol), rng.uniform(.5,2,ncol),(nrow,ncol)),2)
    else: X=rng.standard_t(3,(nrow,ncol))*0.01+rng.normal(0,0.002,ncol)
    np.savetxt(f"eff/m{s}.csv", X, delimiter=",")
    meta.append(s)
json.dump(meta, open("eff/meta.json","w"))
