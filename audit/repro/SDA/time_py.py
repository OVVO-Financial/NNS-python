import time, numpy as np, pandas as pd, nns, os
def tm(f, reps):
    f(); t=time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter()-t)/reps*1000
rng=np.random.default_rng(1)
for n in (1000,10000,30000):
    y=rng.normal(0,1,n); x=y+0.1; reps=3 if n==100000 else 20
    print(f"n={n} fsd {tm(lambda: nns.fsd(x,y),reps):.2f} ssd {tm(lambda: nns.ssd(x,y),reps):.2f} tsd {tm(lambda: nns.tsd(x,y),reps):.2f} | fsd_uni {tm(lambda: nns.fsd_uni(x,y),reps):.2f} fsd_uni_c {(tm(lambda: nns.fsd_uni(x,y,type='continuous'),1) if n<=10000 else float('nan')):.2f} ssd_uni {tm(lambda: nns.ssd_uni(x,y),reps):.2f} tsd_uni {tm(lambda: nns.tsd_uni(x,y),reps):.2f} ms")
X=pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv").iloc[:,1:].to_numpy()
for d in (1,2,3): print(f"eff set 11 assets n={X.shape[0]} deg {d}: {tm(lambda: nns.sd_efficient_set(X,d),10):.2f} ms")
print(f"eff set deg1 continuous: {tm(lambda: nns.sd_efficient_set(X,1,type='continuous'),10):.2f} ms")
print(f"sd_cluster deg 2: {tm(lambda: nns.nns_sd_cluster(X,degree=2),5):.2f} ms")
Y=pd.read_csv("Y200.csv").to_numpy()
for d in (1,2,3): print(f"eff set 200 cols n=1000 deg {d}: {tm(lambda: nns.sd_efficient_set(Y,d),2):.2f} ms")
print(f"eff set 200 cols continuous: {tm(lambda: nns.sd_efficient_set(Y,1,type='continuous'),2):.2f} ms")
