import numpy as np, nns, time
rng = np.random.default_rng(7)
print("n, dist, mean Dep, sd Dep, mean |Corr|, mean copula, mean Pearson|r|")
for dist in ["normal", "exponential", "t3"]:
    for n in [30, 100, 300, 1000, 3000, 10000, 100000]:
        reps = 40 if n <= 10000 else 8
        D=[];C=[];K=[];P=[]
        for _ in range(reps):
            if dist=="normal": x,y = rng.normal(size=n), rng.normal(size=n)
            elif dist=="exponential": x,y = rng.exponential(size=n), rng.exponential(size=n)
            else: x,y = rng.standard_t(3,size=n), rng.standard_t(3,size=n)
            r = nns.nns_dep(x,y); D.append(r["Dependence"]); C.append(abs(r["Correlation"]))
            K.append(nns.nns_copula(x,y)); P.append(abs(np.corrcoef(x,y)[0,1]))
        print(f"{n:6d} {dist:12s} {np.mean(D):.3f} {np.std(D):.3f} {np.mean(C):.3f} {np.mean(K):.3f} {np.mean(P):.3f}")
