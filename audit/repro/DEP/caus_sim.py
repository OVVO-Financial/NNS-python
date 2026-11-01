import numpy as np, nns, pandas as pd, os
rng = np.random.default_rng(11)
os.makedirs("/home/user/audit/work/DEP/caus_cases", exist_ok=True)
def direction(r):
    k = [k for k in r if k.startswith("C(")][0]
    return ("x->y" if k=="C(x--->y)" else "y->x"), r[k]
n = 500; reps = 30
designs = {}
def lag_iid():
    x = rng.normal(size=n+1); y = 0.8*x[:-1] + 0.3*rng.normal(size=n); return x[1:], y
def lag_ar():
    x = np.zeros(n+1); e=rng.normal(size=n+1)
    for t in range(1,n+1): x[t]=0.5*x[t-1]+e[t]
    y = 0.8*x[:-1] + 0.3*rng.normal(size=n); return x[1:], y
def lag_nonlin():
    x = rng.normal(size=n+1); y = x[:-1]**2 + 0.3*rng.normal(size=n); return x[1:], y
def contemp_x2():
    x = rng.normal(size=n); return x, x**2
def contemp_lin():
    x = rng.normal(size=n); return x, 0.8*x + 0.6*rng.normal(size=n)
def indep():
    return rng.normal(size=n), rng.normal(size=n)
designs = dict(lag_iid=lag_iid, lag_ar=lag_ar, lag_nonlin=lag_nonlin, contemp_x2=contemp_x2, contemp_lin=contemp_lin, indep=indep)
rows=[]
for name, f in designs.items():
    for rep in range(reps):
        x, y = f()
        if rep < 3: pd.DataFrame({"x":x,"y":y}).to_csv(f"/home/user/audit/work/DEP/caus_cases/{name}_{rep}.csv", index=False)
        for tau in [0, 1, 3]:
            r = nns.nns_causation(x, y, tau=tau); d, v = direction(r)
            rows.append((name, tau, d, v, r["Causation.x.given.y"], r["Causation.y.given.x"]))
            # also swapped argument order
            r2 = nns.nns_causation(y, x, tau=tau); d2, v2 = direction(r2)
            rows.append((name+"_swapped", tau, d2, v2, r2["Causation.x.given.y"], r2["Causation.y.given.x"]))
df = pd.DataFrame(rows, columns=["design","tau","dir","net","cxgy","cygx"])
s = df.groupby(["design","tau"]).agg(frac_x_to_y=("dir", lambda d: np.mean(d=="x->y")), med_net=("net","median"), mean_cxgy=("cxgy","mean"), mean_cygx=("cygx","mean"))
print(s.to_string())
