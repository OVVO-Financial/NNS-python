import numpy as np, pandas as pd, time, nns, itertools
def tm(f, reps=3):
    best=1e9
    for _ in range(reps):
        t=time.perf_counter(); f(); best=min(best,time.perf_counter()-t)
    return best
for n in [1000,10000,100000]:
    d=pd.read_csv(f"/home/user/audit/work/DEP/time_{n}.csv"); x=d.x.values; y=d.y.values
    row={"n":n}
    row["dep"]=tm(lambda: nns.nns_dep(x,y))
    row["dep_asym"]=tm(lambda: nns.nns_dep(x,y,asym=True))
    row["copula"]=tm(lambda: nns.nns_copula(x,y))
    row["part_off"]=tm(lambda: nns.nns_part(x,y))
    row["part_xonly"]=tm(lambda: nns.nns_part(x,y,type="XONLY"))
    for nr in ["mean","median","mode"]:
        row["part_"+nr]=tm(lambda: nns.nns_part(x,y,noise_reduction=nr), reps=1)
    row["caus0"]=tm(lambda: nns.nns_causation(x,y,tau=0), reps=1) if n<=10000 else float('nan')
    row["caus3"]=tm(lambda: nns.nns_causation(x,y,tau=3), reps=1) if n<=10000 else float('nan')
    print({k:(round(v,4) if isinstance(v,float) else v) for k,v in row.items()}, flush=True)
M=pd.read_csv("/home/user/audit/work/DEP/time_mat10.csv").values
def depmat():
    k=M.shape[1]
    for i,j in itertools.combinations(range(k),2): nns.nns_dep(M[:,i],M[:,j])
print("dep matrix 10 cols n=5000 (python pair loop):", round(tm(depmat),4))
print("copula 10 cols n=5000:", round(tm(lambda: nns.nns_copula(M)),4))
print("causal_matrix 10 cols n=5000:", round(tm(lambda: nns.causal_matrix(M), reps=1),4))
print("causal_matrix 10 cols n=1000:", round(tm(lambda: nns.causal_matrix(M[:1000]), reps=1),4))
