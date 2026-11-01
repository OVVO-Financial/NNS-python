import json, numpy as np, collections
P=json.load(open("part_py.json")); R=json.load(open("part_r.json"))
bad=collections.Counter(); ex={}
for k in P:
    p=P[k]; r=R.get(k)
    if isinstance(p,str) or isinstance(r,(str,list)) and not isinstance(r,dict):
        rr = r[0] if isinstance(r,list) else r
        if not (isinstance(p,str) and isinstance(rr,str)):
            bad["err-mismatch"]+=1; ex.setdefault("err-mismatch",[]).append((k,p if isinstance(p,str) else "ok", rr if isinstance(rr,str) else "ok"))
        continue
    def g(f): v=r[f]; return v[0] if f=="order" else v
    issues=[]
    if p["order"]!=g("order"): issues.append("order")
    if p["q"]!=g("q"): issues.append("q")
    if p["pq"]!=g("pq"): issues.append("pq")
    if p["rpq"]!=g("rpq"): issues.append("rpq")
    else:
        for f in ["rpx","rpy"]:
            a=np.array(p[f],float); b=np.array([np.nan if v in (None,"NA","NaN") else v for v in g(f)],float)
            if a.shape!=b.shape or not np.allclose(a,b,rtol=1e-9,atol=1e-12,equal_nan=True): issues.append(f)
    for i in issues: bad[i]+=1; ex.setdefault(i,[]).append(k)
print(bad)
for i,v in ex.items(): print(i, len(v), v[:12])
