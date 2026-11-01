import numpy as np, json, nns
r=json.load(open("eff/r.json")); bad=0
for s,rr in r.items():
    X=np.loadtxt(f"eff/m{s}.csv",delimiter=",",ndmin=2)
    for key,(d,t) in {"d1":(1,"discrete"),"d2":(2,"discrete"),"d3":(3,"discrete"),"d1c":(1,"continuous")}.items():
        p=nns.sd_efficient_set(X,d,type=t)
        if p!=rr[key]: bad+=1; print("MISMATCH m",s,X.shape,key,"py",p,"R",rr[key])
print("mismatches",bad,"of",len(r)*4)
