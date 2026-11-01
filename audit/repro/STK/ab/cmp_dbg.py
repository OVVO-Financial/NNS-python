import sys, json, numpy as np
import stack_patched as S, py_stack, compare
py_stack.nns_stack=S.nns_stack
name=sys.argv[1]; import subprocess; subprocess.run(["Rscript","r_dbg2.R",name],check=True)
spec=json.load(open(f"cases/{name}/spec.json")); m={"type":"type","balance":"balance","optimize.threshold":"optimize_threshold","dim.red.method":"dim_red_method"}
spec["pyargs"]={m[k]:v for k,v in spec["args"].items()}
P=py_stack.run(spec); D=S.DBG; R=json.load(open(f"cases/{name}/rdbg.json"))
for i,(rv,pv) in enumerate(zip(R["splits"], D["splits"])): print("fold",i+1,"valid same:", np.array_equal(np.array(rv)-1, pv["validation"]))
for i,(rb,pb) in enumerate(zip(R["bal"], D.get("bal",[]))): print("bal",i+1,"same:", np.array_equal(np.array(rb)-1, pb))
rs=np.array(R["dim_sum"]); ps=D["dim_sum"]; print("dim_sum maxdiff per m:", np.abs(rs-ps).max(0))
for k in D["m1sum"]:
    d=np.abs(np.array(R["m1sum"][k])-D["m1sum"][k]).max()
    if d>1e-12: print("m1 cand",k,"diff",d, "rows", np.flatnonzero(np.abs(np.array(R["m1sum"][k])-D["m1sum"][k])>1e-12)[:10])
for m in (1,2,3):
    ru=R["uni"][m-1]; pu=D["uni"][m]
    for k in ("xtr","ytr","xte","raw"):
        a=np.array(ru[k]); b=pu[k]; print(m,k, "n", a.size, b.size, "bitequal", np.array_equal(a,b), "maxdiff", np.abs(a-b).max() if a.size==b.size else None)
import os
os.makedirs("uni",exist_ok=True)
pu=D["uni"][3]; np.savetxt("uni/xtr.txt",pu["xtr"],fmt="%.17g"); np.savetxt("uni/ytr.txt",pu["ytr"],fmt="%.17g"); np.savetxt("uni/xte.txt",pu["xte"],fmt="%.17g")
np.savetxt("uni/py_raw.txt",pu["raw"],fmt="%.17g"); np.savetxt("uni/r_raw.txt",np.array(R["uni"][2]["raw"]),fmt="%.17g")
