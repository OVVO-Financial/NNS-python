import numpy as np, pandas as pd, json, itertools, nns
cases = ["indep_n100","indep_n1000","y_x2","discrete_dep","ties_heavy","step","len5","len3","int_seq","ret_SPY_TLT","circle"]
out = {}
for k in cases:
    d = pd.read_csv(f"/home/user/audit/work/DEP/cases/{k}.csv"); x=d.x.values; y=d.y.values
    for typ, order, obs, mos, nr in itertools.product([None,"XONLY"],[None,1,3,"max"],[0,8,50],[True,False],["off","mean","median","mode","mode_class"]):
        key = f"{k}|{typ}|{order}|{obs}|{mos}|{nr}"
        try:
            r = nns.nns_part(x,y,type=typ,order=order,obs_req=obs,min_obs_stop=mos,noise_reduction=nr)
            rp = r["regression.points"]
            out[key] = {"order": int(r["order"]), "q": list(map(str,r["dt"]["quadrant"])), "pq": list(map(str,r["dt"]["prior.quadrant"])),
                        "rpq": list(map(str,rp["quadrant"])), "rpx": [float(v) for v in rp["x"]], "rpy": [float(v) for v in rp["y"]]}
        except Exception as e:
            out[key] = f"ERR {e}"
json.dump(out, open("/home/user/audit/work/DEP/part_py.json","w"))
print(len(out))
