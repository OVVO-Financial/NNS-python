import json, numpy as np, nns
from collections import Counter, defaultdict
cases = json.load(open("sd_cases.json")); rres = json.load(open("sd_cases_r.json"))
def ecdf(v, t): v=np.sort(v); return np.searchsorted(v, t, side="right")/v.size
def lpm1(v,t): return np.mean(np.maximum(t[:,None]-v[None,:],0),axis=1)
def lpm2(v,t): return np.mean(np.maximum(t[:,None]-v[None,:],0)**2,axis=1)
TOL=1e-12
def fp_dom(x,y,deg):
    """first principles: x dominates y at degree deg (with small tolerance)"""
    g = np.sort(np.unique(np.concatenate([x,y])))
    if deg==1:
        fx,fy = ecdf(x,g), ecdf(y,g)
        return bool(np.all(fx<=fy+TOL) and np.any(fx<fy-TOL))
    if deg==2:
        fx,fy = lpm1(x,g), lpm1(y,g)
        return bool(np.all(fx<=fy+TOL) and np.any(fx<fy-TOL))
    # deg3: dense grid incl. interior points + mean condition
    lo,hi = g[0], g[-1]
    dense = np.unique(np.concatenate([g, np.linspace(lo,hi,4001), (g[:-1]+g[1:])/2] if g.size>1 else [g]))
    fx,fy = lpm2(x,dense), lpm2(y,dense)
    return bool(np.all(fx<=fy+TOL) and np.any(fx<fy-TOL) and np.mean(x)>=np.mean(y)-TOL)
def fp_bi(x,y,deg):
    return 1 if fp_dom(x,y,deg) else (-1 if fp_dom(y,x,deg) else 0)
stats=defaultdict(Counter); mism=defaultdict(list)
for cs, rr in zip(cases, rres):
    x=np.array(cs["x"]); y=np.array(cs["y"]); nm=cs["name"]; import re; grp=re.sub(r"_s\d+$","",nm)
    py = {"FSD":nns.fsd(x,y),"SSD":nns.ssd(x,y),"TSD":nns.tsd(x,y),
          "FSDu_xy":nns.fsd_uni(x,y),"FSDu_yx":nns.fsd_uni(y,x),
          "FSDuc_xy":nns.fsd_uni(x,y,type="continuous"),"FSDuc_yx":nns.fsd_uni(y,x,type="continuous"),
          "SSDu_xy":nns.ssd_uni(x,y),"SSDu_yx":nns.ssd_uni(y,x),"TSDu_xy":nns.tsd_uni(x,y),"TSDu_yx":nns.tsd_uni(y,x)}
    fp = {"FSD":fp_bi(x,y,1),"SSD":fp_bi(x,y,2),"TSD":fp_bi(x,y,3)}
    for k,v in py.items():
        if rr[k]!=v: mism[f"py!=R {k}"].append((nm, v, rr[k]))
    for k,v in fp.items():
        if py[k]!=v: mism[f"py!=FP {k}"].append((nm, py[k], v, len(x), len(y)))
        if rr[k]!=v: mism[f"R!=FP {k}"].append((nm, rr[k], v))
    for k in ("FSD","SSD","TSD"): stats[grp][f"{k}={py[k]}"]+=1
for g in ["qnorm_shift_n10","qnorm_shift_n100","qnorm_shift_n1000","qnorm_mps_n10","qnorm_mps_n100","qnorm_mps_n1000","qnorm_perm_n100","rnorm_shift100","rnorm_shift1000","int_mps","same_dist_diff_len","same_dist_diff_len_rev","const_vs_spread","const_equal","len1","len1_eq","float_mps","float_mps2","float_mps3"]:
    print(g, dict(stats[g]))
for k,v in mism.items():
    print("==",k,len(v)); print("  ",v[:8])
