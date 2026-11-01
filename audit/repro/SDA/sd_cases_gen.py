import json, numpy as np
from scipy.stats import norm
rng = np.random.default_rng(1)
cases = []
def add(name, x, y): cases.append({"name": name, "x": list(map(float,x)), "y": list(map(float,y))})
for n in (10, 100, 1000):
    p = (np.arange(1,n+1)-0.5)/n
    add(f"qnorm_shift_n{n}", norm.ppf(p, 0.5, 1), norm.ppf(p, 0, 1))
    add(f"qnorm_mps_n{n}", norm.ppf(p,0,1), norm.ppf(p,0,2))      # x SSD y expected
    add(f"qnorm_perm_n{n}", norm.ppf(p), rng.permutation(norm.ppf(p)))
for s in range(100):
    add(f"rnorm_shift100_s{s}", rng.normal(0.5,1,100), rng.normal(0,1,100))
for s in range(30):
    add(f"rnorm_shift1000_s{s}", rng.normal(0.5,1,1000), rng.normal(0,1,1000))
# discrete MPS: y = x +/- 1 pairs
for s in range(50):
    base = rng.integers(0,5,20).astype(float)
    eps = np.where(rng.random(20)<0.5,-1.0,1.0); eps[10:] = -eps[:10]
    add(f"int_mps_s{s}", base, base+eps)
# random small integer samples, equal length
for s in range(400):
    n = int(rng.integers(1,8))
    add(f"int_eq_s{s}", rng.integers(0,4,n), rng.integers(0,4,n))
# unequal lengths
for s in range(200):
    add(f"int_uneq_s{s}", rng.integers(0,4,int(rng.integers(1,8))), rng.integers(0,4,int(rng.integers(1,8))))
add("same_dist_diff_len", [1,2],[1,1,2,2])
add("same_dist_diff_len_rev", [1,1,2,2],[1,2])
add("const_vs_spread", [1,1,1,1,1],[0,1,2,3,4])
add("const_equal", [1,1,1],[1,1,1])
add("len1", [1],[0])
add("len1_eq", [1],[1])
add("float_mps", [0.1,0.2,0.3],[0.0,0.2,0.4])
add("float_mps2", [0.1,0.7],[0.0,0.8])
add("float_mps3", [0.3,0.3,0.3],[0.1,0.3,0.5])
json.dump(cases, open("sd_cases.json","w"))
print(len(cases))
