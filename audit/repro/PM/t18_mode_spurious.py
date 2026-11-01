"""NNS.mode(multi=TRUE) (R default) reports many spurious modes on unimodal data."""
import sys; sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, nns
from rrun import rcall
rng = np.random.default_rng(2)
counts = []
for n in [200, 1000, 5000]:
    xs = [rng.normal(size=n) for _ in range(20)]
    r = rcall("apply(xs, 1, function(x) length(NNS.mode(x)))", xs=np.array(xs))
    p = [np.atleast_1d(nns.nns_mode(x, multi=True)).size for x in xs]
    print(f"N(0,1), n={n}: R NNS.mode(x) #modes over 20 draws: {r}; python multi=True identical: {list(r) == p}")
x = rng.standard_t(3, size=4000) * 0.02
print("t3 returns n=4000: #modes", np.atleast_1d(nns.nns_mode(x, multi=True)).size)
