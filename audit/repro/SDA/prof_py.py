import cProfile, pstats, numpy as np, nns, time
rng=np.random.default_rng(1); y=rng.normal(0,1,1000); x=y+0.1
pr=cProfile.Profile(); pr.enable()
for _ in range(300): nns.ssd_uni(x,y)
pr.disable(); pstats.Stats(pr).sort_stats("cumtime").print_stats(8)
y=rng.normal(0,1,3000); x=y+0.1
pr=cProfile.Profile(); pr.enable()
nns.fsd_uni(x,y,type="continuous")
pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(5)
# compare: prefix-sum curve
from nns.core import lpm, upm
g=np.sort(np.r_[x,y]); t=time.perf_counter(); 
for _ in range(20): u=upm(1,g,x)
print("upm(1,grid,x) n=3000: %.2f ms"%((time.perf_counter()-t)/20*1000), np.allclose(u, np.mean(np.maximum(0,x-g[:,None]),1)))
