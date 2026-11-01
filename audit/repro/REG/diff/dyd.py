import time, numpy as np, pandas as pd
from nns import dy_d
d = pd.read_csv("dyd.csv"); x = d[["x1", "x2"]].values; y = d.y.values
ep = np.array([[-1, 0], [0, 0], [1, 0.5], [1.5, -1]])
t0 = time.perf_counter()
r1 = dy_d(x, y, 1, ep, messages=False); r2 = dy_d(x, y, 2, ep, messages=False)
print("time", time.perf_counter() - t0)
print("py wrt1 First ", np.round(r1["First"], 5)); print("py wrt1 Second", np.round(r1["Second"], 5)); print("py wrt2 First ", np.round(r2["First"], 5))
print("py apd-vec wrt1 First", np.round(dy_d(x, y, 1, np.array([-1., 0, 1]), messages=False)["First"], 5))
print("truth wrt1 First", 2 * ep[:, 0], "Second 2; wrt2 3")
