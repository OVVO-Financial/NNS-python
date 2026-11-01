import numpy as np, pandas as pd
from nns import nns_distance
rpm = pd.read_csv("rpm.csv").values
pts = [np.array([0.5, 0.5]), np.array([0.1, 0.9]), np.array([2.0, -1.0]), rpm[2, :2]]
lines = open("dist_r.txt").read().splitlines()
i = 0
for p in pts:
    for k in (1, 2, 5, "all"):
        rv = [float(v) for v in lines[i].split()]; i += 1
        pv = [nns_distance(rpm, p, k=k), nns_distance(rpm, p, k=k, class_=True)]
        flag = "" if np.allclose(rv, pv, rtol=1e-12) else "   <-- DIFF"
        print(p, k, "R", rv, "py", pv, flag)
r2 = np.array([[0, 0, 2.], [1, 0, 1], [5, 5, 3]])
print(lines[i]); print("py tie class k=2:", nns_distance(r2, np.array([.5, 0]), k=2, class_=True), "k=1:", nns_distance(r2, np.array([.5, 0]), k=1, class_=True), nns_distance(r2, np.array([.5, 0]), k=1))
