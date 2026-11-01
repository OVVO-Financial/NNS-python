import sys; sys.path.insert(0, "/home/user/audit/work/REG/smooth")
import numpy as np, pandas as pd
import patched_ss
x = np.array([0, 0.7, 1.3, 2.0, 3.1, 4.0, 5.5, 6.2]); y = np.array([0.1, 0.8, 0.9, 0.95, 0.1, -0.7, -0.75, -0.1])
print(["%.12f" % v for v in patched_ss.predictor(x, y, 0.3)(np.array([-1, 0, 1, 3, 6.2, 7.0]))])
d = pd.read_csv("/home/user/audit/work/REG/smooth/x80.csv")
print("n80", ["%.12f" % v for v in patched_ss.predictor(d.x.values, d.y.values, 0.7)(np.array([1., 5, 9]))])
