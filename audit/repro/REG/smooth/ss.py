import numpy as np, pandas as pd
from nns._reg_engine import _smooth_spline_predictor
x = np.array([0, 0.7, 1.3, 2.0, 3.1, 4.0, 5.5, 6.2]); y = np.array([0.1, 0.8, 0.9, 0.95, 0.1, -0.7, -0.75, -0.1])
for s in (0.3, 0.6, 0.75):
    f = _smooth_spline_predictor(x, y, s)
    print(s, ["%.12f" % v for v in f(np.array([-1, 0, 1, 3, 6.2, 7.0]))])
d = pd.read_csv("/home/user/audit/work/REG/smooth/x80.csv")
f = _smooth_spline_predictor(d.x.values, d.y.values, 0.7)
print("n80", ["%.12f" % v for v in f(np.array([1., 5, 9]))])
