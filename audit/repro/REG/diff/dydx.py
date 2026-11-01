import json, numpy as np, pandas as pd
from nns import dy_dx
x = pd.read_csv("sinx.csv").x.values; y = np.sin(x)
# R seq(0, 2*pi, pi/100) has 201 points
d = pd.read_csv("cubic.csv"); xu, yu = d.x.values, d.y.values
o = np.argsort(xu)
res = {
 "sin1": dy_dx(x, y, 1.75), "sin2": dy_dx(x, y, [1.75, 2.5]), "sin_bound": dy_dx(x, y, [0, 2*np.pi]),
 "sin_overall": dy_dx(x, y, "overall"), "cubic": dy_dx(xu, yu, [-1.5, 0, .5, 1.5]),
 "cubic_sorted": dy_dx(xu[o], yu[o], [-1.5, 0, .5, 1.5])}
r = json.load(open("dydx_r.json"))
print(len(x))
for k, v in res.items():
    if isinstance(v, dict):
        print(k, "py first", np.round(v["first.derivative"], 7), "R", np.round(r[k]["first.derivative"], 7))
        print(k, "py second", np.round(v["second.derivative"], 7), "R", r[k]["second.derivative"])
    else:
        print(k, v, r[k])
