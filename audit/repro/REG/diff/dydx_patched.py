"""Verify root causes of dy_dx mismatch: (1) h_step uses index not h_s value, (2) smooth spline."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/audit/work/REG/smooth")
import patched_ss, nns._reg_engine as eng
from nns.dependence import _gravity
from nns.regression import nns_reg
eng._smooth_spline_predictor = patched_ss.predictor

def dy_dx_fixed(x, y, ev):
    ev = np.atleast_1d(np.asarray(ev, float)); n = x.size
    h_s = np.round(np.exp(np.linspace(np.log(2), np.log(int(np.floor(np.sqrt(n)))), 5)))  # R round = half-even, np.round same
    sp = float(_gravity(np.abs(np.diff(x))))
    rows = []
    for h in h_s:
        idx = np.flatnonzero(h_s == h)
        step = sp * h_s[idx]
        L = max(ev.size, step.size); e = np.resize(ev, L); s = np.resize(step, L)
        rows.append(np.column_stack((np.maximum(x.min(), e - s), e, np.minimum(x.max(), e + s))))
    dp = np.vstack(rows)
    est = np.asarray(nns_reg(x, y, point_est=np.concatenate((dp[:, 0], dp[:, 1], dp[:, 2])), point_only=True, smooth=True)["Point.est"]).reshape(3, -1).T
    r1 = dp[:, 2] - dp[:, 1]; r2 = dp[:, 1] - dp[:, 0]; s1 = est[:, 2] - est[:, 1]; s2 = est[:, 1] - est[:, 0]
    with np.errstate(all="ignore"):
        f = (s1 + s2) / (r1 + r2); sec = (s1 / r1 - s2 / r2) / ((r1 + r2) / 2)
    u = np.unique(dp[:, 1])
    return [float(np.mean(f[dp[:, 1] == p])) for p in u], [float(np.mean(sec[dp[:, 1] == p])) for p in u]

x = pd.read_csv("sinx.csv").x.values; y = np.sin(x)
d = pd.read_csv("cubic.csv"); xu, yu = d.x.values, d.y.values
r = json.load(open("dydx_r.json"))
for k, (xx, yy, ev) in {"sin2": (x, y, [1.75, 2.5]), "sin_bound": (x, y, [0, 2*np.pi]), "cubic": (xu, yu, [-1.5, 0, .5, 1.5])}.items():
    f, s = dy_dx_fixed(xx, yy, ev)
    print(k, "fixed py", np.round(f, 9), np.round(s, 9)); print(k, "R       ", np.round(r[k]["first.derivative"], 9), r[k]["second.derivative"])
