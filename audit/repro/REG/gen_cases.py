"""Generate A/B cases for nns_reg vs NNS.reg.

Each case: directory cases/<name>/ with x.csv, y.csv, (optional) pe.csv and args.json.
args.json holds R-style argument names; the Python runner maps them.
"""
import json
import os
import shutil

import numpy as np
import pandas as pd

ROOT = "/home/user/audit/work/REG/cases"
DATA = "/home/user/audit/data"
shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(ROOT)

rng = np.random.default_rng(20261031)


def write(name, x, y, pe=None, x_factor_cols=None, y_factor=False, **args):
    d = os.path.join(ROOT, name)
    os.makedirs(d)
    xdf = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x
    if not isinstance(x, pd.DataFrame):
        xdf.columns = ["x"] if xdf.shape[1] == 1 else [f"V{i+1}" for i in range(xdf.shape[1])]
    xdf.to_csv(os.path.join(d, "x.csv"), index=False)
    pd.DataFrame({"y": np.asarray(y)}).to_csv(os.path.join(d, "y.csv"), index=False)
    if pe is not None:
        pedf = pd.DataFrame(pe) if not isinstance(pe, pd.DataFrame) else pe
        if not isinstance(pe, pd.DataFrame):
            pedf.columns = list(xdf.columns)
        pedf.to_csv(os.path.join(d, "pe.csv"), index=False)
    meta = {
        "args": args,
        "univariate": xdf.shape[1] == 1,
        "x_factor_cols": x_factor_cols or [],
        "y_factor": y_factor,
        "has_pe": pe is not None,
    }
    with open(os.path.join(d, "args.json"), "w") as f:
        json.dump(meta, f)


# ---------------- simulations ----------------
n = 400
xs = rng.uniform(0, 2 * np.pi, n)  # unsorted
ys = np.sin(xs) + rng.normal(0, 0.1, n)
pe_s = np.array([-0.5, 0.0, 1.0, np.pi, 5.0, 2 * np.pi + 0.5, 7.5])  # includes extrapolation
for nm, a in {
    "default": {},
    "o1": {"order": 1},
    "o3": {"order": 3},
    "o5": {"order": 5},
    "omax": {"order": "max"},
    "smooth": {"smooth": True},
    "smooth_o3": {"smooth": True, "order": 3},
    "ci95": {"confidence.interval": 0.95},
    "nr_mean": {"noise.reduction": "mean"},
    "nr_median": {"noise.reduction": "median"},
    "nr_mode": {"noise.reduction": "mode"},
    "xonly": {"type": "XONLY"},
    "ponly": {"point.only": True},
}.items():
    write(f"sin_{nm}", xs, ys, pe=pe_s, **a)

xq = rng.normal(0, 1, 500)
yq = xq**2 + rng.normal(0, 0.2, 500)
pe_q = np.array([-4.0, -2.0, 0.0, 0.5, 2.0, 4.0])
write("sq_default", xq, yq, pe=pe_q)
write("sq_ci90", xq, yq, pe=pe_q, **{"confidence.interval": 0.9})
write("sq_smooth", xq, yq, pe=pe_q, smooth=True)

# step with ties in x
xt = np.round(rng.uniform(0, 1, 300), 2)
yt = (xt > 0.5).astype(float) * 2 + rng.normal(0, 0.1, 300)
write("step_ties", xt, yt, pe=np.array([0.0, 0.25, 0.5, 0.505, 1.0, 1.2]))
write("step_ties_omax", xt, yt, pe=np.array([0.0, 0.25, 0.5, 0.505, 1.0, 1.2]), order="max")

# large-n sin (smoothing spline with > 49 regression points)
xl = rng.uniform(0, 10, 5000)
yl = np.sin(xl) + rng.normal(0, 0.05, 5000)
write("bigsin_default", xl, yl, pe=np.array([0.5, 5.0, 9.5]))
write("bigsin_smooth", xl, yl, pe=np.array([0.5, 5.0, 9.5]), smooth=True)
write("bigsin_o8_smooth", xl, yl, pe=np.array([0.5, 5.0, 9.5]), smooth=True, order=8)

# integer x, integer y
xi = rng.integers(0, 20, 200)
yi = (xi * 3 + rng.integers(0, 5, 200)).astype(np.int64)
write("int_xy", xi, yi, pe=np.array([0, 5, 19, 25]))

# binary 0/1 y -> auto class
xb = rng.normal(0, 1, 300)
yb = (xb + rng.normal(0, 0.5, 300) > 0).astype(int)
write("binary_auto", xb, yb, pe=np.array([-3.0, -0.1, 0.0, 0.1, 3.0]))
write("binary_auto_ci", xb, yb, pe=np.array([-3.0, -0.1, 0.0, 0.1, 3.0]), **{"confidence.interval": 0.9})

# small cases
write("n2", np.array([1.0, 2.0]), np.array([3.0, 5.0]), pe=np.array([0.0, 1.5, 3.0]))
write("n3", np.array([3.0, 1.0, 2.0]), np.array([1.0, 5.0, 2.0]), pe=np.array([0.0, 1.5, 3.0]))
write("const_y", np.arange(10.0), np.full(10, 2.0), pe=np.array([0.0, 20.0]))
write("const_x", np.full(10, 1.0), np.arange(10.0), pe=np.array([0.0, 1.0]))

# ---------------- real data ----------------
boston = pd.read_csv(f"{DATA}/boston.csv")
bx = boston.drop(columns=["medv"])
by = boston["medv"].to_numpy()
bpe = bx.iloc[[0, 10, 100, 300, 505]].copy()
write("boston_lstat", boston["lstat"].to_numpy(), by, pe=np.array([1.0, 5.0, 20.0, 40.0]))
write("boston_lstat_smooth", boston["lstat"].to_numpy(), by, pe=np.array([1.0, 5.0, 20.0, 40.0]), smooth=True)
write("boston_lstat_ci", boston["lstat"].to_numpy(), by, pe=np.array([1.0, 5.0, 20.0, 40.0]), **{"confidence.interval": 0.95})
write("boston_mv", bx, by, pe=bpe)
write("boston_mv_ci", bx, by, pe=bpe, **{"confidence.interval": 0.9})
write("boston_mv_nb1", bx, by, pe=bpe, **{"n.best": 1})
write("boston_mv_nball", bx, by, pe=bpe, **{"n.best": "all"})
write("boston_mv_omax", bx, by, pe=bpe, order="max")
write("boston_mv_o2", bx, by, pe=bpe, order=2)
write("boston_mv_L1", bx, by, pe=bpe, dist="L1")
write("boston_mv_L2", bx, by, pe=bpe, dist="L2")
write("boston_mv_FACTOR", bx, by, pe=bpe, dist="FACTOR")
write("boston_mv_nrmean", bx, by, pe=bpe, **{"noise.reduction": "mean"})
for m in ["cor", "NNS.dep", "NNS.caus", "equal", "all"]:
    write(f"boston_dr_{m}", bx, by, pe=bpe, **{"dim.red.method": m})
write("boston_dr_cor_thr", bx, by, pe=bpe, **{"dim.red.method": "cor", "threshold": 0.4})
write("boston_dr_num", bx, by, pe=bpe, **{"dim.red.method": list(np.linspace(-1, 1, 13))})
write("boston_dr_cor_smooth_ci", bx, by, pe=bpe, smooth=True, **{"dim.red.method": "cor", "confidence.interval": 0.9})

mt = pd.read_csv(f"{DATA}/mtcars.csv").drop(columns=["Unnamed: 0"])
write("mtcars_wt", mt["wt"].to_numpy(), mt["mpg"].to_numpy(), pe=np.array([1.0, 3.0, 6.0]))
write("mtcars_mv", mt[["wt", "hp", "disp"]], mt["mpg"].to_numpy(), pe=mt[["wt", "hp", "disp"]].iloc[[0, 5]])
mtc = mt[["wt", "hp"]].copy()
cyl_str = mt["cyl"].map({4: "four", 6: "six", 8: "eight"})
mtc["cyl"] = cyl_str
pe_c = mtc.iloc[[0, 2, 4]].copy()
write("mtcars_factor_chr", mtc, mt["mpg"].to_numpy(), pe=pe_c, x_factor_cols=[])
write("mtcars_factor_fac", mtc, mt["mpg"].to_numpy(), pe=pe_c, x_factor_cols=["cyl"])
write("mtcars_factor_chr_nodummy", mtc, mt["mpg"].to_numpy(), pe=pe_c, x_factor_cols=[], **{"factor.2.dummy": False})
write("mtcars_factor_fac_nodummy", mtc, mt["mpg"].to_numpy(), pe=pe_c, x_factor_cols=["cyl"], **{"factor.2.dummy": False})
write("mtcars_am_auto", mt[["wt", "hp"]], mt["am"].to_numpy(), pe=mt[["wt", "hp"]].iloc[[0, 5]])
write("mtcars_cyl_univ_factor", cyl_str.to_numpy(), mt["mpg"].to_numpy(), pe=np.array(["six", "eight"]), x_factor_cols=["x"])

iris = pd.read_csv(f"{DATA}/iris.csv")
ix = iris.iloc[:, :4]
iy = iris["Species"].to_numpy()
ipe = ix.iloc[[0, 51, 101, 149]]
write("iris_class", ix, iy, pe=ipe, y_factor=True, type="CLASS")
write("iris_class_auto", ix, iy, pe=ipe, y_factor=True)
write("iris_class_nb3", ix, iy, pe=ipe, y_factor=True, type="CLASS", **{"n.best": 3})
write("iris_class_dr_cor", ix, iy, pe=ipe, y_factor=True, type="CLASS", **{"dim.red.method": "cor"})
write("iris_class_ci", ix, iy, pe=ipe, y_factor=True, type="CLASS", **{"confidence.interval": 0.9})
write("iris_codes_class", ix, pd.Categorical(iy).codes + 1, pe=ipe, type="CLASS")
write("iris_codes_numeric", ix, pd.Categorical(iy).codes + 1, pe=ipe)
write("iris_univ_class", ix.iloc[:, 2].to_numpy(), iy, pe=np.array([1.0, 3.0, 4.9, 5.0, 7.0]), y_factor=True, type="CLASS")

rets = pd.read_csv(f"{DATA}/returns_daily_equity_etf.csv").dropna()
fx = rets[["QQQ", "TLT"]].reset_index(drop=True)
fy = rets["SPY"].to_numpy()
fpe = pd.DataFrame({"QQQ": [-0.05, -0.01, 0.0, 0.01, 0.08], "TLT": [0.02, 0.0, 0.0, -0.01, -0.03]})
write("fin_univ", fx["QQQ"].to_numpy(), fy, pe=fpe["QQQ"].to_numpy())
write("fin_univ_ci", fx["QQQ"].to_numpy(), fy, pe=fpe["QQQ"].to_numpy(), **{"confidence.interval": 0.95})
write("fin_univ_smooth", fx["QQQ"].to_numpy(), fy, pe=fpe["QQQ"].to_numpy(), smooth=True)
write("fin_mv", fx, fy, pe=fpe)
write("fin_mv_ci", fx, fy, pe=fpe, **{"confidence.interval": 0.95})
write("fin_dr_cor", fx, fy, pe=fpe, **{"dim.red.method": "cor"})
write("fin_dr_dep", fx, fy, pe=fpe, **{"dim.red.method": "NNS.dep"})
write("fin_dr_caus_ts", fx, fy, pe=fpe, **{"dim.red.method": "NNS.caus", "tau": "ts"})
print("cases:", len(os.listdir(ROOT)))
