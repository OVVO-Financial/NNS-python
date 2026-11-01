"""A/B: nns._nnscore (vendored NNS-core) kernels vs R NNS 13.3 Rcpp functions.

Writes inputs + Python outputs to ab_cases.json; ab_native.R recomputes in R and
reports max abs/rel diffs per case.
"""
import json
import numpy as np
import pandas as pd
from nns._native import native_fn as nf

rng = np.random.default_rng(42)
ret = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv").dropna()
spy = ret["SPY"].to_numpy()[:2000]
qqq = ret["QQQ"].to_numpy()[:2000]
tlt = ret["TLT"].to_numpy()[:2000]
disc = rng.integers(0, 4, 300).astype(float)
disc2 = rng.integers(0, 4, 300).astype(float)
ties = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 0, 3, 3.0])
ties2 = np.array([3, 1, 3, 3, 2, 3, 0, 5, 3, 3, 4, 3.0])

cases = []


def add(name, func, args, out):
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    cases.append({"name": name, "func": func, "args": conv(args), "py": conv(out)})


# --- bivariate co-moments with unequal / fractional degrees ---
for (dx, dy) in [(0, 0), (1, 1), (2, 1), (0, 1), (1, 0), (0.5, 1.5), (1.5, 0), (3, 2)]:
    for nm, x, y in [("ret", spy, qqq), ("ties", ties, ties2), ("disc", disc, disc2)]:
        tx, ty = float(np.mean(x)), float(np.mean(y))
        if nm == "ties":
            tx, ty = 3.0, 3.0
        add(f"co_lpm_{nm}_{dx}_{dy}", "co_lpm", dict(dx=dx, dy=dy, x=x, y=y, tx=tx, ty=ty),
            nf("co_lpm")(float(dx), float(dy), x, y, tx, ty))
        add(f"co_upm_{nm}_{dx}_{dy}", "co_upm", dict(dx=dx, dy=dy, x=x, y=y, tx=tx, ty=ty),
            nf("co_upm")(float(dx), float(dy), x, y, tx, ty))
        add(f"d_lpm_{nm}_{dx}_{dy}", "d_lpm", dict(dx=dx, dy=dy, x=x, y=y, tx=tx, ty=ty),
            nf("d_lpm")(float(dx), float(dy), x, y, tx, ty))
        add(f"d_upm_{nm}_{dx}_{dy}", "d_upm", dict(dx=dx, dy=dy, x=x, y=y, tx=tx, ty=ty),
            nf("d_upm")(float(dx), float(dy), x, y, tx, ty))

# --- vector targets (> 32 -> prefix backend in univariate; recycling in bivariate) ---
tg = np.quantile(spy, np.linspace(0.01, 0.99, 50))
for deg in [0, 1, 2, 3, 0.5, 2.5]:
    add(f"lpm_v_{deg}", "lpm_v", dict(deg=deg, t=tg, x=spy), nf("lpm_v")(float(deg), tg, spy))
    add(f"upm_v_{deg}", "upm_v", dict(deg=deg, t=tg, x=spy), nf("upm_v")(float(deg), tg, spy))
    add(f"lpm_ratio_v_{deg}", "lpm_ratio_v", dict(deg=deg, t=tg, x=spy), nf("lpm_ratio_v")(float(deg), tg, spy))
    add(f"upm_ratio_v_{deg}", "upm_ratio_v", dict(deg=deg, t=tg, x=spy), nf("upm_ratio_v")(float(deg), tg, spy))
tgx = np.quantile(spy, np.linspace(0.05, 0.95, 40)); tgy = np.quantile(qqq, np.linspace(0.05, 0.95, 40))
add("co_lpm_v", "co_lpm_v", dict(dx=1, dy=2, x=spy, y=qqq, tx=tgx, ty=tgy), nf("co_lpm_v")(1.0, 2.0, spy, qqq, tgx, tgy))
add("d_upm_v", "d_upm_v", dict(dx=1, dy=0, x=spy, y=qqq, tx=tgx, ty=tgy), nf("d_upm_v")(1.0, 0.0, spy, qqq, tgx, tgy))

# --- nD ---
X3 = np.column_stack([spy, qqq, tlt])
Xt = np.column_stack([ties, ties2, ties[::-1]])
for nm, X, tgt in [("ret", X3, X3.mean(0)), ("ties", Xt, np.array([3.0, 3.0, 3.0]))]:
    flat = np.ascontiguousarray(X.ravel(order="F"))
    for deg in [0, 1, 2, 0.5]:
        for norm in [True, False]:
            for f in ["clpm_nd", "cupm_nd", "dpm_nd"]:
                add(f"{f}_{nm}_{deg}_{norm}", f, dict(X=X, t=tgt, deg=deg, norm=norm),
                    nf(f)(flat, X.shape[0], X.shape[1], tgt, float(deg), norm))
    T = np.quantile(X, [0.2, 0.5, 0.8], axis=0)
    for deg in [0, 1]:
        for norm in [True, False]:
            add(f"clpm_nd_batch_{nm}_{deg}_{norm}", "clpm_nd_batch", dict(X=X, T=T, deg=deg, norm=norm),
                nf("clpm_nd_batch")(flat, X.shape[0], X.shape[1], np.ascontiguousarray(T.ravel(order="F")), T.shape[0], float(deg), norm))

# --- PM matrix ---
for (dl, du, pop, norm) in [(0, 0, False, False), (1, 1, True, False), (1, 1, True, True), (2, 1, False, True), (0.5, 1.5, True, False), (0, 1, True, False)]:
    tgt = X3.mean(0)
    r = nf("pm_matrix")(float(dl), float(du), tgt, np.ascontiguousarray(X3.ravel(order="F")), X3.shape[0], 3, pop, norm)
    add(f"pm_matrix_{dl}_{du}_{pop}_{norm}", "pm_matrix", dict(dl=dl, du=du, t=tgt, X=X3, pop=pop, norm=norm),
        {k: np.asarray(r[k]) for k in ["cupm", "dupm", "dlpm", "clpm", "cov.matrix"]})

# --- fast_lm ---
for nm, x, y in [("ret", spy, qqq), ("offset", 1e9 + np.arange(500) * 1e-3, rng.normal(size=500)),
                 ("time", np.arange(1, 4001, dtype=float), np.cumsum(rng.normal(size=4000)) + 100)]:
    r = nf("fast_lm")(np.ascontiguousarray(x), np.ascontiguousarray(y))
    add(f"fast_lm_{nm}", "fast_lm", dict(x=x, y=y), {"coef": r["coef"], "fitted": r["fitted_values"]})
Xm = rng.normal(size=(200, 3)); ym = Xm @ np.array([1.0, 2, 3]) + rng.normal(size=200)
r = nf("fast_lm_mult")(np.ascontiguousarray(Xm.ravel(order="F")), ym, 200, 3)
add("fast_lm_mult", "fast_lm_mult", dict(X=Xm, y=ym), {"coef": r["coefficients"], "r2": r["r_squared"]})

# --- central tendencies ---
vecs = {"ret": spy, "disc": disc, "ties": ties, "n4": np.array([0, 0, 1, 1.0]), "n5": np.array([1, 2, 2, 3, 9.0]),
        "bimodal": np.concatenate([rng.normal(0, 1, 500), rng.normal(6, 1, 500)]),
        "prices": pd.read_csv("/home/user/audit/data/prices_daily.csv")["SPY"].dropna().to_numpy()[:3000],
        "neg_half": np.array([-2.5, -2.5, -1.5, -0.5, 0.5, 1.5]), "big_int": np.array([1e8, 1e8, 1e8 + 1, 5.0])}
for nm, v in vecs.items():
    add(f"gravity_{nm}", "gravity", dict(x=v, d=False), nf("gravity")(v, False))
    add(f"gravity_disc_{nm}", "gravity", dict(x=v, d=True), nf("gravity")(v, True))
    add(f"gravity_exact_{nm}", "gravity", dict(x=v, d=False), nf("gravity_exact")(v))
    for d in [False, True]:
        for m in [False, True]:
            add(f"mode_{nm}_{d}_{m}", "mode", dict(x=v, d=d, m=m), np.asarray(nf("mode")(v, d, m)))

# --- generate vectors ---
ts = np.cumsum(rng.normal(size=60)) + 50
r = nf("generate_vectors")(ts, np.array([1, 3, 12], dtype=np.int32))
add("generate_vectors", "generate_vectors", dict(x=ts, l=[1, 3, 12]), {"series": r["series"], "index": r["index"]})
for (l, h) in [(1, 1), (4, 3), (12, 6)]:
    r = nf("generate_lin_vectors")(ts, l, h)
    add(f"generate_lin_vectors_{l}_{h}", "generate_lin_vectors", dict(x=ts, l=l, h=h),
        {k: r[k] for k in ["series", "index", "forecast_values", "forecast_index"]})

# --- stochastic superiority ---
for nm, x, y in [("ret", spy, qqq), ("disc", disc, disc2), ("ties", ties, ties2)]:
    r = nf("stochastic_superiority")(x, y)
    add(f"stoch_sup_{nm}", "stoch_sup", dict(x=x, y=y), r)

for nm, v in [("disc", disc), ("ret", spy), ("half", np.array([0.5, 1, 2.0, 3]))]:
    add(f"is_discrete_{nm}", "is_discrete", dict(x=v), bool(nf("is_discrete")(v)))

json.dump(cases, open("/home/user/audit/work/core/ab_cases.json", "w"))
print(len(cases), "cases written")
