import json as _j
import os as _os
import sys

import numpy as np

sys.path.insert(0, ".")
from harness import close, fmt_res, rget, rnames, run_r

import nns

rng = np.random.default_rng(7)
cases, pycalls, cmp = [], {}, {}


def add(cid, expr, data, fn, comparer=None):
    cases.append({"id": cid, "expr": expr, "data": data})
    pycalls[cid] = fn
    cmp[cid] = comparer


V = {
    "norm": rng.normal(size=101),
    "unif": rng.uniform(size=64),
    "ties": rng.integers(0, 6, size=50).astype(float),
    "bin": rng.integers(0, 2, size=30).astype(float),
    "n1": np.array([3.0]),
    "n2": np.array([1.0, 2.0]),
    "n3": np.array([5.0, 1.0, 3.0]),
    "const": np.full(9, -2.0),
    "bimodal": np.concatenate([rng.normal(-3, 0.5, 60), rng.normal(3, 0.5, 60)]),
    "big": 1e6 + rng.normal(size=40),
    "exp": rng.exponential(size=200),
    "int_neg": rng.integers(-10, 10, size=33).astype(float),
    "trimodal": np.concatenate(
        [rng.normal(-5, 0.3, 40), rng.normal(0, 0.3, 50), rng.normal(5, 0.3, 40)]
    ),
}
for vn, v in V.items():
    for disc in (False, True):
        add(
            f"gravity|{vn}|{disc}",
            f"NNS.gravity(x, discrete={str(disc).upper()})",
            {"x": v.tolist()},
            lambda v=v, disc=disc: nns.nns_gravity(v, discrete=disc),
        )
        for multi in (False, True):
            add(
                f"mode|{vn}|{disc}|{multi}",
                f"NNS.mode(x, discrete={str(disc).upper()}, multi={str(multi).upper()})",
                {"x": v.tolist()},
                lambda v=v, disc=disc, multi=multi: nns.nns_mode(v, discrete=disc, multi=multi),
            )
    for a, b in ((0, 1), (-5, 5)):
        add(
            f"rescale.mm|{vn}|{a}",
            f"NNS.rescale(x, {a}, {b})",
            {"x": v.tolist()},
            lambda v=v, a=a, b=b: nns.nns_rescale(v, a, b),
        )
    if np.mean(v) > 0:
        for typ in ("Terminal", "Discounted"):
            add(
                f"rescale.rn|{vn}|{typ}",
                f"NNS.rescale(x, 100, 0.05, method='riskneutral', T=2, type='{typ}')",
                {"x": v.tolist()},
                lambda v=v, typ=typ: nns.nns_rescale(
                    v, 100, 0.05, method="riskneutral", time_to_maturity=2, type=typ
                ),
            )
    # CDF univariate
    if v.size >= 3:
        for deg in (0, 1, 2):
            for typ in ("CDF", "survival", "hazard", "cumulative hazard"):
                for tn, t in (("none", None), ("med", float(np.median(v)))):
                    texpr = "NULL" if t is None else "t"
                    add(
                        f"CDF|{vn}|{deg}|{typ}|{tn}",
                        f"NNS.CDF(x, degree={deg}, target={texpr}, type='{typ}', plot=FALSE)",
                        {"x": v.tolist(), "t": 0 if t is None else t},
                        lambda v=v, deg=deg, typ=typ, t=t: nns.nns_cdf(
                            v, degree=deg, target=t, type=typ
                        ),
                        "cdf",
                    )

# multivariate CDF
for d in (2, 3, 4):
    X = rng.normal(size=(35, d))
    Xt = np.round(X)
    for Xn, XX in (("c", X), ("t", Xt)):
        for deg in (0, 1):
            for typ in ("CDF", "survival", "cumulative hazard"):
                for tn, t in (("none", None), ("mean", XX.mean(axis=0))):
                    texpr = "NULL" if t is None else "t"
                    add(
                        f"CDFmv|{d}|{Xn}|{deg}|{typ}|{tn}",
                        f"NNS.CDF(X, degree={deg}, target={texpr}, type='{typ}', plot=FALSE)",
                        {"X": XX.tolist(), "t": 0 if t is None else t.tolist()},
                        lambda XX=XX, deg=deg, typ=typ, t=t: nns.nns_cdf(
                            XX, degree=deg, target=t, type=typ
                        ),
                        "cdf",
                    )

# norm
for d in (2, 3, 11):
    X = np.abs(rng.normal(size=(30, d))) * rng.uniform(1, 50, size=d)
    for lin in (False, True):
        add(
            f"norm|{d}|{lin}",
            f"unname(NNS.norm(X, linear={str(lin).upper()}))",
            {"X": X.tolist()},
            lambda X=X, lin=lin: nns.nns_norm(X, linear=lin),
        )


# dep pair / asym / matrix ; copula ; caus
def rel(kind, n):
    x = rng.normal(size=n)
    if kind == "lin":
        y = 2 * x + rng.normal(size=n) * 0.5
    elif kind == "quad":
        y = x**2 + rng.normal(size=n) * 0.1
    elif kind == "sin":
        y = np.sin(3 * x) + rng.normal(size=n) * 0.1
    elif kind == "indep":
        y = rng.normal(size=n)
    elif kind == "ties":
        x = np.round(x)
        y = np.round(x + rng.normal(size=n))
    elif kind == "bin":
        x = rng.integers(0, 2, n).astype(float)
        y = x + rng.normal(size=n)
    elif kind == "step":
        y = (x > 0).astype(float) + rng.normal(size=n) * 0.01
    return x, y


for kind in ("lin", "quad", "sin", "indep", "ties", "bin", "step"):
    for n in (8, 15, 30, 100, 401, 1000):
        x, y = rel(kind, n)
        for asym in (False, True):
            add(
                f"dep|{kind}|{n}|{asym}",
                f"NNS.dep(x, y, asym={str(asym).upper()})",
                {"x": x.tolist(), "y": y.tolist()},
                lambda x=x, y=y, asym=asym: nns.nns_dep(x, y, asym=asym),
                "dict",
            )
        add(
            f"copula|{kind}|{n}",
            "NNS.copula(cbind(x, y))",
            {"x": x.tolist(), "y": y.tolist()},
            lambda x=x, y=y: nns.nns_copula(np.column_stack([x, y])),
        )
        add(
            f"copula.d|{kind}|{n}",
            "NNS.copula(cbind(x, y), continuous=FALSE)",
            {"x": x.tolist(), "y": y.tolist()},
            lambda x=x, y=y: nns.nns_copula(np.column_stack([x, y]), continuous=False),
        )
        add(
            f"copula.t|{kind}|{n}",
            "NNS.copula(cbind(x, y), target=c(0.1,-0.2))",
            {"x": x.tolist(), "y": y.tolist()},
            lambda x=x, y=y: nns.nns_copula(np.column_stack([x, y]), target=[0.1, -0.2]),
        )
        if n >= 15:
            for tau in (0, 1, 3):
                add(
                    f"caus|{kind}|{n}|{tau}",
                    f"NNS.caus(x, y, tau={tau})",
                    {"x": x.tolist(), "y": y.tolist()},
                    lambda x=x, y=y, tau=tau: nns.nns_causation(x, y, tau=tau),
                    "dict",
                )
for d in (3, 4):
    X = rng.normal(size=(60, d))
    X[:, 1] = X[:, 0] ** 2 + rng.normal(size=60) * 0.1
    for asym in (False, True):
        add(
            f"depmat|{d}|{asym}",
            f"NNS.dep(X, asym={str(asym).upper()})",
            {"X": X.tolist()},
            lambda X=X, asym=asym: nns.nns_dep(X, asym=asym),
            "dict",
        )
    add(f"copula3|{d}", "NNS.copula(X)", {"X": X.tolist()}, lambda X=X: nns.nns_copula(X))
    add(f"causmat|{d}", "unname(NNS.caus(X))", {"X": X.tolist()}, lambda X=X: nns.causal_matrix(X))
    add(
        f"causmat1|{d}",
        "unname(NNS.caus(X, tau=1))",
        {"X": X.tolist()},
        lambda X=X: nns.causal_matrix(X, tau=1),
    )

# seasonal ts tau
t = np.arange(120)
xs = np.sin(2 * np.pi * t / 12) + rng.normal(size=120) * 0.2
ys = np.roll(xs, 2) + rng.normal(size=120) * 0.2
add(
    "caus.ts",
    "NNS.caus(x, y, tau='ts')",
    {"x": xs.tolist(), "y": ys.tolist()},
    lambda: nns.nns_causation(xs, ys, tau="ts"),
    "dict",
)
add(
    "caus.cs",
    "NNS.caus(x, y, tau='cs')",
    {"x": xs.tolist(), "y": ys.tolist()},
    lambda: nns.nns_causation(xs, ys, tau="cs"),
    "dict",
)

print("cases", len(cases))
R = run_r(cases)
problems = {}
for c in cases:
    cid = c["id"]
    r = R[cid]
    try:
        py = pycalls[cid]()
        perr = None
    except Exception as exc:
        py, perr = None, f"{type(exc).__name__}: {exc}"
    msgs = []
    if not r["ok"]:
        if perr is None:
            msgs.append(f"R error but py ok: R={r['error'][:150]} py={str(py)[:80]}")
    elif perr is not None:
        msgs.append(f"py error but R ok: {perr[:200]}")
    else:
        rv = r["value"]
        kind = cmp[cid]
        if kind == "cdf":
            fr = rget(rv, "Function")
            fp = py["Function"]
            for i, nm in enumerate(rnames(fr)):
                key = nm if nm in fp else list(fp.keys())[i]
                msgs += close(np.asarray(fp[key]), fr["values"][i], path=f"Function.{nm}")
            msgs += close(
                np.asarray(py["target.value"]), rget(rv, "target.value"), path="target.value"
            )
        elif kind == "dict":
            names = rnames(rv)
            if names:
                for nm in names:
                    msgs += close(np.asarray(py[nm]), rget(rv, nm), path=nm)
            else:
                msgs += close(np.asarray([py[k] for k in py]), rv)
        else:
            msgs += close(np.asarray(py), rv)
        if r["warnings"]:
            pass
    problems[cid] = msgs

if _os.environ.get("CT_DUMP"):
    _j.dump({k: v for k, v in problems.items() if v}, open(_os.environ["CT_DUMP"], "w"), indent=1)
print(fmt_res("central/dep/caus", problems, len(cases)))
