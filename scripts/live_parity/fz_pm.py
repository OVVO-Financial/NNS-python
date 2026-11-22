import json as _j
import sys

import numpy as np

sys.path.insert(0, ".")
from harness import close, fmt_res, out_dir, rget, run_r

import nns

rng = np.random.default_rng(20261108)


def datasets():
    out = {}
    out["norm50"] = rng.normal(size=50)
    out["ties40"] = rng.integers(-3, 4, size=40).astype(float)
    out["n1"] = np.array([2.5])
    out["n2"] = np.array([1.0, -1.0])
    out["const"] = np.full(7, 3.0)
    out["shift"] = 1e8 + rng.normal(size=30)
    out["skew"] = rng.exponential(size=60) ** 2
    out["tiny"] = rng.normal(size=25) * 1e-9
    return out


D = datasets()
cases, pycalls = [], {}


def add(cid, expr, data, fn):
    cases.append({"id": cid, "expr": expr, "data": data})
    pycalls[cid] = fn


degrees = [0, 1, 2, 3, 0.5, 1.5, 4, 5]
for dn, x in D.items():
    targets = {
        "mean": float(np.mean(x)),
        "x0": float(x[0]),
        "lo": float(x.min() - 1),
        "hi": float(x.max() + 1),
        "vec": [float(v) for v in np.unique(np.concatenate([x[:5], [np.median(x)]]))],
    }
    for deg in degrees:
        for tn, t in targets.items():
            for fn in ("LPM", "UPM"):
                cid = f"{fn}|{dn}|{deg}|{tn}"
                pyf = getattr(nns, fn.lower())
                add(
                    cid,
                    f"{fn}({deg}, t, x)",
                    {"x": x.tolist(), "t": t},
                    (
                        lambda pyf=pyf, deg=deg, t=t, x=x: pyf(
                            deg, np.array(t) if isinstance(t, list) else t, x
                        )
                    ),
                )
            for fn in ("LPM.ratio", "UPM.ratio"):
                cid = f"{fn}|{dn}|{deg}|{tn}"
                pyf = nns.lpm_ratio if fn == "LPM.ratio" else nns.upm_ratio
                add(
                    cid,
                    f"{fn}({deg}, t, x)",
                    {"x": x.tolist(), "t": t},
                    (
                        lambda pyf=pyf, deg=deg, t=t, x=x: pyf(
                            deg, np.array(t) if isinstance(t, list) else t, x
                        )
                    ),
                )
        # excess_ret
        for fn in ("LPM", "UPM"):
            tv = (x * 0.5 + 0.1).tolist()
            cid = f"{fn}.excess|{dn}|{deg}"
            pyf = getattr(nns, fn.lower())
            add(
                cid,
                f"{fn}({deg}, t, x, excess_ret=TRUE)",
                {"x": x.tolist(), "t": tv},
                (lambda pyf=pyf, deg=deg, tv=tv, x=x: pyf(deg, np.array(tv), x, excess_ret=True)),
            )
    # VaR
    for deg in [0, 1, 2, 3, 4, 0.5, 1.5, 5, 2.5]:
        ps = [0.0, 0.01, 0.05, 0.25, 0.5, 0.77, 0.95, 1.0]
        for fn in ("LPM.VaR", "UPM.VaR"):
            cid = f"{fn}|{dn}|{deg}"
            pyf = nns.lpm_var if fn == "LPM.VaR" else nns.upm_var
            add(
                cid,
                f"unname({fn}(p, {deg}, x))",
                {"x": x.tolist(), "p": ps},
                (lambda pyf=pyf, deg=deg, ps=ps, x=x: pyf(np.array(ps), deg, x)),
            )
            cid = f"{fn}.scalar|{dn}|{deg}"
            add(
                cid,
                f"unname({fn}(0.1, {deg}, x))",
                {"x": x.tolist()},
                (lambda pyf=pyf, deg=deg, x=x: pyf(0.1, deg, x)),
            )
    # moments
    if x.size >= 4:
        for pop in (True, False):
            cid = f"NNS.moments|{dn}|{pop}"
            add(
                cid,
                f"NNS.moments(x, population={'TRUE' if pop else 'FALSE'})",
                {"x": x.tolist()},
                (lambda pop=pop, x=x: nns.nns_moments(x, population=pop)),
            )

# co-moments
pairs = {
    "norm": (rng.normal(size=60), rng.normal(size=60)),
    "ties": (
        rng.integers(-2, 3, size=45).astype(float),
        rng.integers(-2, 3, size=45).astype(float),
    ),
    "dep": (lambda a: (a, a**2 + rng.normal(size=a.size) * 0.1))(rng.normal(size=80)),
    "n1": (np.array([1.0]), np.array([2.0])),
}
for pn, (x, y) in pairs.items():
    tgs = {
        "mean": (float(np.mean(x)), float(np.mean(y))),
        "zero": (0.0, 0.0),
        "pt": (float(x[0]), float(y[0])),
        "vec": ([float(v) for v in x[:4]], [float(v) for v in y[:4]]),
        "mixed": ([float(v) for v in x[:3]], 0.0),
    }
    for d1, d2 in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 1), (2, 0), (0.5, 1.5), (3, 1)]:
        for tn, (tx, ty) in tgs.items():
            for fn in ("Co.LPM", "Co.UPM"):
                cid = f"{fn}|{pn}|{d1}|{d2}|{tn}"
                pyf = nns.co_lpm if fn == "Co.LPM" else nns.co_upm
                add(
                    cid,
                    f"{fn}({d1}, x, y, tx, ty, degree_y={d2})",
                    {"x": x.tolist(), "y": y.tolist(), "tx": tx, "ty": ty},
                    (
                        lambda pyf=pyf, d1=d1, d2=d2, x=x, y=y, tx=tx, ty=ty: pyf(
                            d1,
                            x,
                            y,
                            np.array(tx) if isinstance(tx, list) else tx,
                            np.array(ty) if isinstance(ty, list) else ty,
                            degree_y=d2,
                        )
                    ),
                )
            for fn in ("D.LPM", "D.UPM"):
                cid = f"{fn}|{pn}|{d1}|{d2}|{tn}"
                pyf = nns.d_lpm if fn == "D.LPM" else nns.d_upm
                add(
                    cid,
                    f"{fn}({d1}, {d2}, x, y, tx, ty)",
                    {"x": x.tolist(), "y": y.tolist(), "tx": tx, "ty": ty},
                    (
                        lambda pyf=pyf, d1=d1, d2=d2, x=x, y=y, tx=tx, ty=ty: pyf(
                            d1,
                            d2,
                            x,
                            y,
                            np.array(tx) if isinstance(tx, list) else tx,
                            np.array(ty) if isinstance(ty, list) else ty,
                        )
                    ),
                )
        # degree_y NULL default
        cid = f"Co.LPM.default|{pn}|{d1}"
        add(
            cid,
            f"Co.LPM({d1}, x, y, mean(x), mean(y))",
            {"x": x.tolist(), "y": y.tolist()},
            (
                lambda d1=d1, x=x, y=y: nns.co_lpm(
                    d1,
                    x,
                    y,
                    __import__("nns._rmean", fromlist=["r_mean"]).r_mean(x),
                    __import__("nns._rmean", fromlist=["r_mean"]).r_mean(y),
                )
            ),
        )

# nD
for nd in (2, 3, 4):
    X = rng.normal(size=(40, nd))
    Xt = np.round(X, 0)
    for Xn, XX in (("cont", X), ("ties", Xt)):
        for deg in (0, 1, 2, 0.5):
            for norm in (True, False):
                for tn, tgt in (("mean", XX.mean(axis=0)), ("zero", np.zeros(nd)), ("row", XX[3])):
                    for fn, pyf in (
                        ("Co.LPM_nD", nns.co_lpm_nd),
                        ("Co.UPM_nD", nns.co_upm_nd),
                        ("DPM_nD", nns.dpm_nd),
                    ):
                        cid = f"{fn}|{nd}|{Xn}|{deg}|{norm}|{tn}"
                        add(
                            cid,
                            f"{fn}(X, t, degree={deg}, norm={'TRUE' if norm else 'FALSE'})",
                            {"X": XX.tolist(), "t": tgt.tolist()},
                            (
                                lambda pyf=pyf, XX=XX, tgt=tgt, deg=deg, norm=norm: pyf(
                                    XX, tgt, degree=deg, norm=norm
                                )
                            ),
                        )

# PM.matrix
for nd in (2, 3):
    X = rng.normal(size=(30, nd))
    for lower, upper in ((1, 1), (2, 2), (0, 0), (1, 2), (0.5, 3)):
        for pop in (True, False):
            for norm in (True, False):
                for tn, tgt in (
                    ("mean", "mean"),
                    ("vec", X.mean(axis=0).tolist()),
                    ("zero", [0.0] * nd),
                ):
                    cid = f"PM.matrix|{nd}|{lower}|{upper}|{pop}|{norm}|{tn}"
                    texpr = '"mean"' if tgt == "mean" else "t"
                    add(
                        cid,
                        f"PM.matrix({lower}, {upper}, {texpr}, X, {'TRUE' if pop else 'FALSE'}, norm={'TRUE' if norm else 'FALSE'})",
                        {"X": X.tolist(), "t": tgt if tgt != "mean" else 0},
                        (
                            lambda X=X, lower=lower, upper=upper, tgt=tgt, pop=pop, norm=norm: (
                                nns.pm_matrix(
                                    lower,
                                    upper,
                                    tgt if tgt == "mean" else np.array(tgt),
                                    X,
                                    pop,
                                    norm=norm,
                                )
                            )
                        ),
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
            msgs.append(f"R error but py ok: R={r['error'][:120]}")
    elif perr is not None:
        msgs.append(f"py error but R ok: {perr[:160]}")
    else:
        rv = r["value"]
        if isinstance(py, dict):
            # moments / pm_matrix
            if cid.startswith("NNS.moments"):
                for k in ("mean", "variance", "skewness", "kurtosis"):
                    msgs += close(py[k], rget(rv, k), path=k)
            else:
                for k in rv["names"]:
                    pk = py.get(k)
                    msgs += close(np.asarray(pk), rget(rv, k), path=k)
        else:
            msgs += close(py, rv)
    problems[cid] = msgs

_j.dump({k: v for k, v in problems.items() if v}, open(f"{out_dir()}/pm_problems.json", "w"))
print(fmt_res("partial moments", problems, len(cases)))
