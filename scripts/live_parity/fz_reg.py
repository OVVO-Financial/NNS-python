import json
import json as _j
import sys

import numpy as np

sys.path.insert(0, ".")
from harness import fmt_res, out_dir, run_r, tree_close

import nns

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
cases, pycalls = [], {}


def add(cid, expr, data, fn):
    cases.append({"id": cid, "expr": expr, "data": data})
    pycalls[cid] = fn


def rlit(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str):
        return repr(v).replace("'", '"')
    if isinstance(v, (list, tuple, np.ndarray)):
        return "c(" + ",".join(repr(float(z)) for z in v) + ")"
    return repr(v)


def uni(kind, n):
    x = rng.normal(size=n)
    if kind == "lin":
        y = 2 * x + rng.normal(size=n) * 0.3
    elif kind == "quad":
        y = x**2 + rng.normal(size=n) * 0.1
    elif kind == "sin":
        y = np.sin(2 * x) + rng.normal(size=n) * 0.1
    elif kind == "rand":
        y = rng.normal(size=n)
    elif kind == "xties":
        x = np.round(x * 2) / 2
        y = x**3 + rng.normal(size=n) * 0.2
    elif kind == "step":
        y = np.floor(x) + rng.normal(size=n) * 0.01
    elif kind == "bin01":
        y = (x + rng.normal(size=n) * 0.5 > 0).astype(float)
    elif kind == "cls3":
        y = np.digitize(x + rng.normal(size=n) * 0.3, [-0.5, 0.5]).astype(float) + 1
    elif kind == "yconst":
        y = np.full(n, 2.0)
    elif kind == "expo":
        x = np.abs(x) * 3
        y = np.exp(x / 3) + rng.normal(size=n) * 0.1
    return x, y


for kind in ("lin", "quad", "sin", "rand", "xties", "step", "bin01", "cls3", "expo"):
    for n in (12, 60, 300):
        x, y = uni(kind, n)
        pe_opts = {
            "none": None,
            "in": np.array([float(np.quantile(x, 0.3)), float(np.quantile(x, 0.8))]),
            "out": np.array([float(x.min() - 1), float(x.max() + 0.5)]),
            "obs": x[:3].copy(),
        }
        combos = [
            (None, "off", "none", None, False, None),
            (None, "off", "in", 0.95, False, None),
            (1, "mean", "out", None, False, None),
            (2, "median", "obs", None, False, None),
            ("max", "off", "in", None, False, None),
            (3, "mode", "none", None, False, None),
            (None, "off", "out", 0.9, True, None),
            (None, "off", "in", None, False, "REG"),
            (2, "off", "in", None, False, "CLASS")
            if kind in ("bin01", "cls3")
            else (2, "off", "in", None, True, "REG"),
            (None, "mean", "in", None, False, None),
        ]
        for order, noise, pen, ci, smooth, typ in combos:
            pe = pe_opts[pen]
            cid = f"reg|{kind}|{n}|o={order}|{noise}|{pen}|ci={ci}|sm={smooth}|t={typ}"
            expr = (
                f"NNS.reg(x, y, order={rlit(order)}, noise.reduction={rlit(noise)}, point.est={rlit(pe)}, "
                f"confidence.interval={rlit(ci)}, smooth={rlit(smooth)}, type={rlit(typ)}, plot=FALSE)"
            )
            add(
                cid,
                expr,
                {"x": x.tolist(), "y": y.tolist()},
                lambda x=x, y=y, order=order, noise=noise, pe=pe, ci=ci, smooth=smooth, typ=typ: (
                    nns.nns_reg(
                        x,
                        y,
                        order=order,
                        noise_reduction=noise,
                        point_est=pe,
                        confidence_interval=ci,
                        smooth=smooth,
                        type=typ,
                    )
                ),
            )
        # point.only and multivariate.call
        pe = pe_opts["in"]
        add(
            f"reg.ponly|{kind}|{n}",
            "NNS.reg(x, y, point.est=pe, point.only=TRUE, plot=FALSE)",
            {"x": x.tolist(), "y": y.tolist(), "pe": pe.tolist()},
            lambda x=x, y=y, pe=pe: nns.nns_reg(x, y, point_est=pe, point_only=True),
        )
        add(
            f"reg.mvcall|{kind}|{n}",
            "NNS.reg(x, y, point.est=pe, multivariate.call=TRUE, plot=FALSE)",
            {"x": x.tolist(), "y": y.tolist(), "pe": pe.tolist()},
            lambda x=x, y=y, pe=pe: nns.nns_reg(x, y, point_est=pe, multivariate_call=True),
        )


def multi(kind, n, d):
    X = rng.normal(size=(n, d))
    if kind == "add":
        y = X[:, 0] + X[:, 1] ** 2 + rng.normal(size=n) * 0.2
    elif kind == "inter":
        y = X[:, 0] * X[:, 1] + rng.normal(size=n) * 0.2
    elif kind == "ties":
        X = np.round(X)
        y = X.sum(axis=1) + rng.normal(size=n) * 0.3
    elif kind == "bin01":
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
    elif kind == "cls3":
        y = np.digitize(X[:, 0] - X[:, 1], [-0.7, 0.7]).astype(float) + 1
    elif kind == "dummy":
        X[:, 1] = (X[:, 1] > 0).astype(float)
        y = X[:, 0] + 2 * X[:, 1] + rng.normal(size=n) * 0.2
    return X, y


for kind in ("add", "inter", "ties", "bin01", "cls3", "dummy"):
    for n, d in ((40, 2), (150, 3)):
        X, y = multi(kind, n, d)
        Xt = X[:4] + 0.05
        combos = [
            (None, None, 0, None, "off", None, None, "NNS"),
            (None, 1, 0, None, "off", 0.95, None, "NNS"),
            ("max", None, 0, None, "mean", None, None, "L1"),
            (2, "all", 0, None, "off", None, None, "L2"),
            (None, 3, 0, None, "median", None, None, "NNS"),
            ("cor", None, 0, None, "off", None, None, "NNS"),
            ("NNS.dep", None, 0.1, None, "off", None, None, "NNS"),
            ("NNS.caus", None, 0, None, "off", None, None, "NNS"),
            ("all", None, 0, None, "off", 0.95, None, "NNS"),
            ("equal", None, 0, 2, "off", None, None, "NNS"),
            ([0.5, -1.0] + [0.2] * (d - 2), None, 0, None, "off", None, None, "NNS"),
            (
                None,
                None,
                0,
                None,
                "off",
                None,
                "CLASS" if kind in ("bin01", "cls3") else "REG",
                "NNS",
            ),
            (None, None, 0, None, "off", None, None, "FACTOR"),
        ]
        for i, (dr, nb, thr, order, noise, ci, typ, dist) in enumerate(combos):
            if (
                dr in (None, 1, 2, 3, "all", "max")
                and not isinstance(dr, list)
                and dr not in ("cor", "NNS.dep", "NNS.caus", "equal", "all")
            ):
                # here first element encodes order when not a dim.red method
                pass
            is_dr = isinstance(dr, list) or dr in ("cor", "NNS.dep", "NNS.caus", "all", "equal")
            dim_red = dr if is_dr else None
            ordv = order if is_dr else dr
            if ordv is None and not is_dr:
                ordv = order
            cid = f"mreg|{kind}|{n}x{d}|{i}|dr={dim_red}|o={ordv}|nb={nb}|thr={thr}|{noise}|ci={ci}|t={typ}|{dist}"
            expr = (
                f"NNS.reg(X, y, point.est=Xt, dim.red.method={rlit(dim_red)}, order={rlit(ordv)}, n.best={rlit(nb)}, threshold={thr}, "
                f"noise.reduction={rlit(noise)}, confidence.interval={rlit(ci)}, type={rlit(typ)}, dist={rlit(dist)}, plot=FALSE)"
            )
            add(
                cid,
                expr,
                {"X": X.tolist(), "y": y.tolist(), "Xt": Xt.tolist()},
                lambda X=X, y=y, Xt=Xt, dim_red=dim_red, ordv=ordv, nb=nb, thr=thr, noise=noise, ci=ci, typ=typ, dist=dist: (
                    nns.nns_reg(
                        X,
                        y,
                        point_est=Xt,
                        dim_red_method=dim_red,
                        order=ordv,
                        n_best=nb,
                        threshold=thr,
                        noise_reduction=noise,
                        confidence_interval=ci,
                        type=typ,
                        dist=dist,
                    )
                ),
            )

print("cases", len(cases), flush=True)
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
            msgs.append(f"R error but py ok: R={r['error'][:150]}")
    elif perr is not None:
        msgs.append(f"py error but R ok: {perr[:200]}")
    else:
        msgs += tree_close(
            py,
            r["value"],
            rtol=float(__import__("os").environ.get("RTOL", "0")),
            atol=float(__import__("os").environ.get("ATOL", "0")),
        )
    problems[cid] = msgs
print(fmt_res("regression", problems, len(cases)))

_j.dump({k: v for k, v in problems.items() if v}, open(f"{out_dir()}/reg_problems.json", "w"))
json.dump(
    {k: v for k, v in problems.items() if v},
    open(f"{out_dir()}/fz_reg_problems.json", "w"),
    indent=1,
)
