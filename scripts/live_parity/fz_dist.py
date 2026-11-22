"""Differential sweep: NNS.distance (scalar) and NNS.distance.bulk at exact equality."""

import sys
import warnings

import numpy as np

sys.path.insert(0, ".")
from harness import close, fmt_res, run_r

import nns

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 77)
cases, pycalls = [], {}


def add(cid, expr, data, fn):
    cases.append({"id": cid, "expr": expr, "data": data})
    pycalls[cid] = fn


def r_class(cls: bool) -> str:
    """The R literal for NNS.distance's class argument."""
    return '"class"' if cls else "NULL"


def rpm_expr(p):
    cols = ", ".join(f"X{j + 1} = f[, {j + 1}]" for j in range(p))
    return f"f <- matrix(fv, ncol = {p}); rpm <- data.frame({cols}, y.hat = yh)"


for kind in ("cont", "ties", "grid", "dup"):
    for n, p in ((12, 1), (30, 2), (60, 3), (150, 2)):
        if kind == "cont":
            f = rng.normal(size=(n, p))
        elif kind == "ties":
            f = np.round(rng.normal(size=(n, p)) * 1.5)
        elif kind == "grid":
            f = rng.integers(0, 4, size=(n, p)).astype(float)
        else:
            base = rng.normal(size=(max(2, n // 4), p))
            f = base[rng.integers(0, base.shape[0], size=n)]
        yh = np.round(rng.normal(size=n) * 3, 2)
        ycls = rng.integers(1, 4, size=n).astype(float)
        dests = [f[0].copy(), np.round(rng.normal(size=p), 1), f.mean(axis=0)]
        xt = np.vstack([f[:3], np.round(rng.normal(size=(3, p)), 0)])
        for di, dest in enumerate(dests):
            for k, rk in (
                (1, "1"),
                (2, "2"),
                (3, "3"),
                (5, "5"),
                ("all", "'all'"),
                (0, "0"),
                (2.7, "2.7"),
                (float("inf"), "Inf"),
            ):
                for cls in (False, True):
                    y = ycls if cls else yh
                    cid = f"dist|{kind}|{n}|{p}|d{di}|k={k}|cls={cls}"
                    add(
                        cid,
                        rpm_expr(p)
                        + f"; NNS.distance(rpm, dest, k = {rk}, class = {r_class(cls)})",
                        {
                            "fv": f.ravel(order="F").tolist(),
                            "yh": y.tolist(),
                            "dest": dest.tolist(),
                        },
                        lambda f=f, y=y, dest=dest, k=k, cls=cls: nns.nns_distance(
                            np.column_stack([f, y]), dest, k=k, class_="class" if cls else None
                        ),
                    )
        for k, rk in ((1, "1"), (2, "2"), (4, "4"), ("all", "'all'"), (0, "0")):
            for cls in (False, True):
                y = ycls if cls else yh
                cid = f"bulk|{kind}|{n}|{p}|k={k}|cls={cls}"
                add(
                    cid,
                    rpm_expr(p)
                    + f"; NNS:::NNS.distance.bulk(rpm, matrix(xv, ncol = {p}), k = {rk}, "
                    f"class = {r_class(cls)})",
                    {
                        "fv": f.ravel(order="F").tolist(),
                        "yh": y.tolist(),
                        "xv": xt.ravel(order="F").tolist(),
                    },
                    lambda f=f, y=y, xt=xt, k=k, cls=cls: nns.nns_distance_bulk(
                        np.column_stack([f, y]), xt, k=k, class_="class" if cls else None
                    ),
                )

print("cases", len(cases), flush=True)
R = run_r(cases)
problems = {}
for c in cases:
    cid = c["id"]
    r = R[cid]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        msgs += close(np.atleast_1d(py), r["value"], rtol=0, atol=0)
    problems[cid] = msgs
print(fmt_res("distance", problems, len(cases)))
