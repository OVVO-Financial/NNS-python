"""Differential sweep: smooth.spline(x, y, spar) coef/knots/predictions at exact equality."""

import sys
import warnings

import numpy as np

sys.path.insert(0, ".")
from harness import close, fmt_res, rget, run_r

from nns._rsmspline import smooth_spline

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
cases, pycalls = [], {}
for kind in ("unif", "sorted", "dups", "clustered", "wide"):
    for n in (4, 7, 12, 49, 50, 51, 120, 260, 900):
        if kind == "unif":
            x = rng.uniform(-3, 5, n)
        elif kind == "sorted":
            x = np.sort(rng.normal(size=n)) * 10
        elif kind == "dups":
            x = np.round(rng.uniform(0, 3, n), 1)
        elif kind == "clustered":
            x = np.concatenate([rng.normal(0, 0.01, n // 2), rng.normal(5, 2, n - n // 2)])
        else:
            x = rng.uniform(1e5, 1e5 + 1e3, n)
        y = np.sin(x / (np.ptp(x) + 1) * 6) + rng.normal(size=n) * 0.3
        if (
            np.unique(
                np.round((x - x.mean()) / (1e-6 * np.subtract(*np.percentile(x, [75, 25]))))
            ).size
            < 4
        ):
            continue
        lo, hi = float(np.min(x)), float(np.max(x))
        xo = np.concatenate(
            [np.linspace(lo, hi, 7), [lo - 0.3 * (hi - lo), hi + 0.25 * (hi - lo), lo, hi]]
        )
        for spar in (0.25, 0.4, 0.5, 0.6, 0.75, 1.0 / 3.0):
            cid = f"{kind}|{n}|{spar:.4f}"
            cases.append(
                {
                    "id": cid,
                    "expr": f"f <- smooth.spline(x, y, spar = {spar!r}); list(coef = f$fit$coef, knot = f$fit$knot, p = predict(f, xo)$y)",
                    "data": {"x": x.tolist(), "y": y.tolist(), "xo": xo.tolist()},
                }
            )
            pycalls[cid] = lambda x=x, y=y, xo=xo, spar=spar: (
                lambda f: {"coef": f.coef[1:], "knot": f.knot[1:], "p": f.predict(xo)}
            )(smooth_spline(x, y, spar))
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
        for key in ("knot", "coef", "p"):
            msgs += close(py[key], rget(r["value"], key), rtol=0, atol=0, path=key)
    problems[cid] = msgs
print(fmt_res("smooth.spline", problems, len(cases)))
