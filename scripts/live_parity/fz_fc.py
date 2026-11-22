import cmath
import json
import math
import sys
import warnings

import numpy as np
from harness import fmt_res, out_dir, run_r, tree_close

import nns

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 31
only = sys.argv[2] if len(sys.argv) > 2 else "all"
rng = np.random.default_rng(seed)
cases, pycalls = [], {}


def rlit(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, (list, tuple, np.ndarray)):
        return "c(" + ",".join(rlit(z) for z in v) + ")"
    return repr(v)


def add(cid, expr, data, fn):
    if only != "all" and not cid.startswith(only):
        return
    cases.append({"id": cid, "expr": expr, "data": data})
    pycalls[cid] = fn


def series(kind, n):
    t = np.arange(n, dtype=float)
    if kind == "m12":
        return 10 + np.sin(2 * np.pi * t / 12) * 3 + rng.normal(size=n) * 0.4
    if kind == "trend":
        return 50 + 0.8 * t + np.sin(2 * np.pi * t / 7) * 4 + rng.normal(size=n)
    if kind == "rw":
        return 100 + np.cumsum(rng.normal(size=n))
    if kind == "ar":
        x = np.zeros(n)
        for i in range(2, n):
            x[i] = 0.6 * x[i - 1] - 0.2 * x[i - 2] + rng.normal()
        return x + 20
    if kind == "ties":
        return np.round(5 + np.sin(2 * np.pi * t / 6) * 2 + rng.normal(size=n) * 0.5)
    if kind == "mult":
        return (20 + t * 0.3) * (1 + 0.2 * np.sin(2 * np.pi * t / 12)) + rng.normal(size=n) * 0.3


for kind in ("m12", "trend", "rw", "ar", "ties", "mult"):
    for n in (48, 120):
        x = series(kind, n)
        d = {"x": x.tolist()}
        # seasonality
        for mod, monly, detrend, mp, disp in (
            (None, True, True, None, "auto"),
            (None, False, True, None, "auto"),
            (12, True, True, None, "auto"),
            ([6, 12], False, False, None, "auto"),
            (None, True, False, 20, "absolute"),
            (None, True, True, None, "absolute"),
        ):
            cid = f"seas|{kind}|{n}|{mod}|{monly}|{detrend}|{mp}|{disp}"
            add(
                cid,
                f"NNS.seas(x, modulo = {rlit(mod)}, mod.only = {rlit(monly)}, plot = FALSE, detrend = {rlit(detrend)}, max.period = {rlit(mp)}, dispersion = {rlit(disp)})",
                d,
                lambda x=x, mod=mod, monly=monly, detrend=detrend, mp=mp, disp=disp: nns.nns_seas(
                    x, modulo=mod, mod_only=monly, detrend=detrend, max_period=mp, dispersion=disp
                ),
            )
        # ARMA
        for h, sf, meth, w, bp, dyn, shrink, neg, pi_, ts in (
            (3, True, "nonlin", None, 1, False, False, False, None, None),
            (5, 12, "lin", None, 1, False, False, False, None, None),
            (4, [12, 6], "both", None, 1, False, False, False, None, None),
            (4, True, "lin", None, 2, False, True, False, None, None),
            (6, False, "nonlin", None, 1, True, False, True, None, None),
            (5, [3, 7], "lin", [0.7, 0.3], 1, False, False, False, 0.9, None),
            (3, True, "both", None, 1, False, True, False, 0.8, n - 6),
            (4, 12, "nonlin", None, 1, False, True, False, None, None),
        ):
            cid = f"arma|{kind}|{n}|h={h}|sf={sf}|{meth}|w={w}|bp={bp}|dyn={dyn}|shr={shrink}|neg={neg}|pi={pi_}|ts={ts}"
            expr = (
                f"set.seed(17); NNS.ARMA(x, h = {h}, training.set = {rlit(ts)}, seasonal.factor = {rlit(sf)}, weights = {rlit(w)}, best.periods = {bp}, "
                f"negative.values = {rlit(neg)}, method = {rlit(meth)}, dynamic = {rlit(dyn)}, shrink = {rlit(shrink)}, plot = FALSE, seasonal.plot = FALSE, pred.int = {rlit(pi_)})"
            )
            add(
                cid,
                expr,
                d,
                lambda x=x, h=h, sf=sf, meth=meth, w=w, bp=bp, dyn=dyn, shrink=shrink, neg=neg, pi_=pi_, ts=ts: (
                    nns.nns_arma(
                        x,
                        h=h,
                        training_set=ts,
                        seasonal_factor=sf,
                        weights=None if w is None else np.array(w),
                        best_periods=bp,
                        negative_values=neg,
                        method=meth,
                        dynamic=dyn,
                        shrink=shrink,
                        pred_int=pi_,
                        random_seed=17,
                    )
                ),
            )
        # ARMA.optim (moderate cost)
        if n == 48 or kind in ("m12", "trend"):
            for h, sf, lin, bc, pi_, ts in (
                (3, [12, 6, 4], False, "none", 0.95, None),
                (4, [3, 6, 12], True, "split", 0.9, None),
                (2, 1, False, "none", None, n - 10),
            ):
                cid = f"optim|{kind}|{n}|h={h}|sf={sf}|lin={lin}|bc={bc}|pi={pi_}|ts={ts}"
                expr = (
                    f"NNS.ARMA.optim(x, h = {h}, training.set = {rlit(ts)}, seasonal.factor = {rlit(sf)}, lin.only = {rlit(lin)}, "
                    f"pred.int = {rlit(pi_)}, print.trace = FALSE, plot = FALSE, bias.correction = {rlit(bc)}, ncores = 1)"
                )
                add(
                    cid,
                    expr,
                    d,
                    lambda x=x, h=h, sf=sf, lin=lin, bc=bc, pi_=pi_, ts=ts: nns.nns_arma_optim(
                        x,
                        h=h,
                        training_set=ts,
                        seasonal_factor=np.atleast_1d(sf).tolist(),
                        lin_only=lin,
                        pred_int=pi_,
                        print_trace=False,
                        bias_correction=bc,
                        ncores=1,
                    ),
                )

# derivatives


def _cx(real_fn, cx_fn):
    return lambda v: cx_fn(v) if isinstance(v, complex) else real_fn(v)


funcs = {
    "cube": ("function(v) v^3", lambda v: v**3),
    "sin": ("function(v) sin(v)", _cx(math.sin, cmath.sin)),
    "exp": ("function(v) exp(v / 2)", _cx(lambda v: math.exp(v / 2), lambda v: cmath.exp(v / 2))),
    "abs": ("function(v) abs(v)", lambda v: abs(v)),
    "log": ("function(v) log(v)", _cx(math.log, cmath.log)),
    "poly": ("function(v) 3*v^2 - 2*v + 1", lambda v: 3 * (v * v) - 2 * v + 1),
}
for fn, (rf, pf) in funcs.items():
    for pt in (0.5, 1.5, 3.0, -2.0):
        if fn == "log" and pt <= 0:
            continue
        for h, tol, digits in ((None, 1e-10, 12), (0.5, 1e-8, 8), (None, 1e-6, 6)):
            cid = f"diff|{fn}|{pt}|{h}|{tol}|{digits}"
            hexpr = "" if h is None else f", h = {h}"
            add(
                cid,
                f"res <- NNS.diff({rf}, {pt}{hexpr}, tol = {tol}, digits = {digits}); setNames(as.numeric(res[, 1]), rownames(res))",
                {},
                lambda pf=pf, pt=pt, h=h, tol=tol, digits=digits: nns.nns_diff(
                    pf, pt, h=h, tol=tol, digits=digits
                ),
            )
for kind in ("quad", "sin", "noisy", "ties", "cubic"):
    for n in (30, 120, 400):
        x = np.sort(rng.uniform(-3, 3, n))
        if kind == "quad":
            y = x**2
        elif kind == "sin":
            y = np.sin(2 * x)
        elif kind == "noisy":
            y = x**3 + rng.normal(size=n) * 0.5
        elif kind == "ties":
            x = np.round(x, 1)
            y = np.exp(x / 3)
        else:
            y = x**3 - x
        for ep in (None, [0.0], [-1.5, 0.2, 2.0], "obs"):
            cid = f"dydx|{kind}|{n}|{ep}"
            epx = "NULL" if ep is None else ("'obs'" if ep == "obs" else rlit(ep))
            add(
                cid,
                f"dy.dx(x, y, eval.point = {epx})",
                {"x": x.tolist(), "y": y.tolist()},
                lambda x=x, y=y, ep=ep: nns.dy_dx(
                    x, y, eval_point=None if ep is None else (ep if ep == "obs" else np.array(ep))
                ),
            )
for kind in ("add", "inter", "ties"):
    for n in (60, 200):
        X = rng.normal(size=(n, 2))
        if kind == "add":
            y = X[:, 0] ** 2 + np.sin(X[:, 1])
        elif kind == "inter":
            y = X[:, 0] * X[:, 1] + rng.normal(size=n) * 0.1
        else:
            X = np.round(X, 1)
            y = X[:, 0] + X[:, 1] ** 2
        for wrt in (1, 2):
            for ep in ("obs", "pt"):
                epx = "'obs'" if ep == "obs" else "matrix(c(0.1, -0.3, 0.5, 0.2), ncol = 2)"
                cid = f"dyd|{kind}|{n}|wrt={wrt}|{ep}"
                add(
                    cid,
                    f"res <- dy.d_(X, y, wrt = {wrt}, eval.points = {epx}, messages = FALSE); list(First = res['First', 1][[1]], Second = res['Second', 1][[1]])",
                    {"X": X.tolist(), "y": y.tolist()},
                    lambda X=X, y=y, wrt=wrt, ep=ep: nns.dy_d(
                        X,
                        y,
                        wrt=wrt,
                        eval_points="obs" if ep == "obs" else np.array([[0.1, 0.5], [-0.3, 0.2]]),
                        messages=False,
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
        msgs += tree_close(
            py,
            r["value"],
            rtol=float(__import__("os").environ.get("RTOL", "0")),
            atol=float(__import__("os").environ.get("ATOL", "0")),
        )
    problems[cid] = msgs
print(fmt_res("forecasting/derivatives", problems, len(cases)))
json.dump(
    {k: v for k, v in problems.items() if v},
    open(f"{out_dir()}/fz_fc_{only}_{seed}.json", "w"),
    indent=1,
)
print("R secs", round(sum(R[c["id"]]["secs"] for c in cases), 1))
