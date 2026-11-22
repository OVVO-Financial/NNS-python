"""Differential sweep: NNS.meboot / NNS.MC / NNS.SS / NNS.ARMA pred.int under set.seed."""

import sys
import warnings

import numpy as np
from harness import fmt_res, out_dir, run_r, tree_close

import nns

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 101
only = sys.argv[2] if len(sys.argv) > 2 else "all"
rng = np.random.default_rng(seed)
cases, pycalls, post = [], {}, {}


def rlit(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, (list, tuple, np.ndarray)):
        return "c(" + ",".join(rlit(z) for z in v) + ")"
    return repr(float(v)) if isinstance(v, (float, np.floating)) else repr(v)


def add(cid, expr, data, fn):
    if only != "all" and not cid.startswith(only):
        return
    cases.append({"id": cid, "expr": expr, "data": data})
    pycalls[cid] = fn


def series(kind, n):
    t = np.arange(n, dtype=float)
    if kind == "trend":
        return 5 + 0.3 * t + rng.normal(size=n)
    if kind == "rw":
        return 100 + np.cumsum(rng.normal(size=n))
    if kind == "seas":
        return 10 + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(size=n) * 0.5
    if kind == "ties":
        return np.round(rng.normal(size=n) * 2) + 0.1 * t
    if kind == "offset":
        return 1e6 + rng.normal(size=n)
    if kind == "linear":
        return 2.0 + 0.5 * t
    if kind == "const":
        return np.full(n, 3.25)
    raise ValueError(kind)


MEBOOT_R = (
    "r <- NNS.meboot(x, {args}); if (!length(r)) list() else "
    "lapply(seq_len(ncol(r)), function(j) {{ e <- r[, j]; e$replicates <- unname(e$replicates); e }})"
)


def meboot_py(x, kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = nns.nns_meboot(x, **kw)
    if isinstance(res, dict):
        return [] if not res else [res]
    return res


OPTS = [
    ({"reps": 5, "rho": 0.5}, "reps = 5, rho = 0.5"),
    ({"reps": 1, "rho": -1.0}, "reps = 1, rho = -1"),
    ({"reps": 2, "rho": 1.0}, "reps = 2, rho = 1"),
    ({"reps": 6, "rho": [-1.0, 0.0, 0.7]}, "reps = 6, rho = c(-1, 0, 0.7)"),
    ({"reps": 4, "rho": 0.3, "type": "pearson"}, "reps = 4, rho = 0.3, type = 'pearson'"),
    ({"reps": 3, "rho": 0.6, "type": "nnsdep"}, "reps = 3, rho = 0.6, type = 'nnsdep'"),
    ({"reps": 3, "rho": -0.4, "type": "NNScor"}, "reps = 3, rho = -0.4, type = 'NNScor'"),
    ({"reps": 4, "target_drift": 0.05}, "reps = 4, target_drift = 0.05"),
    (
        {"reps": 4, "rho": 0.2, "target_drift_scale": [0.5, 2.0]},
        "reps = 4, rho = 0.2, target_drift_scale = c(0.5, 2)",
    ),
    ({"reps": 4, "rho": 0.9, "drift": False}, "reps = 4, rho = 0.9, drift = FALSE"),
    (
        {"reps": 5, "rho": 0.5, "xmin": 0.0, "xmax": 40.0},
        "reps = 5, rho = 0.5, xmin = 0, xmax = 40",
    ),
    (
        {"reps": 5, "rho": 0.5, "reachbnd": False, "trim": 0.25},
        "reps = 5, rho = 0.5, reachbnd = FALSE, trim = 0.25",
    ),
    (
        {"reps": 5, "rho": 0.5, "expand_sd": False, "force_clt": False},
        "reps = 5, rho = 0.5, expand.sd = FALSE, force.clt = FALSE",
    ),
    (
        {"reps": 5, "rho": 0.5, "scl_adjustment": True, "sym": True},
        "reps = 5, rho = 0.5, scl.adjustment = TRUE, sym = TRUE",
    ),
    (
        {"reps": 5, "rho": 0.5, "expand_residual_sd": False, "digits": 3},
        "reps = 5, rho = 0.5, expand.residual.sd = FALSE, digits = 3",
    ),
    ({"reps": 5, "rho": 0.5, "trim": 0.0}, "reps = 5, rho = 0.5, trim = 0"),
    ({"reps": 5, "rho": 0.5, "trim": 0.6}, "reps = 5, rho = 0.5, trim = 0.6"),
    ({"reps": 3}, "reps = 3"),
]

for kind in ("trend", "rw", "seas", "ties", "offset", "linear", "const"):
    for n in (2, 3, 13, 60):
        if kind in ("linear", "const") and n == 3:
            continue
        x = series(kind, n)
        for k, (kw, rargs) in enumerate(OPTS):
            if n == 60 and k % 3:
                continue
            s = int(rng.integers(1, 10_000))
            cid = f"meboot|{kind}|{n}|{k}|s{s}"
            kw2 = dict(kw, random_seed=s)
            add(
                cid,
                f"set.seed({s}); " + MEBOOT_R.format(args=rargs),
                {"x": x.tolist()},
                lambda x=x, kw2=kw2: meboot_py(x, kw2),
            )

for kind in ("trend", "rw", "seas", "ties"):
    for n in (8, 40):
        x = series(kind, n)
        for lo, hi, by, ex, reps, extra_r, extra_py in (
            (-1, 1, 0.5, 1, 5, "", {}),
            (-1, 1, 0.25, 2, 3, "", {}),
            (-0.5, 0.8, 0.1, 0.5, 4, "", {}),
            (0.2, 0.9, 0.35, 3, 3, ", target_drift = 0.02", {"target_drift": 0.02}),
            (
                -1,
                0,
                0.5,
                1,
                2,
                ", target_drift_scale = c(1, 0.5, 2)",
                {"target_drift_scale": [1.0, 0.5, 2.0]},
            ),
            (
                -1,
                1,
                1,
                1,
                3,
                ", type = 'pearson', drift = FALSE",
                {"type": "pearson", "drift": False},
            ),
            (0, 0, 0.1, 1, 3, "", {}),
            (-1, 1, 0.0001, 1, 1, ", xmin = -50, xmax = 500", {"xmin": -50.0, "xmax": 500.0}),
        ):
            if by == 0.0001 and n > 8:
                continue
            s = int(rng.integers(1, 10_000))
            cid = f"mc|{kind}|{n}|{lo}|{hi}|{by}|{ex}|{reps}|{extra_r}|s{s}"
            expr = (
                f"set.seed({s}); r <- NNS.MC(x, reps = {reps}, lower_rho = {lo}, upper_rho = {hi}, by = {by}, exp = {ex}{extra_r}); "
                "list(ensemble = r$ensemble, names = names(r$replicates), reps = unname(r$replicates))"
            )

            def mc_py(x=x, lo=lo, hi=hi, by=by, ex=ex, reps=reps, extra_py=extra_py, s=s):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = nns.nns_mc(
                        x,
                        reps=reps,
                        lower_rho=lo,
                        upper_rho=hi,
                        by=by,
                        exp=ex,
                        random_seed=s,
                        **extra_py,
                    )
                return {
                    "ensemble": r["ensemble"],
                    "names": list(r["replicates"].keys()),
                    "reps": list(r["replicates"].values()),
                }

            add(cid, expr, {"x": x.tolist()}, mc_py)

for n1, n2 in ((15, 20), (40, 33)):
    for kind in ("shift", "ties", "const"):
        if kind == "shift":
            a, b = rng.normal(size=n1) + 0.3, rng.normal(size=n2)
        elif kind == "ties":
            a, b = np.round(rng.normal(size=n1) * 2), np.round(rng.normal(size=n2) * 2)
        else:
            a, b = np.full(n1, 1.5), rng.normal(size=n2)
        for reps, ci, rho in ((30, 0.95, 1.0), (25, 0.8, 0.0), (10, 0.9, -0.5)):
            s = int(rng.integers(1, 10_000))
            cid = f"ss|{kind}|{n1}|{reps}|{ci}|{rho}|s{s}"
            add(
                cid,
                f"set.seed({s}); NNS.SS(x, y, confidence.interval = TRUE, reps = {reps}, ci = {ci}, rho = {rho})",
                {"x": a.tolist(), "y": b.tolist()},
                lambda a=a, b=b, reps=reps, ci=ci, rho=rho, s=s: nns.nns_ss(
                    a, b, confidence_interval=True, reps=reps, ci=ci, rho=rho, random_seed=s
                ),
            )

for kind in ("seas", "trend", "rw", "ties"):
    for n in (48, 90):
        x = series(kind, n)
        for h, sf, meth, pi_ in (
            (4, True, "nonlin", 0.95),
            (5, 12, "lin", 0.9),
            (3, [12, 6], "both", 0.8),
            (1, True, "lin", 0.95),
            (6, False, "means", 0.95),
        ):
            s = int(rng.integers(1, 10_000))
            cid = f"arma|{kind}|{n}|h={h}|sf={sf}|{meth}|pi={pi_}|s{s}"
            expr = (
                f"set.seed({s}); NNS.ARMA(x, h = {h}, seasonal.factor = {rlit(sf)}, method = {rlit(meth)}, "
                f"plot = FALSE, seasonal.plot = FALSE, pred.int = {pi_})"
            )
            add(
                cid,
                expr,
                {"x": x.tolist()},
                lambda x=x, h=h, sf=sf, meth=meth, pi_=pi_, s=s: nns.nns_arma(
                    x, h=h, seasonal_factor=sf, method=meth, pred_int=pi_, random_seed=s
                ),
            )

print("cases", len(cases), flush=True)
if "DUMP" in __import__("os").environ:
    import json

    json.dump(cases, open(f"{out_dir()}/dump_cases.json", "w"))
    raise SystemExit(0)
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
        if "TRACE" in __import__("os").environ:
            __import__("traceback").print_exc()
    msgs = []
    if not r["ok"]:
        if perr is None:
            msgs.append(f"R error but py ok: R={r['error'][:150]}")
    elif perr is not None:
        msgs.append(f"py error but R ok: {perr[:200]}")
    else:
        msgs += tree_close(py, r["value"], rtol=0, atol=0)
    problems[cid] = msgs
print(fmt_res("BOOT", problems, len(cases)))
