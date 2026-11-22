import re
import sys
import warnings

import numpy as np
from harness import fmt_res, run_r, tree_close

import nns

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 61
rng = np.random.default_rng(seed)
cases, pycalls = [], {}


def rlit(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str):
        return '"' + v + '"'
    return repr(v)


def pair(kind, n1, n2):
    if kind == "shift":
        return rng.normal(size=n1), rng.normal(size=n2) + 0.4
    if kind == "same":
        return rng.normal(size=n1), rng.normal(size=n2)
    if kind == "ties":
        return np.round(rng.normal(size=n1)), np.round(rng.normal(size=n2) + 0.3)
    if kind == "skew":
        return rng.exponential(size=n1), rng.exponential(size=n2) * 1.5
    if kind == "const":
        return np.full(n1, 2.0), np.full(n2, 2.0)


for kind in ("shift", "same", "ties", "skew", "const"):
    for n1, n2 in ((15, 20), (60, 45)):
        a, b = pair(kind, n1, n2)
        for mo in (False, True):
            for med in (False, True):
                for ci, tails in ((0.95, "Both"), (0.9, "left"), (0.8, "right"), (None, "Both")):
                    cid = f"anova|{kind}|{n1}|{mo}|{med}|{ci}|{tails}"
                    add_seed = 7
                    expr = f"set.seed({add_seed}); unlist(NNS.ANOVA(a, b, means.only = {rlit(mo)}, medians = {rlit(med)}, confidence.interval = {rlit(ci)}, tails = {rlit(tails)}, plot = FALSE))"
                    cases.append(
                        {"id": cid, "expr": expr, "data": {"a": a.tolist(), "b": b.tolist()}}
                    )
                    pycalls[cid] = lambda a=a, b=b, mo=mo, med=med, ci=ci, tails=tails: (
                        nns.nns_anova(
                            a,
                            b,
                            means_only=mo,
                            medians=med,
                            confidence_interval=ci,
                            tails=tails,
                            random_seed=7,
                        )
                    )
        for med in (False, True):
            for ci, tails in ((0.95, "Both"), (0.9, "left")):
                cid = f"robust|{kind}|{n1}|{med}|{ci}|{tails}"
                expr = f"set.seed(11); unlist(NNS.ANOVA(a, b, medians = {rlit(med)}, confidence.interval = {rlit(ci)}, tails = {rlit(tails)}, plot = FALSE, robust = TRUE))"
                cases.append({"id": cid, "expr": expr, "data": {"a": a.tolist(), "b": b.tolist()}})
                pycalls[cid] = lambda a=a, b=b, med=med, ci=ci, tails=tails: nns.nns_anova(
                    a,
                    b,
                    medians=med,
                    confidence_interval=ci,
                    tails=tails,
                    robust=True,
                    random_seed=11,
                )
for k in (2, 3, 4):
    groups = [rng.normal(size=int(rng.integers(10, 40))) + 0.2 * i for i in range(k)]
    for mo in (False, True):
        for med in (False, True):
            cid = f"multi|{k}|{mo}|{med}"
            expr = f"unlist(NNS.ANOVA(G, means.only = {rlit(mo)}, medians = {rlit(med)}, plot = FALSE))"
            cases.append({"id": cid, "expr": expr, "data": {"G": [g.tolist() for g in groups]}})
            pycalls[cid] = lambda groups=groups, mo=mo, med=med: nns.nns_anova(
                groups, means_only=mo, medians=med
            )
            cid = f"pairwise|{k}|{mo}|{med}"
            expr = f"as.numeric(NNS.ANOVA(G, means.only = {rlit(mo)}, medians = {rlit(med)}, pairwise = TRUE, plot = FALSE))"
            cases.append({"id": cid, "expr": expr, "data": {"G": [g.tolist() for g in groups]}})
            pycalls[cid] = lambda groups=groups, mo=mo, med=med: np.asarray(
                nns.nns_anova(groups, means_only=mo, medians=med, pairwise=True)
            ).ravel(order="F")
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
        rv = r["value"]
        if isinstance(py, dict):
            names = [re.sub(r"\.\d+(\.\d+)?%$", "", n) for n in rv.get("names", [])]
            vals = rv["v"].ravel()
            if sorted(names) != sorted(py.keys()):
                msgs.append(f"keys r={names} py={list(py.keys())}")
            for nm, val in zip(names, vals, strict=True):
                if nm in py:
                    msgs += tree_close(
                        py[nm],
                        {"__vec__": True, "t": "num", "v": np.array([val])},
                        f".{nm}",
                        rtol=0,
                        atol=0,
                    )
        else:
            msgs += tree_close(np.asarray(py), rv, rtol=0, atol=0)
    problems[cid] = msgs
print(fmt_res("ANOVA", problems, len(cases)))
