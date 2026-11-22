import json
import sys
import warnings

import numpy as np
from harness import fmt_res, out_dir, run_r, tree_close

import nns

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
which = sys.argv[2] if len(sys.argv) > 2 else "both"
rng = np.random.default_rng(seed)
cases, pycalls = [], {}


def rlit(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, (list, tuple)):
        return "c(" + ",".join(rlit(z) for z in v) + ")"
    return repr(v)


def add(cid, expr, data, fn):
    cases.append({"id": cid, "expr": expr, "data": data})
    pycalls[cid] = fn


def design(kind, n, d):
    X = rng.normal(size=(n, d))
    if kind == "reg":
        y = X[:, 0] + np.sin(X[:, 1]) + rng.normal(size=n) * 0.3
    elif kind == "bin":
        y = (X[:, 0] - X[:, 1] + rng.normal(size=n) * 0.5 > 0).astype(float)
    elif kind == "cls3":
        y = np.digitize(X[:, 0] + rng.normal(size=n) * 0.3, [-0.5, 0.5]).astype(float) + 1
    elif kind == "ties":
        X = np.round(X)
        y = X[:, 0] * 2 - X[:, 1] + rng.normal(size=n) * 0.2
    return X, y


if which in ("both", "stack"):
    for kind in ("reg", "bin", "cls3", "ties"):
        for n, d in ((60, 2), (120, 3)):
            X, y = design(kind, n, d)
            Xt = rng.normal(size=(7, d))
            combos = [
                dict(
                    method=[1, 2],
                    folds=2,
                    cv=None,
                    dr="cor",
                    order=None,
                    ts=None,
                    pi=None,
                    bal=False,
                    typ=None,
                    s=123,
                ),
                dict(
                    method=[1],
                    folds=1,
                    cv=None,
                    dr="cor",
                    order=None,
                    ts=None,
                    pi=None,
                    bal=False,
                    typ=None,
                    s=7,
                ),
                dict(
                    method=[2],
                    folds=2,
                    cv=0.25,
                    dr="all",
                    order=None,
                    ts=None,
                    pi=None,
                    bal=False,
                    typ=None,
                    s=11,
                ),
                dict(
                    method=[1, 2],
                    folds=2,
                    cv=None,
                    dr="NNS.dep",
                    order=2,
                    ts=None,
                    pi=0.9 if kind == "reg" else None,
                    bal=False,
                    typ=None,
                    s=123,
                ),
                dict(
                    method=[1, 2],
                    folds=1,
                    cv=None,
                    dr="equal",
                    order=None,
                    ts=10,
                    pi=None,
                    bal=False,
                    typ=None,
                    s=5,
                ),
            ]
            if kind in ("bin", "cls3"):
                combos.append(
                    dict(
                        method=[1, 2],
                        folds=2,
                        cv=None,
                        dr="cor",
                        order=None,
                        ts=None,
                        pi=None,
                        bal=True,
                        typ="CLASS",
                        s=3,
                    )
                )
            for i, c in enumerate(combos):
                cid = f"stack|{kind}|{n}x{d}|{i}|{c}"
                expr = (
                    f"NNS.stack(X, y, IVs.test = Xt, method = {rlit(c['method'])}, folds = {c['folds']}, CV.size = {rlit(c['cv'])}, "
                    f"dim.red.method = {rlit(c['dr'])}, order = {rlit(c['order'])}, ts.test = {rlit(c['ts'])}, pred.int = {rlit(c['pi'])}, "
                    f"balance = {rlit(c['bal'])}, type = {rlit(c['typ'])}, seed = {c['s']}L, status = FALSE, ncores = 1)"
                )
                add(
                    cid,
                    expr,
                    {"X": X.tolist(), "y": y.tolist(), "Xt": Xt.tolist()},
                    lambda X=X, y=y, Xt=Xt, c=c: nns.nns_stack(
                        X,
                        y,
                        Xt,
                        method=tuple(c["method"]),
                        folds=c["folds"],
                        cv_size=c["cv"],
                        dim_red_method=c["dr"],
                        order=c["order"],
                        ts_test=c["ts"],
                        pred_int=c["pi"],
                        balance=c["bal"],
                        type=c["typ"],
                        seed=c["s"],
                        ncores=1,
                    ),
                )
if which in ("both", "boost"):
    for kind in ("reg", "bin", "cls3"):
        for n, d in ((80, 3),):
            X, y = design(kind, n, d)
            Xt = rng.normal(size=(6, d))
            combos = [
                dict(
                    trials=10,
                    epochs=None,
                    cv=None,
                    depth=None,
                    fo=False,
                    ts=None,
                    bal=False,
                    s=123,
                    pi=None,
                ),
                dict(
                    trials=6, epochs=3, cv=0.3, depth=2, fo=False, ts=None, bal=False, s=9, pi=None
                ),
                dict(
                    trials=6,
                    epochs=None,
                    cv=None,
                    depth="max",
                    fo=True,
                    ts=None,
                    bal=kind != "reg",
                    s=4,
                    pi=None,
                ),
                dict(
                    trials=5,
                    epochs=None,
                    cv=None,
                    depth=None,
                    fo=False,
                    ts=8,
                    bal=False,
                    s=123,
                    pi=0.9 if kind != "cls3" else None,
                ),
            ]
            for i, c in enumerate(combos):
                cid = f"boost|{kind}|{n}x{d}|{i}|{c}"
                expr = (
                    f"NNS.boost(X, y, IVs.test = Xt, learner.trials = {c['trials']}, epochs = {rlit(c['epochs'])}, CV.size = {rlit(c['cv'])}, "
                    f"depth = {rlit(c['depth'])}, features.only = {rlit(c['fo'])}, ts.test = {rlit(c['ts'])}, balance = {rlit(c['bal'])}, "
                    f"pred.int = {rlit(c['pi'])}, seed = {c['s']}L, status = FALSE, feature.importance = FALSE)"
                )
                add(
                    cid,
                    expr,
                    {"X": X.tolist(), "y": y.tolist(), "Xt": Xt.tolist()},
                    lambda X=X, y=y, Xt=Xt, c=c: nns.nns_boost(
                        X,
                        y,
                        Xt,
                        learner_trials=c["trials"],
                        epochs=c["epochs"],
                        cv_size=c["cv"],
                        depth=c["depth"],
                        features_only=c["fo"],
                        ts_test=c["ts"],
                        balance=c["bal"],
                        pred_int=c["pi"],
                        seed=c["s"],
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
print(fmt_res("stack/boost", problems, len(cases)))
json.dump(
    {k: v for k, v in problems.items() if v},
    open(f"{out_dir()}/fz_stack_{which}_{seed}.json", "w"),
    indent=1,
)
print("R secs total", sum(R[c["id"]]["secs"] for c in cases))
