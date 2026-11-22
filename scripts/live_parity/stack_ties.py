"""Classification stacks built to tie: coarse predictors, small samples, pooled folds."""

import json
import os
import sys
import warnings

import numpy as np
from harness import fmt_res, out_dir, run_r, tree_close

import nns

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
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


for K in (2, 3, 4):
    for n in (14, 23, 40):
        for rep in range(2):
            d = int(rng.integers(1, 4))
            X = rng.integers(0, 3, size=(n, d)).astype(float)
            y = rng.integers(1, K + 1, size=n).astype(float)
            y[:K] = np.arange(1, K + 1)
            rng.shuffle(y)
            Xt = rng.integers(0, 3, size=(5, d)).astype(float)
            for i, c in enumerate(
                [
                    dict(
                        method=[1, 2],
                        folds=3,
                        cv=0.4,
                        dr="cor",
                        bal=False,
                        pi=None,
                        s=int(rng.integers(1, 999)),
                    ),
                    dict(
                        method=[1],
                        folds=4,
                        cv=0.5,
                        dr="cor",
                        bal=True,
                        pi=0.8,
                        s=int(rng.integers(1, 999)),
                    ),
                    dict(
                        method=[2],
                        folds=3,
                        cv=0.35,
                        dr="all",
                        bal=False,
                        pi=0.6,
                        s=int(rng.integers(1, 999)),
                    ),
                    dict(
                        method=[1, 2],
                        folds=2,
                        cv=None,
                        dr="equal",
                        bal=True,
                        pi=None,
                        s=int(rng.integers(1, 999)),
                    ),
                ]
            ):
                cid = f"ties|K{K}|n{n}|r{rep}|{i}|{c}"
                expr = (
                    f"NNS.stack(X, y, IVs.test = Xt, method = {rlit(c['method'])}, folds = {c['folds']}, "
                    f"CV.size = {rlit(c['cv'])}, dim.red.method = {rlit(c['dr'])}, balance = {rlit(c['bal'])}, "
                    f"pred.int = {rlit(c['pi'])}, type = 'CLASS', seed = {c['s']}L, status = FALSE, ncores = 1)"
                )
                cases.append(
                    {
                        "id": cid,
                        "expr": expr,
                        "data": {"X": X.tolist(), "y": y.tolist(), "Xt": Xt.tolist()},
                    }
                )
                pycalls[cid] = lambda X=X, y=y, Xt=Xt, c=c: nns.nns_stack(
                    X,
                    y,
                    Xt,
                    method=tuple(c["method"]),
                    folds=c["folds"],
                    cv_size=c["cv"],
                    dim_red_method=c["dr"],
                    balance=c["bal"],
                    pred_int=c["pi"],
                    type="CLASS",
                    seed=c["s"],
                    ncores=1,
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
            rtol=float(os.environ.get("RTOL", "0")),
            atol=float(os.environ.get("ATOL", "0")),
        )
    problems[cid] = msgs
print(fmt_res("stack ties", problems, len(cases)))
json.dump(
    {k: v for k, v in problems.items() if v},
    open(f"{out_dir()}/stack_ties_{seed}.json", "w"),
    indent=1,
)
