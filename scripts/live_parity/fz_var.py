import sys
import warnings

import numpy as np
from harness import fmt_res, run_r, tree_close

import nns

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 41
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


def panel(n, kind):
    t = np.arange(1, n + 1)
    A = np.sin(2 * np.pi * t / 12) * 2 + rng.normal(size=n) * 0.3 + 10
    B = 0.6 * np.r_[A[0], A[:-1]] + rng.normal(size=n) * 0.3 + 3
    C = np.cumsum(rng.normal(size=n)) + 20
    X = np.column_stack([A, B, C])
    if kind == "gap":
        X[-2:, 2] = np.nan
    if kind == "gap2":
        X[-1, 0] = np.nan
        X[-3:, 1] = np.nan
    if kind == "mid":
        X[10:12, 1] = np.nan
    return X


combos = [
    (3, 1, "cor", "pipeline", True),
    (4, 2, "NNS.dep", "pipeline", True),
    (3, 1, "all", "observed", False),
    (2, 1, "cor", "observed", True),
    (3, 3, "cor", "pipeline", False),
]
for kind in ("clean", "gap", "gap2", "mid"):
    for n in (60, 96):
        X = panel(n, kind)
        for h, tau, dr, val, nw in combos:
            cid = f"var|{kind}|{n}|h={h}|tau={tau}|{dr}|{val}|nw={nw}"
            expr = (
                f"X <- Xm; colnames(X) <- c('A', 'B', 'C'); r <- NNS.VAR(X, h = {h}, tau = {tau}, dim.red.method = {rlit(dr)}, naive.weights = {rlit(nw)}, "
                f"status = FALSE, ncores = 1, validation = {rlit(val)}); r$relevant_variables <- as.matrix(r$relevant_variables); r"
            )

            def fn(X=X, h=h, tau=tau, dr=dr, val=val, nw=nw):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import pandas as pd

                    r = nns.nns_var(
                        pd.DataFrame(X, columns=["A", "B", "C"]),
                        h,
                        tau=tau,
                        dim_red_method=dr,
                        naive_weights=nw,
                        status=False,
                        ncores=1,
                        validation=val,
                    )
                names = list(r["names"])
                out = {}
                for key in (
                    "interpolated_and_extrapolated",
                    "univariate",
                    "multivariate",
                    "ensemble",
                ):
                    a = np.asarray(r[key], dtype=float)
                    out[key] = {nm: a[:, j] for j, nm in enumerate(names)}
                out["relevant_variables"] = np.asarray(r["relevant_variables"], dtype=object)
                v = dict(r["validation"])
                out["validation"] = v
                return out

            cases.append(
                {"id": cid, "expr": expr, "data": {"Xm": np.where(np.isnan(X), None, X).tolist()}}
            )
            pycalls[cid] = fn
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
        rv = r["value"]
        msgs += tree_close(
            py,
            rv,
            rtol=float(__import__("os").environ.get("RTOL", "0")),
            atol=float(__import__("os").environ.get("ATOL", "0")),
            skip=("relevant_variables",),
        )
        # relevant variables: compare sets per target column
        names = rv["names"]
        relv = rv["values"][names.index("relevant_variables")]["v"]
        a = np.asarray(relv).astype(str)
        relevant = np.asarray(py["relevant_variables"], dtype=object)
        missing = np.vectorize(lambda v: v is None, otypes=[bool])(relevant)
        b = np.where(missing, "NA", relevant).astype(str)
        if a.shape != b.shape or not np.all(a == b):
            msgs.append(f".relevant_variables differ r={a.tolist()} py={b.tolist()}")
    problems[cid] = msgs
print(fmt_res("VAR", problems, len(cases)))
print("R secs", round(sum(R[c["id"]]["secs"] for c in cases), 1))
