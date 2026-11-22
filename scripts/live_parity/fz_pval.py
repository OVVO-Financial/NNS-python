"""Differential sweep: R14 permutation p-values (NNS.seas, NNS.dep, NNS.caus, NNS.caus.matrix)."""

import sys
import warnings

import numpy as np
from harness import fmt_res, run_r, tree_close

import nns
from nns.causation import causal_matrix

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 301
only = sys.argv[2] if len(sys.argv) > 2 else "all"
rng = np.random.default_rng(seed)
cases, pycalls = [], {}


def add(cid, expr, data, fn):
    if only != "all" and not cid.startswith(only):
        return
    cases.append({"id": cid, "expr": expr, "data": data})
    pycalls[cid] = fn


def series(kind, n):
    t = np.arange(n, dtype=float)
    if kind == "seas":
        return 10 + 2 * np.sin(2 * np.pi * t / 12) + rng.normal(size=n) * 0.6
    if kind == "trend":
        return 5 + 0.2 * t + np.sin(2 * np.pi * t / 7) + rng.normal(size=n) * 0.5
    if kind == "noise":
        return rng.normal(size=n) + 3
    if kind == "ties":
        return np.round(4 + np.sin(2 * np.pi * t / 6) * 2 + rng.normal(size=n) * 0.4)
    raise ValueError(kind)


for kind in ("seas", "trend", "noise", "ties"):
    for n in (40, 96):
        x = series(kind, n)
        for mp, mod, monly, nperm in (
            (None, None, True, 25),
            (8, None, True, 15),
            (None, 6, False, 20),
            (None, [5, 7], True, 10),
        ):
            s = int(rng.integers(1, 9999))
            rmod = "NULL" if mod is None else ("c(" + ",".join(map(str, np.atleast_1d(mod))) + ")")
            rmp = "NULL" if mp is None else str(mp)
            cid = f"seas|{kind}|{n}|mp={mp}|mod={mod}|{monly}|{nperm}|s{s}"
            add(
                cid,
                f"set.seed({s}); r <- NNS.seas(x, modulo = {rmod}, mod.only = {'TRUE' if monly else 'FALSE'}, plot = FALSE, max.period = {rmp}, p.value = TRUE, nperm = {nperm}); "
                "list(all = as.list(r$all.periods), periods = r$periods, best = r$best.period)",
                {"x": x.tolist()},
                lambda x=x, mp=mp, mod=mod, monly=monly, nperm=nperm, s=s: (
                    lambda r: {
                        "all": dict(r["all.periods"]),
                        "periods": r["periods"],
                        "best": r["best.period"],
                    }
                )(
                    nns.nns_seas(
                        x,
                        modulo=mod,
                        mod_only=monly,
                        max_period=mp,
                        p_value=True,
                        nperm=nperm,
                        random_seed=s,
                    )
                ),
            )

for kind in ("lin", "quad", "indep", "ties"):
    for n in (30, 80):
        x = rng.normal(size=n)
        if kind == "lin":
            y = 0.6 * x + rng.normal(size=n)
        elif kind == "quad":
            y = x * x + rng.normal(size=n) * 0.3
        elif kind == "indep":
            y = rng.normal(size=n)
        else:
            x = np.round(x * 2)
            y = np.round(x + rng.normal(size=n))
        for asym in (False, True):
            s = int(rng.integers(1, 9999))
            cid = f"dep|{kind}|{n}|asym={asym}|s{s}"
            add(
                cid,
                f"set.seed({s}); r <- NNS.dep(x, y, asym = {'TRUE' if asym else 'FALSE'}, p.value = TRUE); lapply(r, unname)",
                {"x": x.tolist(), "y": y.tolist()},
                lambda x=x, y=y, asym=asym, s=s: nns.nns_dep(
                    x, y, asym=asym, p_value=True, random_seed=s
                ),
            )

for kind in ("lead", "indep", "seas"):
    for n in (36, 70):
        if kind == "lead":
            x = np.cumsum(rng.normal(size=n))
            y = np.r_[0, 0, x[:-2]] + rng.normal(size=n) * 0.5
        elif kind == "indep":
            x = rng.normal(size=n)
            y = rng.normal(size=n)
        else:
            x = series("seas", n)
            y = np.r_[x[3:], x[:3]] + rng.normal(size=n) * 0.3
        for tau, rtau in ((0, "0"), (2, "2"), ("cs", '"cs"'), ("ts", '"ts"')):
            for permute in ("auto", "y", "x", "both", "shift"):
                if tau == "ts" and permute not in ("auto", "shift"):
                    continue
                s = int(rng.integers(1, 9999))
                nperm = 12 if permute != "shift" else 15
                cid = f"caus|{kind}|{n}|tau={tau}|{permute}|s{s}"
                add(
                    cid,
                    f"r <- NNS.caus(x, y, tau = {rtau}, p.value = TRUE, nperm = {nperm}, permute = '{permute}', seed = {s}, conf.int = 0.9); "
                    "list(causation = r$causation, p.value = r$p.value)",
                    {"x": x.tolist(), "y": y.tolist()},
                    lambda x=x, y=y, tau=tau, permute=permute, s=s, nperm=nperm: nns.nns_causation(
                        x,
                        y,
                        tau=tau,
                        p_value=True,
                        nperm=nperm,
                        permute=permute,
                        seed=s,
                        conf_int=0.9,
                    ),
                )

for kind in ("mixed", "seas"):
    for n, k in ((40, 3), (60, 4)):
        if kind == "mixed":
            base = rng.normal(size=n)
            M = np.column_stack(
                [base] + [base * (j + 1) * 0.3 + rng.normal(size=n) for j in range(k - 1)]
            )
        else:
            cols = [series("seas", n)]
            for j in range(k - 1):
                cols.append(np.r_[cols[0][j + 1 :], cols[0][: j + 1]] + rng.normal(size=n) * 0.3)
            M = np.column_stack(cols)
        data = {"v": M.ravel(order="F").tolist()}
        for tau, rtau in ((0, "0"), (1, "1"), ("ts", '"ts"')):
            for permute in ("auto", "both", "x", "y", "shift"):
                if tau == "ts" and permute not in ("auto",):
                    continue
                s = int(rng.integers(1, 9999))
                cid = f"cmat|{kind}|{n}|{k}|tau={tau}|{permute}|s{s}"
                add(
                    cid,
                    f"r <- NNS.caus(matrix(v, ncol = {k}), tau = {rtau}, p.value = TRUE, nperm = 8, permute = '{permute}', seed = {s}); lapply(r, unname)",
                    data,
                    lambda M=M, tau=tau, permute=permute, s=s: nns.nns_causation(
                        M, tau=tau, p_value=True, nperm=8, permute=permute, seed=s
                    ),
                )
        cid = f"cmat|{kind}|{n}|{k}|plain"
        add(
            cid,
            f"unname(NNS.caus(matrix(v, ncol = {k}), tau = 0))",
            data,
            lambda M=M: causal_matrix(M, tau=0),
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
        msgs += tree_close(py, r["value"], rtol=0, atol=0)
    problems[cid] = msgs
print(fmt_res("PVAL", problems, len(cases)))
