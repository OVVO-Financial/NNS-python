"""NNS.dep p-values where shuffles reproduce the observed table (R 14.2)."""

import warnings

import numpy as np
from harness import fmt_res, run_r, tree_close

import nns

rng = np.random.default_rng(5)
cases, pyc = [], {}


def add(cid, x, y, asym, seed):
    a = "TRUE" if asym else "FALSE"
    cases.append(
        {
            "id": cid,
            "expr": f"set.seed({seed}); r <- NNS.dep(x, y, p.value = TRUE, asym = {a}); unlist(r)",
            "data": {"x": x.tolist(), "y": y.tolist()},
        }
    )
    pyc[cid] = (x, y, asym, seed)


for n in (4, 5, 6, 8, 12):
    for asym in (False, True):
        for seed in (1, 7, 42):
            bx = rng.integers(0, 2, n).astype(float)
            by = rng.integers(0, 2, n).astype(float)
            if bx.min() == bx.max():
                bx[0] = 1 - bx[0]
            if by.min() == by.max():
                by[0] = 1 - by[0]
            add(f"bin|{n}|{asym}|{seed}", bx, by, asym, seed)
            bal = np.array([0.0, 1.0] * (n // 2) + ([0.0] if n % 2 else []))
            add(f"balanced|{n}|{asym}|{seed}", bal, rng.permutation(bal), asym, seed)
            add(f"consty|{n}|{asym}|{seed}", rng.normal(size=n), np.full(n, 2.0), asym, seed)
            t = np.round(rng.normal(size=n))
            add(f"ties|{n}|{asym}|{seed}", t, rng.permutation(t), asym, seed)
            v = rng.normal(size=n)
            add(f"sameset|{n}|{asym}|{seed}", v, rng.permutation(v), asym, seed)
res = run_r(cases)
problems = {}
for c in cases:
    cid = c["id"]
    r = res[cid]
    x, y, asym, seed = pyc[cid]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py = nns.nns_dep(x, y, asym=asym, p_value=True, random_seed=seed)
            perr = None
    except Exception as exc:
        py, perr = None, f"{type(exc).__name__}: {exc}"
    msgs = []
    if not r["ok"]:
        if perr is None:
            msgs.append(f"R error but py ok: {r['error'][:150]}")
    elif perr is not None:
        msgs.append(f"py error but R ok: {perr[:200]}")
    else:
        flat = np.concatenate([np.atleast_1d(np.asarray(v, float)).ravel() for v in py.values()])
        msgs += tree_close(flat, r["value"], rtol=0, atol=0)
    problems[cid] = msgs
print(fmt_res("dep permutation matches", problems, len(cases)))
