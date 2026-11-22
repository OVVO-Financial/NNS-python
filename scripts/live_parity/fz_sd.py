"""Differential sweep: stochastic dominance family vs R 14."""

import sys
import warnings

import numpy as np
from harness import fmt_res, rget, rnames, run_r, rvec

import nns

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 71
rng = np.random.default_rng(seed)
cases, pycalls, kinds = [], {}, {}

NAMES = {
    "FSD": ("X FSD Y", "Y FSD X", "NO FSD EXISTS"),
    "OMEGA": ("X OMEGA Y", "Y OMEGA X", "NO OMEGA DOMINANCE EXISTS"),
    "SSD": ("X SSD Y", "Y SSD X", "NO SSD EXISTS"),
    "TSD": ("X TSD Y", "Y TSD X", "NO TSD EXISTS"),
}


def label(code, family):
    a, b, c = NAMES[family]
    return {1: a, -1: b, 0: c}[code]


def pair(kind, n1, n2):
    if kind == "shift":
        return rng.normal(size=n1) + 0.3, rng.normal(size=n2)
    if kind == "scale":
        return rng.normal(size=n1) * 0.5 + 0.1, rng.normal(size=n2) * 1.5
    if kind == "ties":
        return np.round(rng.normal(size=n1) * 2) / 2 + 0.5, np.round(rng.normal(size=n2) * 2) / 2
    if kind == "offset":
        return 1e6 + rng.normal(size=n1) + 0.2, 1e6 + rng.normal(size=n2)
    if kind == "bigscale":
        return 1e8 * (rng.normal(size=n1) + 0.5), 1e8 * rng.normal(size=n2)
    if kind == "dom":  # x = y + positive increments (strict FSD by construction)
        y = rng.normal(size=n1)
        return y + rng.uniform(0, 0.5, n1), y
    if kind == "spread":  # mean-preserving spread
        y = rng.normal(size=n1)
        return y, y + rng.normal(size=n1) * 0.5
    if kind == "eps":  # y nudged by one ulp
        y = rng.normal(size=n1)
        return np.nextafter(y, np.inf), y
    if kind == "same":
        y = rng.normal(size=n1)
        return y.copy(), y.copy()
    if kind == "const":
        return np.full(n1, 1.5), np.full(n2, 1.0)
    if kind == "constsame":
        return np.full(n1, 0.1), np.full(n2, 0.1)
    if kind == "perm":
        y = rng.normal(size=n1)
        return rng.permutation(y), y
    if kind == "tiny":
        return rng.normal(size=n1) * 1e-12 + 1e-13, rng.normal(size=n2) * 1e-12
    if kind == "exp":
        return rng.exponential(size=n1) * 1.2, rng.exponential(size=n2)
    raise ValueError(kind)


KINDS = [
    "shift",
    "scale",
    "ties",
    "offset",
    "bigscale",
    "dom",
    "spread",
    "eps",
    "same",
    "const",
    "constsame",
    "perm",
    "tiny",
    "exp",
]
for kind in KINDS:
    for n1, n2 in ((8, 8), (40, 40), (30, 47), (200, 200)):
        if kind in ("dom", "spread", "eps", "same", "perm") and n1 != n2:
            continue
        x, y = pair(kind, n1, n2)
        data = {"x": x.tolist(), "y": y.tolist()}
        base = f"{kind}|{n1}|{n2}"
        for typ, fam in (("discrete", "FSD"), ("continuous", "OMEGA")):
            for a, b, tag in (("x", "y", "xy"), ("y", "x", "yx")):
                cid = f"FSD.{typ}|{base}|{tag}"
                cases.append(
                    {
                        "id": cid,
                        "expr": f"NNS.FSD({a}, {b}, type = '{typ}', plot = FALSE)",
                        "data": data,
                    }
                )
                A, B = (x, y) if tag == "xy" else (y, x)
                pycalls[cid] = lambda A=A, B=B, typ=typ, fam=fam: label(
                    nns.fsd(A, B, type=typ), fam
                )
                cid = f"FSD.uni.{typ}|{base}|{tag}"
                cases.append(
                    {"id": cid, "expr": f"NNS.FSD.uni({a}, {b}, type = '{typ}')", "data": data}
                )
                pycalls[cid] = lambda A=A, B=B, typ=typ: float(nns.fsd_uni(A, B, type=typ))
        for fn, fam in (("SSD", "SSD"), ("TSD", "TSD")):
            for a, b, tag in (("x", "y", "xy"), ("y", "x", "yx")):
                A, B = (x, y) if tag == "xy" else (y, x)
                cid = f"{fn}|{base}|{tag}"
                cases.append({"id": cid, "expr": f"NNS.{fn}({a}, {b}, plot = FALSE)", "data": data})
                f = getattr(nns, fn.lower())
                pycalls[cid] = lambda A=A, B=B, f=f, fam=fam: label(f(A, B), fam)
                cid = f"{fn}.uni|{base}|{tag}"
                cases.append({"id": cid, "expr": f"NNS.{fn}.uni({a}, {b})", "data": data})
                fu = getattr(nns, fn.lower() + "_uni")
                pycalls[cid] = lambda A=A, B=B, fu=fu: float(fu(A, B))


def matrix(kind, n, k):
    if kind == "normal":
        return rng.normal(size=(n, k)) + rng.uniform(-0.5, 0.5, k)
    if kind == "ties":
        return np.round(rng.normal(size=(n, k)) * 2) / 2 + np.round(rng.uniform(-1, 1, k), 1)
    if kind == "nested":  # columns built to dominate one another
        base = rng.normal(size=n)
        cols = [base + i * rng.uniform(0, 0.3, n) for i in range(k)]
        return np.column_stack(cols)
    if kind == "offset":
        return 1e6 + rng.normal(size=(n, k)) * rng.uniform(0.5, 2, k) + rng.uniform(-0.3, 0.3, k)
    if kind == "dup":  # duplicated columns and shuffled copies
        m = rng.normal(size=(n, max(2, k // 2)))
        cols = [m[:, i % m.shape[1]] for i in range(k)]
        cols[-1] = rng.permutation(cols[-1])
        return np.column_stack(cols)
    if kind == "returns":
        return rng.standard_t(4, size=(n, k)) * 0.02 + rng.uniform(-0.002, 0.004, k)
    raise ValueError(kind)


for mkind in ("normal", "ties", "nested", "offset", "dup", "returns"):
    for n, k in ((20, 4), (60, 8), (120, 12)):
        M = matrix(mkind, n, k)
        data = {"v": M.ravel(order="F").tolist()}
        mexpr = f"M <- matrix(v, ncol = {k}); "
        for degree, typ in ((1, "discrete"), (1, "continuous"), (2, "discrete"), (3, "discrete")):
            cid = f"eff|{mkind}|{n}|{k}|{degree}|{typ}"
            cases.append(
                {
                    "id": cid,
                    "expr": mexpr
                    + f"NNS.SD.efficient.set(M, degree = {degree}, type = '{typ}', status = FALSE)",
                    "data": data,
                }
            )
            pycalls[cid] = lambda M=M, degree=degree, typ=typ: [
                f"X_{i + 1}" for i in nns.sd_efficient_set(M, degree, type=typ)
            ]
            for mc in (0, 1, 2, 3):
                cid = f"clu|{mkind}|{n}|{k}|{degree}|{typ}|{mc}"
                cases.append(
                    {
                        "id": cid,
                        "expr": mexpr
                        + f"NNS.SD.cluster(M, degree = {degree}, type = '{typ}', min_cluster = {mc})$Clusters",
                        "data": data,
                    }
                )
                pycalls[cid] = lambda M=M, degree=degree, typ=typ, mc=mc: nns.nns_sd_cluster(
                    M, degree=degree, type=typ, min_cluster=mc
                )["Clusters"]
                kinds[cid] = "clusters"
        cid = f"dend|{mkind}|{n}|{k}"
        cases.append(
            {
                "id": cid,
                "expr": mexpr
                + "pdf(NULL); h <- NNS.SD.cluster(M, degree = 1, dendrogram = TRUE)$Dendrogram; list(merge = h$merge, height = h$height, order = h$order, labels = h$labels)",
                "data": data,
            }
        )

        def dend(M=M):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import matplotlib

                matplotlib.use("Agg")
                d = nns.nns_sd_cluster(M, degree=1, dendrogram=True)["Dendrogram"]
            return d

        pycalls[cid] = dend
        kinds[cid] = "dend"

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
        kind = kinds.get(cid)
        if kind == "clusters":
            rn = rnames(rv)
            rd = (
                {
                    n: [str(s) for s in rvec(v).ravel()] if v is not None else []
                    for n, v in zip(rn, rv["values"], strict=True)
                }
                if rn
                else {}
            )
            if rd != {k: list(v) for k, v in py.items()}:
                msgs.append(f"clusters py={py} r={rd}")
        elif kind == "dend":
            for key in ("merge", "height", "order", "labels"):
                a = np.asarray(py[key])
                b = rvec(rget(rv, key))
                if key == "labels":
                    ok = a.astype(str).tolist() == b.astype(str).tolist()
                else:
                    ok = a.shape == b.shape and np.allclose(
                        a.astype(float), b.astype(float), rtol=0, atol=0
                    )
                if not ok:
                    msgs.append(f"{key}: py={a.tolist()[:12]} r={b.tolist()[:12]}")
        else:
            b = rvec(rv).ravel()
            if isinstance(py, str):
                if b.tolist() != [py]:
                    msgs.append(f"py={py} r={b.tolist()}")
            elif isinstance(py, list):
                if [str(s) for s in b.tolist()] != py:
                    msgs.append(f"py={py} r={b.tolist()}")
            else:
                if b.size != 1 or float(b[0]) != float(py):
                    msgs.append(f"py={py} r={b.tolist()}")
    problems[cid] = msgs
print(fmt_res("SD", problems, len(cases)))
