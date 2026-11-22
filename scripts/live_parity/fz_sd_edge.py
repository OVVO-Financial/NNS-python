"""SD edge sweep: infinite values, validation order and messages vs R 14."""

import sys
import warnings

import numpy as np
from harness import fmt_res, rnames, run_r, rvec

import nns

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 81
rng = np.random.default_rng(seed)
cases, pycalls, kinds = [], {}, {}
INF = np.inf


def with_inf(v, where):
    v = v.copy()
    if where == "hi":
        v[rng.integers(v.size)] = INF
    elif where == "lo":
        v[rng.integers(v.size)] = -INF
    elif where == "both":
        i, j = rng.choice(v.size, 2, replace=False)
        v[i], v[j] = INF, -INF
    elif where == "allhi":
        v[:] = INF
    return v


def encode(v):
    return [
        ("Inf" if x > 0 else "-Inf") if np.isinf(x) else ("NA" if np.isnan(x) else float(x))
        for x in v
    ]


R_DECODE = "dec <- function(v) as.numeric(unlist(lapply(v, function(e) if (is.character(e)) switch(e, 'Inf' = Inf, '-Inf' = -Inf, 'NA' = NA_real_) else e))); "

for n1, n2 in ((6, 6), (25, 31)):
    for wx in ("none", "hi", "lo", "both", "allhi"):
        for wy in ("none", "hi", "lo", "both"):
            if wx == "none" and wy == "none":
                continue
            for shift in (0.0, 0.4):
                x = with_inf(rng.normal(size=n1) + shift, wx)
                y = with_inf(rng.normal(size=n2), wy)
                data = {"xr": encode(x), "yr": encode(y)}
                base = f"{n1}|{wx}|{wy}|{shift}"
                for a, b, tag in (("x", "y", "xy"), ("y", "x", "yx")):
                    A, B = (x, y) if tag == "xy" else (y, x)
                    for typ in ("discrete", "continuous"):
                        cid = f"FSD.uni.{typ}|{base}|{tag}"
                        cases.append(
                            {
                                "id": cid,
                                "expr": R_DECODE
                                + f"x <- dec(xr); y <- dec(yr); NNS.FSD.uni({a}, {b}, type = '{typ}')",
                                "data": data,
                            }
                        )
                        pycalls[cid] = lambda A=A, B=B, typ=typ: float(nns.fsd_uni(A, B, type=typ))
                    for fn in ("SSD", "TSD"):
                        cid = f"{fn}.uni|{base}|{tag}"
                        cases.append(
                            {
                                "id": cid,
                                "expr": R_DECODE
                                + f"x <- dec(xr); y <- dec(yr); NNS.{fn}.uni({a}, {b})",
                                "data": data,
                            }
                        )
                        fu = getattr(nns, fn.lower() + "_uni")
                        pycalls[cid] = lambda A=A, B=B, fu=fu: float(fu(A, B))
                    for fn, kw in (
                        ("FSD", "type = 'discrete', "),
                        ("FSD", "type = 'continuous', "),
                        ("SSD", ""),
                        ("TSD", ""),
                    ):
                        cid = f"{fn}.top.{kw[8:-3] or 'na'}|{base}|{tag}"
                        cases.append(
                            {
                                "id": cid,
                                "expr": R_DECODE
                                + f"x <- dec(xr); y <- dec(yr); NNS.{fn}({a}, {b}, {kw}plot = FALSE)",
                                "data": data,
                            }
                        )
                        f = getattr(nns, fn.lower())
                        if fn == "FSD":
                            typ = "continuous" if "continuous" in kw else "discrete"
                            pycalls[cid] = lambda A=A, B=B, f=f, typ=typ: f(A, B, type=typ)
                        else:
                            pycalls[cid] = lambda A=A, B=B, f=f: f(A, B)
                        kinds[cid] = "top"

for n, k in ((10, 3), (30, 6)):
    for where in ("hi", "lo", "both", "col"):
        M = rng.normal(size=(n, k)) + rng.uniform(-0.5, 0.5, k)
        if where == "col":
            M[:, 0] = INF
        else:
            for j in rng.choice(k, 2, replace=False):
                M[:, j] = with_inf(M[:, j], where)
        data = {"vr": encode(M.ravel(order="F"))}
        mexpr = R_DECODE + f"M <- matrix(dec(vr), ncol = {k}); "
        for degree, typ in ((1, "discrete"), (1, "continuous"), (2, "discrete"), (3, "discrete")):
            cid = f"eff|{where}|{n}|{k}|{degree}|{typ}"
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
            for mc in (0, 1, 1.5, -1):
                cid = f"clu|{where}|{n}|{k}|{degree}|{typ}|{mc}"
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

# validation-order cases
V = [
    (
        "nan_x",
        "NNS.FSD(c(1, NA, 3), c(0, 1, 2), plot = FALSE)",
        lambda: nns.fsd(np.array([1, np.nan, 3]), np.array([0.0, 1, 2])),
    ),
    (
        "nan_uni",
        "NNS.SSD.uni(c(1, 2), c(0, NA))",
        lambda: nns.ssd_uni(np.array([1.0, 2]), np.array([0, np.nan])),
    ),
    (
        "empty_x",
        "NNS.FSD(numeric(0), c(1, 2), plot = FALSE)",
        lambda: nns.fsd(np.array([]), np.array([1.0, 2])),
    ),
    (
        "empty_uni",
        "NNS.TSD.uni(c(1, 2), numeric(0))",
        lambda: nns.tsd_uni(np.array([1.0, 2]), np.array([])),
    ),
    (
        "eff_0rows",
        "NNS.SD.efficient.set(matrix(numeric(0), 0, 2), 1, status = FALSE)",
        lambda: nns.sd_efficient_set(np.zeros((0, 2)), 1),
    ),
    (
        "eff_0cols_bad_degree",
        "NNS.SD.efficient.set(matrix(numeric(0), 3, 0), 7, status = FALSE)",
        lambda: nns.sd_efficient_set(np.zeros((3, 0)), 7),
    ),
    (
        "eff_nan",
        "NNS.SD.efficient.set(matrix(c(1, 2, 3, 4, 5, NA), 3), 2, status = FALSE)",
        lambda: nns.sd_efficient_set(np.array([[1, 4], [2, 5], [3, np.nan]]), 2),
    ),
    (
        "eff_nan_bad_degree",
        "NNS.SD.efficient.set(matrix(c(1, 2, 3, 4, 5, NA), 3), 5, status = FALSE)",
        lambda: nns.sd_efficient_set(np.array([[1, 4], [2, 5], [3, np.nan]]), 5),
    ),
    (
        "eff_vector",
        "NNS.SD.efficient.set(c(3, 1, 2), 2, status = FALSE)",
        lambda: [f"X_{i + 1}" for i in nns.sd_efficient_set(np.array([3.0, 1, 2]), 2)],
    ),
    (
        "clu_nan_small",
        "NNS.SD.cluster(matrix(c(1, 2, NA, 4), 2), min_cluster = 2)$Clusters",
        lambda: nns.nns_sd_cluster(np.array([[1, np.nan], [2, 4]]), min_cluster=2)["Clusters"],
    ),
    (
        "clu_bad_degree_small",
        "NNS.SD.cluster(matrix(c(1, 2, 3, 4), 2), degree = 9, min_cluster = 3)$Clusters",
        lambda: nns.nns_sd_cluster(np.array([[1.0, 3], [2, 4]]), degree=9, min_cluster=3)[
            "Clusters"
        ],
    ),
    (
        "clu_nan",
        "NNS.SD.cluster(matrix(c(1, 2, NA, 4), 2))$Clusters",
        lambda: nns.nns_sd_cluster(np.array([[1, np.nan], [2, 4]]))["Clusters"],
    ),
    (
        "clu_0rows",
        "NNS.SD.cluster(matrix(numeric(0), 0, 3))$Clusters",
        lambda: nns.nns_sd_cluster(np.zeros((0, 3)))["Clusters"],
    ),
]
for vid, expr, fn in V:
    cid = f"val|{vid}"
    cases.append({"id": cid, "expr": expr, "data": {}})
    pycalls[cid] = fn
    kinds[cid] = "clusters" if vid.startswith("clu") else "top"

print("cases", len(cases), flush=True)
R = run_r(cases)
LABEL = {
    ("FSD", "discrete"): ("X FSD Y", "Y FSD X", "NO FSD EXISTS"),
    ("FSD", "continuous"): ("X OMEGA Y", "Y OMEGA X", "NO OMEGA DOMINANCE EXISTS"),
    ("SSD", "na"): ("X SSD Y", "Y SSD X", "NO SSD EXISTS"),
    ("TSD", "na"): ("X TSD Y", "Y TSD X", "NO TSD EXISTS"),
}
problems = {}
for c in cases:
    cid = c["id"]
    r = R[cid]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            py = pycalls[cid]()
        perr = None
    except Exception as exc:
        py, perr = None, f"{type(exc).__name__}: {exc}"
    msgs = []
    if not r["ok"]:
        if perr is None:
            msgs.append(f"R error but py ok: R={r['error'][:150]} py={py}")
        elif r["error"].strip() not in perr:
            msgs.append(f"error text R={r['error'].strip()[:120]!r} py={perr[:160]!r}")
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
        elif kind == "top" and isinstance(py, (int, np.integer)):
            fam = cid.split("|")[0].split(".")
            key = (fam[0], fam[2]) if fam[0] == "FSD" else (fam[0], "na")
            want = LABEL[key][{1: 0, -1: 1, 0: 2}[int(py)]]
            b = rvec(rv).ravel().tolist()
            if b != [want]:
                msgs.append(f"py={want} r={b}")
        else:
            b = rvec(rv).ravel()
            if isinstance(py, list):
                if [str(s) for s in b.tolist()] != py:
                    msgs.append(f"py={py} r={b.tolist()}")
            elif b.size != 1 or float(b[0]) != float(py):
                msgs.append(f"py={py} r={b.tolist()}")
    problems[cid] = msgs
print(fmt_res("SD edge", problems, len(cases)))
