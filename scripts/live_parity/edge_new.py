"""Edge cases of the kernels compiled in the R 14.0 update, Python vs live R."""

import sys
import warnings

import numpy as np
from harness import fmt_res, run_r, tree_close

import nns

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 21)
cases, pyc = [], {}


def add(cid, expr, data, fn):
    cases.append({"id": cid, "expr": expr, "data": data})
    pyc[cid] = fn


# LPM.VaR / UPM.VaR: integer degrees 1-4 (compiled), ties, constant, tiny n, shifts
ps = [0.0, 1e-12, 0.001, 0.125, 0.25, 0.5, 0.75, 0.875, 0.999, 1 - 1e-12, 1.0]
datasets = {
    "n1": np.array([3.5]),
    "n2": np.array([1.0, 4.0]),
    "const": np.full(9, -2.0),
    "ties": np.round(rng.normal(size=40)),
    "shift": rng.normal(size=50) * 1e-3 + 1e6,
    "skew": rng.exponential(size=70) ** 2,
    "bimodal": np.r_[rng.normal(-5, 1, 30), rng.normal(5, 1, 30)],
    "int": rng.integers(0, 5, 33).astype(float),
}
for dn, x in datasets.items():
    for deg in (1, 2, 3, 4):
        for fn in ("LPM.VaR", "UPM.VaR"):
            pyf = nns.lpm_var if fn == "LPM.VaR" else nns.upm_var
            add(
                f"{fn}|{dn}|{deg}",
                f"unname({fn}(p, {deg}, x))",
                {"x": x.tolist(), "p": ps},
                lambda pyf=pyf, deg=deg, x=x: pyf(np.array(ps), deg, x),
            )

# NNS.copula: constant margins, boundary targets, perfect (anti)dependence, many columns
base = rng.normal(size=60)
mats = {
    "const_col": np.column_stack([base, np.full(60, 1.0), base * 2 + rng.normal(size=60)]),
    "two_const": np.column_stack([base, np.full(60, 1.0), np.full(60, 3.0)]),
    "perfect": np.column_stack([base, base * 3 + 1]),
    "anti": np.column_stack([base, -base]),
    "discrete": np.round(np.column_stack([base, base + rng.normal(size=60), rng.normal(size=60)])),
    "wide6": np.column_stack([base * (j % 3) * 0.5 + rng.normal(size=60) for j in range(6)]),
}
for mn, X in mats.items():
    for cont in (True, False):
        flag = "TRUE" if cont else "FALSE"
        add(
            f"copula|{mn}|{cont}",
            f"NNS.copula(matrix(v, ncol = {X.shape[1]}), continuous = {flag})",
            {"v": X.ravel(order="F").tolist()},
            lambda X=X, cont=cont: nns.nns_copula(X, continuous=cont),
        )
        tmin = X.min(axis=0)
        add(
            f"copula.tmin|{mn}|{cont}",
            f"NNS.copula(matrix(v, ncol = {X.shape[1]}), target = t, continuous = {flag})",
            {"v": X.ravel(order="F").tolist(), "t": tmin.tolist()},
            lambda X=X, cont=cont, tmin=tmin: nns.nns_copula(X, target=tmin, continuous=cont),
        )

# NNS.dep matrix (shared splits) with and without asym, ties
for kind in ("cont", "ties"):
    M = rng.normal(size=(80, 4))
    M[:, 1] = M[:, 0] ** 2 + rng.normal(size=80) * 0.2
    if kind == "ties":
        M = np.round(M)
    for asym in (False, True):
        a = "TRUE" if asym else "FALSE"
        add(
            f"depmat|{kind}|{asym}",
            f"r <- NNS.dep(matrix(v, ncol = 4), asym = {a}); c(unname(r$Correlation), unname(r$Dependence))",
            {"v": M.ravel(order="F").tolist()},
            lambda M=M, asym=asym: (
                lambda r: np.r_[
                    np.asarray(r["Correlation"]).ravel(order="F"),
                    np.asarray(r["Dependence"]).ravel(order="F"),
                ]
            )(nns.nns_dep(M, asym=asym)),
        )

# Classification predictions (compiled loop): dist, n.best, ties
Xc = np.round(rng.normal(size=(120, 3)) * 2)
yc = (Xc[:, 0] + Xc[:, 1] > 0).astype(float) + (Xc[:, 2] > 1)
P = np.round(rng.normal(size=(25, 3)) * 2)
for dist in ("NNS", "L2", "L1", "FACTOR"):
    for k, rk in ((None, "NULL"), (1, "1"), (4, "4"), ("all", "'all'")):
        add(
            f"class|{dist}|{rk}",
            f"r <- NNS.reg(matrix(xv, ncol = 3), y, type = 'CLASS', point.est = matrix(pv, ncol = 3), dist = '{dist}', n.best = {rk}, plot = FALSE); c(r$Point.est, unname(as.matrix(r$Point.probabilities)))",
            {
                "xv": Xc.ravel(order="F").tolist(),
                "y": yc.tolist(),
                "pv": P.ravel(order="F").tolist(),
            },
            lambda dist=dist, k=k: (
                lambda r: np.r_[
                    np.asarray(r["Point.est"], float).ravel(),
                    np.column_stack(
                        [np.asarray(v, float) for v in r["Point.probabilities"].values()]
                    ).ravel(order="F"),
                ]
            )(nns.nns_reg(Xc, yc, type="CLASS", point_est=P, dist=dist, n_best=k)),
        )

# dy.d_ mixed (shared stencils)
Xd = rng.normal(size=(150, 2))
yd = Xd[:, 0] * Xd[:, 1] + np.sin(Xd[:, 0])
ev = np.array([[0.1, -0.2], [0.5, 0.5], [-0.7, 0.3]])
for wrt in (1, 2):
    add(
        f"dyd_mixed|{wrt}",
        f"r <- dy.d_(matrix(xv, ncol = 2), y, wrt = {wrt}, eval.points = matrix(ev, ncol = 2), mixed = TRUE, messages = FALSE); c(r['First', 1][[1]], r['Second', 1][[1]], r['Mixed', 1][[1]])",
        {"xv": Xd.ravel(order="F").tolist(), "y": yd.tolist(), "ev": ev.ravel(order="F").tolist()},
        lambda wrt=wrt: np.concatenate(
            [
                np.asarray(v, float).ravel(order="F")
                for v in (lambda d: [d["First"], d["Second"], d["Mixed"]])(
                    nns.dy_d(Xd, yd, wrt=wrt, eval_points=ev, mixed=True, messages=False)
                )
            ]
        ),
    )

res = run_r(cases)
problems = {}
for c in cases:
    cid = c["id"]
    r = res[cid]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py = pyc[cid]()
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
        msgs += tree_close(np.asarray(py, dtype=float), r["value"], rtol=0, atol=0)
    problems[cid] = msgs
print(fmt_res("R14-update edges", problems, len(cases)))
