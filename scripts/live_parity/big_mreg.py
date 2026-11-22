"""Large multivariate NNS.reg predictions vs the new R: the pruned neighbour search."""

import sys

import numpy as np
from harness import run_r

import nns

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
cases, py = [], {}
for p in (2, 3, 5):
    for kind in ("cont", "ties"):
        n = 1500 if p < 5 else 900
        X = rng.normal(size=(n, p))
        if kind == "ties":
            X = np.round(X * 2.0)
        y = np.sin(X[:, 0]) + X[:, 1] ** 2 + rng.normal(size=n) * 0.3
        pe = rng.normal(size=(400, p))
        if kind == "ties":
            pe = np.round(pe * 2.0)
        for dist in ("NNS", "L2", "L1", "FACTOR"):
            for k, rk in ((None, "NULL"), (1, "1"), (3, "3"), ("all", "'all'")):
                if dist == "FACTOR" and kind == "cont":
                    continue
                cid = f"{p}|{kind}|{dist}|{rk}"
                expr = (
                    f"X <- matrix(xv, ncol = {p}); P <- matrix(pv, ncol = {p}); "
                    f"r <- NNS.reg(X, y, point.est = P, plot = FALSE, ncores = 1, dist = '{dist}', n.best = {rk}); "
                    "c(r$Point.est, r$Fitted.xy$y.hat)"
                )
                cases.append(
                    {
                        "id": cid,
                        "expr": expr,
                        "data": {
                            "xv": X.ravel(order="F").tolist(),
                            "y": y.tolist(),
                            "pv": pe.ravel(order="F").tolist(),
                        },
                    }
                )
                py[cid] = (X, y, pe, dist, k)
res = run_r(cases)
bad = 0
for c in cases:
    cid = c["id"]
    r = res[cid]
    X, y, pe, dist, k = py[cid]
    if not r["ok"]:
        print(cid, "R error:", r["error"][:120])
        bad += 1
        continue
    out = nns.nns_reg(X, y, point_est=pe, dist=dist, n_best=k)
    got = np.concatenate(
        [
            np.asarray(out["Point.est"], float).ravel(),
            np.asarray(out["Fitted.xy"]["y.hat"], float).ravel(),
        ]
    )
    want = np.asarray(r["value"]["v"], float).ravel()
    same = got.shape == want.shape and np.array_equal(got, want)
    if not same:
        bad += 1
        diff = np.flatnonzero(got != want) if got.shape == want.shape else []
        print(cid, "MISMATCH", got.shape, want.shape, len(diff), "differ")
print(f"== big mreg: {len(cases) - bad}/{len(cases)} cases match")
