"""Co-/divergent partial moments and PM.matrix: identities + A/B vs R."""
import sys, warnings
sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, pandas as pd
import nns
from rrun import rcall, arr

ret = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)
X = ret.to_numpy()
x, y = ret["SPY"].to_numpy(), ret["TLT"].to_numpy()
mx, my = x.mean(), y.mean()

print("== CLPM+CUPM-DLPM-DUPM == population cov (degree 1, mean targets)")
s = nns.co_lpm(1, x, y, mx, my) + nns.co_upm(1, x, y, mx, my) - nns.d_lpm(1, 1, x, y, mx, my) - nns.d_upm(1, 1, x, y, mx, my)
print("identity:", s, "np.cov(ddof=0):", np.cov(x, y, ddof=0)[0, 1], "diff", s - np.cov(x, y, ddof=0)[0, 1])
print("== degree 0 quadrant probabilities sum to 1")
s0 = nns.co_lpm(0, x, y, mx, my) + nns.co_upm(0, x, y, mx, my) + nns.d_lpm(0, 0, x, y, mx, my) + nns.d_upm(0, 0, x, y, mx, my)
print("sum:", s0)
# ties exactly at target
xt = np.array([0.0, 1, 2, 1, 1]); yt = np.array([1.0, 1, 0, 2, 1])
s0t = nns.co_lpm(0, xt, yt, 1, 1) + nns.co_upm(0, xt, yt, 1, 1) + nns.d_lpm(0, 0, xt, yt, 1, 1) + nns.d_upm(0, 0, xt, yt, 1, 1)
print("ties sum:", s0t)

print("== d_lpm / d_upm quadrant semantics: D.LPM = x above, y below")
dl = nns.d_lpm(1, 1, x, y, mx, my)
print("d_lpm:", dl, "naive mean(max(x-tx,0)*max(ty-y,0)):", np.mean(np.maximum(x - mx, 0) * np.maximum(my - y, 0)))
du = nns.d_upm(1, 1, x, y, mx, my)
print("d_upm:", du, "naive mean(max(tx-x,0)*max(y-ty,0)):", np.mean(np.maximum(mx - x, 0) * np.maximum(y - my, 0)))
print("== mixed degrees (D.LPM(2,1)) python vs naive")
print("d_lpm(degree_lpm=2, degree_upm=1):", nns.d_lpm(2, 1, x, y, mx, my), "naive (x upper^1)*(y lower^2):",
      np.mean(np.maximum(x - mx, 0) ** 1 * np.maximum(my - y, 0) ** 2))
print("d_upm(degree_lpm=2, degree_upm=1):", nns.d_upm(2, 1, x, y, mx, my), "naive (x lower^2)*(y upper^1):",
      np.mean(np.maximum(mx - x, 0) ** 2 * np.maximum(y - my, 0) ** 1))
print("co_lpm(2, degree_y=1):", nns.co_lpm(2, x, y, mx, my, degree_y=1), "naive:", np.mean(np.maximum(mx - x, 0) ** 2 * np.maximum(my - y, 0)))

print("== A/B vs R co-moments (scalar + vector targets, mixed/fractional degrees)")
tx = np.quantile(x, np.linspace(0.05, 0.95, 50)); ty = np.quantile(y, np.linspace(0.05, 0.95, 25))
r = rcall("""lapply(list(c(0,0),c(1,1),c(2,1),c(0.5,1.5),c(0,2)), function(d) list(
  cl=Co.LPM(d[1], x, y, tx, ty, d[2]), cu=Co.UPM(d[1], x, y, tx, ty, d[2]),
  dl=D.LPM(d[1], d[2], x, y, tx, ty), du=D.UPM(d[1], d[2], x, y, tx, ty),
  cl1=Co.LPM(d[1], x, y, mean(x), mean(y)), dl1=D.LPM(d[1], d[2], x, y, mean(x), mean(y))))""", x=x, y=y, tx=tx, ty=ty)
for d, rr in zip([(0, 0), (1, 1), (2, 1), (0.5, 1.5), (0, 2)], r):
    py = dict(cl=nns.co_lpm(d[0], x, y, tx, ty, degree_y=d[1]), cu=nns.co_upm(d[0], x, y, tx, ty, degree_y=d[1]),
              dl=nns.d_lpm(d[0], d[1], x, y, tx, ty), du=nns.d_upm(d[0], d[1], x, y, tx, ty),
              cl1=nns.co_lpm(d[0], x, y, mx, my), dl1=nns.d_lpm(d[0], d[1], x, y, mx, my))
    for k in py:
        dd = np.max(np.abs(np.atleast_1d(py[k]) - np.atleast_1d(arr(rr[k]))))
        if dd > 1e-12:
            print(f"MISMATCH d={d} {k}: {dd:.3g}")
print("co-moment A/B done; target recycling 50 vs 25 lengths OK if no mismatch above")

print("== PM.matrix")
for pop_adj in [True, False]:
    pm = nns.pm_matrix(1, 1, None, X, pop_adj)
    ref = np.cov(X.T, ddof=1 if pop_adj else 0)
    print(f"pop_adj={pop_adj}: max|cov.matrix - np.cov(ddof={1 if pop_adj else 0})| = {np.max(np.abs(pm['cov.matrix'] - ref)):.3g}")
pm = nns.pm_matrix(1, 1, None, X, True)
print("DUPM == DLPM.T:", np.allclose(pm["dupm"], pm["dlpm"].T))
print("pairwise consistency: pm['clpm'][i,j] == co_lpm(x_i,x_j):", np.isclose(pm["clpm"][7, 8] * (len(x) - 1) / len(x), nns.co_lpm(1, X[:, 7], X[:, 8], X[:, 7].mean(), X[:, 8].mean())))
print("pm['dlpm'][i,j] corresponds to d_lpm(x_i, x_j)? ", np.isclose(pm["dlpm"][7, 8] * (len(x) - 1) / len(x), nns.d_lpm(1, 1, X[:, 7], X[:, 8], X[:, 7].mean(), X[:, 8].mean())),
      " or d_lpm(x_j, x_i)?", np.isclose(pm["dlpm"][7, 8] * (len(x) - 1) / len(x), nns.d_lpm(1, 1, X[:, 8], X[:, 7], X[:, 8].mean(), X[:, 7].mean())))
for degs, norm, tgt in [((1, 1), False, None), ((0, 0), False, None), ((2, 2), True, None), ((0, 1), False, np.zeros(X.shape[1])), ((1.5, 0.5), True, np.median(X, axis=0)), ((0, 0), True, None)]:
    for pop_adj in [True, False]:
        rr = rcall("o <- PM.matrix(a, b, tg, X, pa, nm); lapply(o, function(m) unname(m))", a=degs[0], b=degs[1], tg=tgt, X=X, pa=pop_adj, nm=norm)
        pm = nns.pm_matrix(degs[0], degs[1], tgt, X, pop_adj, norm=norm)
        dd = max(np.max(np.abs(pm[k] - arr(rr[k]))) for k in ["cupm", "dupm", "dlpm", "clpm", "cov.matrix"])
        print(f"PM.matrix degs={degs} norm={norm} pop_adj={pop_adj} target={'mean' if tgt is None else 'given'}: max diff vs R {dd:.3g}")

print("== PM.matrix pop_adj gating: degree (0,1) -> no adjustment even though pop_adj=TRUE")
a = nns.pm_matrix(0, 1, None, X[:, :2], True)["cov.matrix"]; b = nns.pm_matrix(0, 1, None, X[:, :2], False)["cov.matrix"]
print("equal:", np.allclose(a, b))
print("== norm=TRUE cov.matrix diag (should be 1 for self-dependence?)", np.diag(nns.pm_matrix(1, 1, None, X, True, norm=True)["cov.matrix"])[:3])
print("== python fallback (no native) vs native")
import importlib; pmm = sys.modules["nns.pm_matrix"]
orig = pmm.native_fn
pmm.native_fn = lambda name: None
fb = nns.pm_matrix(1, 1, None, X, True, norm=True)
fb0 = nns.pm_matrix(0, 0, None, X, True)
pmm.native_fn = orig
nt = nns.pm_matrix(1, 1, None, X, True, norm=True); nt0 = nns.pm_matrix(0, 0, None, X, True)
print("fallback==native:", all(np.allclose(fb[k], nt[k]) for k in fb), all(np.allclose(fb0[k], nt0[k]) for k in fb0))
