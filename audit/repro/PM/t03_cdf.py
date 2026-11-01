"""NNS.CDF univariate/multivariate A/B and first-principles checks."""
import sys, warnings, time
sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, pandas as pd
import nns
from rrun import rcall, arr

px = pd.read_csv("/home/user/audit/data/prices_daily.csv", index_col=0)
ret = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)
spy = px["SPY"].dropna().to_numpy()
btc = px["BTC-USD"].dropna().to_numpy()
nvda = ret["NVDA"].to_numpy()
rng = np.random.default_rng(7)

print("== univariate CDF grid accuracy (python cdf._finite_sorted_grid_lpm_ratio vs direct LPM.ratio)")
from nns.cdf import _finite_sorted_grid_lpm_ratio
for label, x in [("SPY px", spy), ("BTC px", btc), ("NVDA", nvda), ("shifted N(1e6,1)", 1e6 + rng.normal(size=2000))]:
    xs = np.sort(x)
    for d in (1, 2, 3):
        g = _finite_sorted_grid_lpm_ratio(d, xs)
        ex = np.concatenate([np.atleast_1d(nns.lpm_ratio(d, xs[i:i+32], x)) for i in range(0, len(xs), 32)])
        bad = np.sum((g < 0) | (g > 1) | ~np.isfinite(g))
        print(f"{label:16s} d={d}: max |grid - exact| = {np.nanmax(np.abs(g-ex)):.3g}; out-of-[0,1]/nan: {bad}; non-monotone: {np.sum(np.diff(g) < -1e-12)}")

print("== nns_cdf vs R NNS.CDF, univariate, all types")
x = rng.normal(size=300)
xs_shift = 1e6 + rng.normal(size=300)
for label, v in [("norm", x), ("shift1e6", xs_shift), ("NVDA_unique", np.unique(nvda[:1000]))]:
    res = rcall("""lapply(types, function(tp) lapply(c(0,1,2), function(d) {
        o <- NNS.CDF(v, degree=d, target=v[5], type=tp, plot=FALSE); list(f=o$Function[[2]], p=o$target.value)}))""",
        v=v, types=["CDF", "survival", "hazard", "cumulative hazard"])
    for tp, rr in zip(["CDF", "survival", "hazard", "cumulative hazard"], res):
        for d, r in zip([0, 1, 2], rr):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o = nns.nns_cdf(v, degree=d, target=v[4], type=tp)
            f = list(o["Function"].values())[1]
            rf, rp = arr(r["f"]), arr(r["p"])
            df = np.nanmax(np.abs(f - rf)); dp = np.nanmax(np.abs(np.atleast_1d(o["target.value"]) - np.atleast_1d(rp)))
            flag = "" if (df < 1e-8 and dp < 1e-8) else "  <-- MISMATCH"
            print(f"{label:8s} {tp:17s} d={d}: max|dF|={df:.3g} |dP|={dp:.3g}{flag}")

print("== CDF properties: degree 0 equals ECDF; survival=1-CDF; cumhaz=-log(S)")
o = nns.nns_cdf(x, 0)
print("ecdf ok:", np.allclose(o["Function"]["CDF"], np.searchsorted(np.sort(x), np.sort(x), side="right") / len(x)))
s = nns.nns_cdf(x, 0, type="survival")["Function"]["S(x)"]
H = nns.nns_cdf(x, 0, type="cumulative hazard")["Function"]["H(x)"]
print("H == -log(S):", np.allclose(H, -np.log(np.maximum(s, 1e-10))))

print("== multivariate CDF A/B")
A = np.column_stack([rng.normal(size=200), rng.normal(size=200)])
A[:, 1] += 0.5 * A[:, 0]
for tp in ["CDF", "survival", "cumulative hazard", "hazard"]:
    for d in [0, 1]:
        r = rcall("o <- NNS.CDF(A, degree=d, target=c(0,0), type=tp, plot=FALSE); list(f=o$Function$CDF, p=o$target.value)", A=A, d=d, tp=tp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = nns.nns_cdf(A, degree=d, target=np.array([0.0, 0.0]), type=tp)
        df = np.nanmax(np.abs(o["Function"]["CDF"] - arr(r["f"]))); dp = np.nanmax(np.abs(np.atleast_1d(o["target.value"]) - arr(r["p"])))
        print(f"mv {tp:17s} d={d}: max|dF|={df:.3g} |dP|={dp:.3g}{'  <-- MISMATCH' if max(df, dp) > 1e-8 else ''}")

print("== multivariate survival degree 0 first principles: S(t) = P(X>t1, Y>t2)")
o = nns.nns_cdf(A, 0, type="survival")
true_s = np.array([np.mean((A[:, 0] > a) & (A[:, 1] > b)) for a, b in A])
print("max |nns survival - P(X>x,Y>y)| =", np.max(np.abs(o["Function"]["CDF"] - true_s)))
o2 = nns.nns_cdf(A, 0)
true_c = np.array([np.mean((A[:, 0] <= a) & (A[:, 1] <= b)) for a, b in A])
print("max |nns joint CDF - P(X<=x,Y<=y)| =", np.max(np.abs(o2["Function"]["CDF"] - true_c)))
o3 = nns.nns_cdf(A, 1)
print("degree-1 multivariate CDF range", o3["Function"]["CDF"].min(), o3["Function"]["CDF"].max())

print("== multivariate CDF memory/time: python builds n x n x d tensors")
for n in (1000, 3000):
    B = rng.normal(size=(n, 2))
    t0 = time.perf_counter(); nns.nns_cdf(B, 1); tp_ = time.perf_counter() - t0
    r = rcall("t0 <- proc.time()[[3]]; o <- NNS.CDF(B, 1, plot=FALSE); proc.time()[[3]] - t0", B=B)
    print(f"n={n} degree=1 python {tp_:.2f}s  R {r}s")
