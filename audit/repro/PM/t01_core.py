"""Univariate LPM/UPM/ratio: first principles + A/B vs R."""
import sys, warnings
sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, pandas as pd
import nns
from rrun import rcall, arr

def naive_lpm(d, t, x):
    t = np.atleast_1d(np.asarray(t, float))
    if d == 0:
        return np.array([(x <= tt).mean() for tt in t])
    return np.array([np.mean(np.maximum(tt - x, 0) ** d) for tt in t])

def naive_upm(d, t, x):
    t = np.atleast_1d(np.asarray(t, float))
    if d == 0:
        return np.array([(x > tt).mean() for tt in t])
    return np.array([np.mean(np.maximum(x - tt, 0) ** d) for tt in t])

rng = np.random.default_rng(1)
ret = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)
px = pd.read_csv("/home/user/audit/data/prices_daily.csv", index_col=0)
btc = px["BTC-USD"].dropna().pct_change().dropna().to_numpy()
spy_px = px["SPY"].dropna().to_numpy()
cases = {
    "norm100": rng.normal(size=100),
    "ties": np.array([1, 2, 2, 2, 3, 3, 4.0]),
    "nvda": ret["NVDA"].to_numpy(),
    "btc": btc,
    "spy_px": spy_px,
    "len1": np.array([0.5]),
    "len2": np.array([0.5, -1.0]),
    "const": np.full(10, 3.0),
}
print("== first principles: python lpm/upm vs naive (scalar + >32 vector targets)")
worst = {}
for name, x in cases.items():
    tvec = np.concatenate([np.sort(x), np.linspace(x.min() - 1, x.max() + 1, 50)])
    for d in [0, 0.5, 1, 2, 3, 4, 1.7]:
        for t in [float(np.mean(x)), 0.0]:
            a = nns.lpm(d, t, x); b = naive_lpm(d, t, x)[0]
            if not np.isclose(a, b, rtol=1e-10, atol=1e-14):
                print("SCALAR MISMATCH lpm", name, d, t, a, b)
        a = np.asarray(nns.lpm(d, tvec, x)); b = naive_lpm(d, tvec, x)
        au = np.asarray(nns.upm(d, tvec, x)); bu = naive_upm(d, tvec, x)
        scale = np.maximum(np.abs(b), 1e-300)
        rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), np.max(np.abs(b)) * 1e-300 + 1e-300))
        relu = np.max(np.abs(au - bu) / np.maximum(np.abs(bu), 1e-300))
        absd = max(np.max(np.abs(a - b)), np.max(np.abs(au - bu)))
        neg = int(np.sum(a < 0) + np.sum(au < 0))
        worst[(name, d)] = (rel, relu, absd, neg)
        if rel > 1e-8 or relu > 1e-8 or neg:
            print(f"VECTOR {name:7s} d={d}: max rel err lpm={rel:.3g} upm={relu:.3g} abs={absd:.3g} negatives={neg}")

print("== identities")
for name, x in cases.items():
    m = x.mean()
    i1 = nns.lpm(1, m, x) - nns.upm(1, m, x)
    i2 = nns.upm(2, m, x) + nns.lpm(2, m, x) - np.var(x)
    i3 = nns.lpm(0, m, x) + nns.upm(0, m, x) - 1
    i4 = nns.upm(1, 0, x) - nns.lpm(1, 0, x) - m
    print(f"{name:7s} LPM1-UPM1(at mean)={i1:.2e} var={i2:.2e} LPM0+UPM0-1={i3:.1e} mean={i4:.2e}")

print("== ratio continuity (degree 1) and CDF degree 0")
x = cases["ties"]
for t in [0.9999, 1, 1.0001, 2 - 1e-9, 2, 2 + 1e-9, 4, 5]:
    print(t, nns.lpm_ratio(0, t, x), nns.lpm_ratio(1, t, x), nns.upm_ratio(1, t, x))

print("== A/B vs R on all cases (scalar mean target, vector targets <=32 and >32)")
for name, x in cases.items():
    t_small = np.linspace(x.min(), x.max(), 7)
    t_big = np.concatenate([np.sort(x)[:: max(1, len(x) // 200)], np.linspace(x.min() - 0.1, x.max() + 0.1, 40)])
    for d in [0, 1, 2, 3, 0.5, 1.5]:
        r = rcall("list(l=LPM(d,mean(x),x), u=UPM(d,mean(x),x), ls=LPM(d,ts,x), us=UPM(d,ts,x), lb=LPM(d,tb,x), ub=UPM(d,tb,x), rs=LPM.ratio(d,tb,x), rus=UPM.ratio(d,tb,x))",
                  d=d, x=x, ts=t_small, tb=t_big) if False else None
    # batch per case to keep Rscript calls low
    res = rcall("""lapply(ds, function(d) list(l=LPM(d,mean(x),x), u=UPM(d,mean(x),x), ls=LPM(d,ts,x), us=UPM(d,ts,x), lb=LPM(d,tb,x), ub=UPM(d,tb,x), rb=LPM.ratio(d,tb,x), urb=UPM.ratio(d,tb,x)))""",
                ds=[0, 1, 2, 3, 0.5, 1.5], x=x, ts=t_small, tb=t_big)
    for d, r in zip([0, 1, 2, 3, 0.5, 1.5], res):
        py = dict(l=nns.lpm(d, x.mean(), x), u=nns.upm(d, x.mean(), x),
                  ls=nns.lpm(d, t_small, x), us=nns.upm(d, t_small, x),
                  lb=nns.lpm(d, t_big, x), ub=nns.upm(d, t_big, x),
                  rb=nns.lpm_ratio(d, t_big, x), urb=nns.upm_ratio(d, t_big, x))
        for k in py:
            a = np.atleast_1d(np.asarray(py[k], float)); b = np.atleast_1d(arr(r[k]))
            ok = np.allclose(a, b, rtol=1e-9, atol=1e-15, equal_nan=True)
            if not ok:
                print(f"R MISMATCH {name} d={d} {k}: maxdiff={np.nanmax(np.abs(a-b)):.3g}")
print("done")
