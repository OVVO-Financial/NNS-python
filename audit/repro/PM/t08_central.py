"""NNS.mode / NNS.gravity / NNS.rescale A/B vs R."""
import sys, warnings
sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, pandas as pd
import nns
from rrun import rcall, arr

ret = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)
mt = pd.read_csv("/home/user/audit/data/mtcars.csv")
air = pd.read_csv("/home/user/audit/data/airquality.csv")
rng = np.random.default_rng(11)
cases = {
    "norm100": rng.normal(size=100), "bimodal": np.concatenate([rng.normal(-3, 1, 300), rng.normal(3, 1, 200)]),
    "nvda": ret["NVDA"].to_numpy(), "ints": rng.integers(0, 10, 200).astype(float), "mpg": mt["mpg"].to_numpy(),
    "cyl": mt["cyl"].to_numpy(dtype=float), "len3": np.array([1., 5, 2]), "len4": np.array([1., 5, 2, 2]),
    "const": np.full(7, 2.5), "two": np.array([1.0, 2.0]), "ozone_na": air["Ozone"].to_numpy(dtype=float),
    "neg_half": np.array([-2.5, -2.5, -1.5, -0.5, 0.5, 0.5, 1.5]), "expon": rng.exponential(size=1000),
}
bad = 0
for name, x in cases.items():
    r = rcall("""list(m_c_multi=NNS.mode(x, FALSE, TRUE), m_c=NNS.mode(x, FALSE, FALSE), m_d_multi=NNS.mode(x, TRUE, TRUE), m_d=NNS.mode(x, TRUE, FALSE),
                     m_default=NNS.mode(x), g=NNS.gravity(x), g_d=NNS.gravity(x, TRUE))""", x=[v if np.isfinite(v) else "NA" for v in x])
    py = {"m_c_multi": nns.nns_mode(x, False, True), "m_c": nns.nns_mode(x, False, False), "m_d_multi": nns.nns_mode(x, True, True),
          "m_d": nns.nns_mode(x, True, False), "m_default": nns.nns_mode(x), "g": nns.nns_gravity(x), "g_d": nns.nns_gravity(x, True)}
    for k in py:
        a = np.atleast_1d(np.asarray(py[k], float)); b = np.atleast_1d(arr(r[k]))
        ok = a.shape == b.shape and np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)
        if not ok:
            bad += 1
            print(f"MISMATCH {name:9s} {k:10s}: py={np.round(a, 6)} R={np.round(b, 6)}")
print("mismatches:", bad)
print("== rescale")
x = rng.normal(size=50)
for args in [dict(a=5, b=10), dict(a=10, b=5)]:
    r = rcall("NNS.rescale(x, a, b)", x=x, **args)
    print("minmax", args, np.max(np.abs(nns.nns_rescale(x, **args) - arr(r))))
pr = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 100)))
for tp in ["Terminal", "Discounted"]:
    r = rcall("NNS.rescale(pr, 100, 0.05, 'riskneutral', T=1, type=tp)", pr=pr, tp=tp)
    py = nns.nns_rescale(pr, 100, 0.05, "riskneutral", time_to_maturity=1, type=tp)
    print("riskneutral", tp, np.max(np.abs(py - arr(r))), "mean", py.mean())
for code, pyf in [("NNS.rescale(c(1,1,1), 0, 1)", lambda: nns.nns_rescale(np.array([1., 1, 1]), 0, 1)),
                  ("NNS.rescale(c(1,NA,3), 0, 1)", lambda: nns.nns_rescale(np.array([1., np.nan, 3]), 0, 1)),
                  ("NNS.rescale(c(1,2,3), 0, 1, 'riskneutral')", lambda: nns.nns_rescale(np.array([1., 2, 3]), 0, 1, 'riskneutral')),
                  ("NNS.rescale(c(1,2,3), 100, 0.05, 'riskneutral', T=1, type='bogus')", lambda: nns.nns_rescale(np.array([1., 2, 3]), 100, 0.05, 'riskneutral', 1, 'bogus')),
                  ("NNS.rescale(c(1,2,3), 0, 1, 'MinMax')", lambda: nns.nns_rescale(np.array([1., 2, 3]), 0, 1, 'MinMax'))]:
    try:
        rv = rcall(code)
    except RuntimeError as e:
        rv = "ERROR " + e.args[0].strip().splitlines()[0][:80]
    try:
        pv = pyf()
    except Exception as e:
        pv = f"RAISES {type(e).__name__}: {e}"
    print(code, "\n   R:", rv, "\n   py:", pv)
