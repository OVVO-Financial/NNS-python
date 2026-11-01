"""API / edge-case probes for PM core functions (Python and R)."""
import sys, warnings
sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, pandas as pd
import nns
from rrun import rcall, arr

def tryit(label, f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = f()
            print(f"{label}: {out!r}" + (f"  [warn: {w[0].message}]" if w else ""))
        except Exception as e:
            print(f"{label}: RAISES {type(e).__name__}: {e}")

x = np.array([1.0, 2, 3, np.nan, 5])
print("== NaN handling")
tryit("py lpm(1, 2, x with NaN)", lambda: nns.lpm(1, 2.0, x))
tryit("py lpm(0, [1..40], x with NaN) (vector)", lambda: nns.lpm(0, np.arange(40.0), x)[:4])
tryit("py lpm_ratio(1, 2, x with NaN)", lambda: nns.lpm_ratio(1, 2.0, x))
tryit("py lpm(1, nan, x)", lambda: nns.lpm(1, np.nan, np.array([1.0, 2, 3])))
tryit("py upm(1, inf, x)", lambda: nns.upm(1, np.inf, np.array([1.0, 2, 3])))
for code in ["LPM(1, 2, x)", "LPM.ratio(1, 2, x)", "UPM.ratio(0, 2, x)", "Co.LPM(1, x, x, 2, 2)", "PM.matrix(1,1,NULL,cbind(x,x),TRUE)$cov.matrix"]:
    try:
        print("R", code, "->", rcall(code.replace("x", "xx") if False else code, x=[1, 2, 3, "NA", 5]))
    except RuntimeError as e:
        print("R", code, "-> ERROR", e.args[0].strip().splitlines()[0][:100])
tryit("py co_lpm(1, x, x, 2, 2) with NaN", lambda: nns.co_lpm(1, x, x, 2.0, 2.0))
tryit("py pm_matrix with NaN", lambda: nns.pm_matrix(1, 1, None, np.column_stack([x, x]), True)["cov.matrix"])
tryit("py var_pm with NaN", lambda: nns.var_pm(x))
tryit("py nns_moments with NaN", lambda: nns.nns_moments(x))
tryit("py lpm_var(0.05, 1, x with NaN)", lambda: nns.lpm_var(0.05, 1, x))

print("== pandas / dtype inputs")
s = pd.Series([1, 2, 3, 4, 5], name="a")
tryit("lpm(1, 3, int Series)", lambda: nns.lpm(1, 3, s))
tryit("lpm(1, Series target, Series)", lambda: nns.lpm(1, pd.Series([2, 3]), s))
tryit("lpm(1, [2,3] list target, list x)", lambda: nns.lpm(1, [2, 3], [1, 2, 3, 4, 5]))
tryit("lpm(1, 2D target)", lambda: nns.lpm(1, np.array([[2.0, 3.0]]), s))
tryit("lpm(1, empty target)", lambda: nns.lpm(1, np.array([]), s))
tryit("lpm(-1, 3, x)", lambda: nns.lpm(-1, 3, s))
tryit("lpm(nan degree)", lambda: nns.lpm(np.nan, 3, s))
tryit("lpm(1, 3, DataFrame 1 col)", lambda: nns.lpm(1, 3, s.to_frame()))
tryit("lpm(1, 3, bool array)", lambda: nns.lpm(1, 0.5, np.array([True, False, True])))
tryit("lpm(1, 3, empty)", lambda: nns.lpm(1, 3, np.array([])))
df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 1, 4, 3]})
tryit("pm_matrix(DataFrame) keys", lambda: list(nns.pm_matrix(1, 1, None, df, True).keys()))
tryit("pm_matrix(DataFrame) cov type", lambda: type(nns.pm_matrix(1, 1, None, df, True)["cov.matrix"]))
tryit("pm_matrix(1D vector)", lambda: nns.pm_matrix(1, 1, None, np.array([1.0, 2, 3]), True))
tryit("pm_matrix target='median'", lambda: nns.pm_matrix(1, 1, "median", df, True)["cov.matrix"])
tryit("pm_matrix target=0 scalar", lambda: nns.pm_matrix(1, 1, 0, df, True)["cov.matrix"])
try:
    print("R PM.matrix scalar target:", rcall("PM.matrix(1,1,0,df,TRUE)$cov.matrix", df=df.to_dict("list")))
except RuntimeError as e:
    print("R PM.matrix scalar target -> ERROR", e.args[0].strip().splitlines()[0])
tryit("pm_matrix n=1 row", lambda: nns.pm_matrix(1, 1, None, np.array([[1.0, 2.0]]), True)["cov.matrix"])
print("R PM.matrix n=1:", rcall("PM.matrix(1,1,NULL,matrix(c(1,2),1),TRUE)$cov.matrix"))
tryit("co_lpm length mismatch", lambda: nns.co_lpm(1, np.arange(3.0), np.arange(4.0), 1, 1))
tryit("co_lpm(0, Series, Series, mean) ", lambda: nns.co_lpm(0, s, s * 2, 3, 6))
tryit("co_lpm target None (R default mean)", lambda: nns.co_lpm(1, s, s, None, None))
tryit("lpm target None (R default mean)", lambda: nns.lpm(1, None, s))
tryit("lpm_var vector percentile", lambda: nns.lpm_var(np.array([0.05, 0.5]), 1, np.arange(10.0)))
print("R LPM.VaR vector pct:", rcall("LPM.VaR(c(0.05,0.5), 1, 0:9)"))
print("R LPM.VaR deg0 returns named:", rcall("names(LPM.VaR(0.05, 0, 0:9))"))
tryit("lpm_var pct=1.5 (clamped)", lambda: nns.lpm_var(1.5, 1, np.arange(10.0)))
tryit("lpm_var(0.05, 1, Series)", lambda: nns.lpm_var(0.05, 1, s))
tryit("lpm_var(0.05, -1, x)", lambda: nns.lpm_var(0.05, -1, np.arange(10.0)))
print("R LPM.VaR deg -1:", rcall("LPM.VaR(0.05, -1, 0:9)"))
tryit("nns_moments(len 3, population=False)", lambda: nns.nns_moments(np.array([1.0, 2, 4]), population=False))
print("R NNS.moments len 3 sample:", rcall("NNS.moments(c(1,2,4), population=FALSE)"))
tryit("nns_moments(const)", lambda: nns.nns_moments(np.ones(5)))
print("R NNS.moments const:", rcall("NNS.moments(rep(1,5))"))
tryit("excess_ret kw", lambda: nns.lpm(1, np.array([1.0, 2, 3, 4, 5]), s, excess_ret=True))
print("R LPM excess_ret:", rcall("LPM(1, c(1,1,4,4,4), 1:5, excess_ret=TRUE)"))
