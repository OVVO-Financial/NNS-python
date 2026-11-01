"""Timing: Python vs R (R single-threaded via RcppParallel::setThreadOptions(1)).
Run as:  python t09_timing.py            (python uses all hw threads - no knob)
         taskset -c 0 python t09_timing.py py1   (python pinned to one core)"""
import sys, time, json, warnings
sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np
import nns
from rrun import rcall

only_py = len(sys.argv) > 1 and sys.argv[1] == "py1"
rng = np.random.default_rng(0)

def tpy(f, reps):
    f()
    best = np.inf
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            f()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best

R_TIMER = """
tm <- function(expr_fun, reps) { expr_fun(); best <- Inf; for (k in 1:3) { t0 <- proc.time()[[3]]; for (i in 1:reps) expr_fun(); best <- min(best, (proc.time()[[3]] - t0)/reps) }; best }
"""

results = []
for n in [10_000, 100_000, 1_000_000]:
    x = rng.standard_t(3, size=n) * 0.01
    reps = max(1, int(2e6 // n))
    tv = np.quantile(x, np.linspace(0.001, 0.999, 1000))
    t10 = tv[::100][:10]
    bench = {
        "lpm(1, scalar, x)": (lambda: nns.lpm(1, 0.0, x), "LPM(1, 0, x)", reps),
        "lpm(2, scalar, x)": (lambda: nns.lpm(2, 0.0, x), "LPM(2, 0, x)", reps),
        "lpm(1.5, scalar, x)": (lambda: nns.lpm(1.5, 0.0, x), "LPM(1.5, 0, x)", reps),
        "lpm_ratio(1, scalar, x)": (lambda: nns.lpm_ratio(1, 0.0, x), "LPM.ratio(1, 0, x)", reps),
        "lpm(1, 10 targets, x)": (lambda: nns.lpm(1, t10, x), "LPM(1, t10, x)", max(1, reps // 10)),
        "lpm(2, 1000 targets, x)": (lambda: nns.lpm(2, tv, x), "LPM(2, tv, x)", max(1, reps // 20)),
        "lpm(1.5, 1000 targets, x)": (lambda: nns.lpm(1.5, tv, x), "LPM(1.5, tv, x)", 1),
        "lpm_ratio(1, 1000 targets, x)": (lambda: nns.lpm_ratio(1, tv, x), "LPM.ratio(1, tv, x)", max(1, reps // 20)),
        "lpm_var(0.05, 1, x)": (lambda: nns.lpm_var(0.05, 1, x), "LPM.VaR(0.05, 1, x)", max(1, reps // 20)),
        "lpm_var(0.05, 2, x)": (lambda: nns.lpm_var(0.05, 2, x), "LPM.VaR(0.05, 2, x)", max(1, reps // 20)),
        "lpm_var(0.05, 0, x)": (lambda: nns.lpm_var(0.05, 0, x), "LPM.VaR(0.05, 0, x)", max(1, reps // 20)),
        "co_lpm(1, x, x, 0, 0)": (lambda: nns.co_lpm(1, x, x[::-1], 0.0, 0.0), "Co.LPM(1, x, rev(x), 0, 0)", reps),
    }
    if n <= 100_000:
        bench["lpm_var(0.05, 1.5, x)"] = (lambda: nns.lpm_var(0.05, 1.5, x), "LPM.VaR(0.05, 1.5, x)", 1)
        bench["nns_cdf(x, 0)"] = (lambda: nns.nns_cdf(x, 0), "NNS.CDF(x, 0, plot=FALSE)", 1)
        bench["nns_cdf(x, 1)"] = (lambda: nns.nns_cdf(x, 1), "NNS.CDF(x, 1, plot=FALSE)", 1)
    X = rng.standard_t(4, size=(n, 11)) * 0.01
    bench["pm_matrix(1,1,mean, n x 11)"] = (lambda: nns.pm_matrix(1, 1, None, X, True), "PM.matrix(1, 1, NULL, X, TRUE)", max(1, reps // 20))
    if n == 10_000:
        X100 = rng.normal(size=(n, 100))
        bench["pm_matrix(1,1,mean, 1e4 x 100)"] = (lambda: nns.pm_matrix(1, 1, None, X100, True), "PM.matrix(1, 1, NULL, X100, TRUE)", 1)
    py = {k: tpy(v[0], v[2]) for k, v in bench.items()}
    if only_py:
        for k in bench:
            print(f"n={n:>8} {k:34s} py(1core)={py[k]*1e3:9.3f} ms", flush=True)
        continue
    rcode = R_TIMER + "list(" + ",".join(f"`{k}`=tm(function() {v[1]}, {v[2]})" for k, v in bench.items()) + ")"
    data = dict(x=x, tv=tv, t10=t10, X=X)
    if n == 10_000:
        data["X100"] = X100
    rr = rcall(rcode, **data)
    for k in bench:
        print(f"n={n:>8} {k:34s} py={py[k]*1e3:9.3f} ms  R={rr[k]*1e3:9.3f} ms  py/R={py[k]/max(rr[k],1e-9):7.2f}", flush=True)
