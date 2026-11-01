"""Check: does using R's uniroot tolerance (eps^0.25) in _uniroot_extend reproduce R's NNS.diff?"""
import json, numpy as np
import nns.diff as nd
from scipy import optimize
def uniroot_r(fn, lower, upper):
    lo, hi = lower, upper
    f_lo, f_hi = fn(lo), fn(hi)
    for _ in range(100):
        if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0:
            return float(optimize.brentq(fn, lo, hi, xtol=np.finfo(float).eps ** 0.25, rtol=4 * np.finfo(float).eps, maxiter=1000))
        lo *= 2; hi *= 2; f_lo, f_hi = fn(lo), fn(hi)
    return np.nan
nd._uniroot_extend = uniroot_r
fs = {"sin1": (np.sin, 1), "sq3": (lambda x: x**2, 3), "cub2": (lambda x: x**3, 2), "log5": (np.log, 5), "tanh05": (np.tanh, 0.5)}
r = json.load(open("nnsdiff_r.json"))
for k, (f, p) in fs.items():
    py = nd.nns_diff(f, p); rr = dict(zip(py.keys(), r[k]))
    print(f"{k}: R deriv={rr['DERIVATIVE']:.12f} it={rr['iterations']} conv={rr['converged']} | py(R-tol) deriv={py['DERIVATIVE']:.12f} it={py['iterations']:.0f} conv={py['converged']:.0f}")
