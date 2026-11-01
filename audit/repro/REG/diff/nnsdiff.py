import json, numpy as np
from nns import nns_diff
fs = {"sin1": (np.sin, 1), "sq3": (lambda x: x**2, 3), "exp0": (np.exp, 0), "cub2": (lambda x: x**3, 2), "log5": (np.log, 5),
      "abs1": (abs, -1), "lin": (lambda x: 2*x + 1, 4), "sq0": (lambda x: x**2, 0), "tanh05": (np.tanh, 0.5), "neg": (lambda x: x**4 - 3*x, -2)}
truth = {"sin1": np.cos(1), "sq3": 6, "exp0": 1, "cub2": 12, "log5": .2, "abs1": -1, "lin": 2, "sq0": 0, "tanh05": 1/np.cosh(.5)**2, "neg": 4*(-8) - 3}
r = json.load(open("nnsdiff_r.json"))
for k, (f, p) in fs.items():
    try:
        py = nns_diff(f, p)
    except Exception as e:
        print(k, "ERROR", e); continue
    rr = dict(zip(py.keys(), r[k]))
    bad = [kk for kk in py if not (np.isclose(py[kk], rr[kk] if rr[kk] is not None else np.nan, rtol=1e-9, equal_nan=True))]
    print(f"{k:7s} truth={truth[k]:.10f} R={rr['DERIVATIVE']:.10f} py={py['DERIVATIVE']:.10f} iterR={rr['iterations']} iterPy={py['iterations']} conv R/py={rr['converged']}/{py['converged']} diffs={bad}")
