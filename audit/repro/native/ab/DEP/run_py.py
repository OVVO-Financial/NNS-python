import numpy as np, pandas as pd, json, glob, os, sys, warnings
import nns
out = {}
for f in sorted(glob.glob("/home/user/audit/work/DEP/cases/*.csv")):
    k = os.path.basename(f)[:-4]
    d = pd.read_csv(f); x = d.x.values; y = d.y.values
    r = {}
    def tr(name, fn):
        try: r[name] = fn()
        except Exception as e: r[name] = f"ERR {type(e).__name__}: {e}"
    tr("dep", lambda: nns.nns_dep(x, y))
    tr("dep_yx", lambda: nns.nns_dep(y, x))
    tr("asym_xy", lambda: nns.nns_dep(x, y, asym=True))
    tr("asym_yx", lambda: nns.nns_dep(y, x, asym=True))
    tr("copula", lambda: nns.nns_copula(x, y))
    tr("copula_disc", lambda: nns.nns_copula(x, y, continuous=False))
    tr("caus0", lambda: nns.nns_causation(x, y, tau=0) if len(x) <= 3000 else "skip")
    tr("caus3", lambda: nns.nns_causation(x, y, tau=3) if 10 < len(x) <= 3000 else "skip")
    tr("pearson", lambda: float(np.corrcoef(x, y)[0, 1]))
    out[k] = r
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
