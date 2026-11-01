import numpy as np, json, subprocess, sys
from nns._native import native_fn as nf
from nns.dependence import _gravity_python
rng = np.random.default_rng(0); vecs = []; fb_eq = 0
for i in range(3000):
    n = int(rng.integers(4, 300)); k = rng.integers(0, 6)
    if k == 0: x = rng.normal(size=n)
    elif k == 1: x = rng.integers(0, 5, n).astype(float)
    elif k == 2: x = np.round(rng.standard_t(2, size=n), 2)
    elif k == 3: x = rng.exponential(size=n) * 1e3 + 1e6
    elif k == 4: x = rng.normal(size=n); x[0] = -1e5          # DEP-10: > 1e6 bins
    else: x = 5 + 1e-7 * rng.normal(size=n); x[0] = 6.0       # CORE-08 tight IQR
    exact = nf('gravity_exact')(x); fb = _gravity_python(x); fb_eq += (fb == exact)
    vecs.append({'x': x.tolist(), 'core': nf('gravity')(x, False), 'exact': exact, 'fallback': fb})
print("fallback == native gravity_exact:", fb_eq, "/", len(vecs))
json.dump(vecs, open('grav_vecs.json', 'w'))
print(subprocess.run(["Rscript", "-e", 'suppressMessages(library(NNS)); v<-jsonlite::fromJSON("grav_vecs.json", simplifyVector=FALSE); r<-vapply(v, function(e) NNS.gravity(unlist(e$x)), 1); core<-vapply(v, function(e) e$core, 1); ex<-vapply(v, function(e) e$exact, 1); fb<-vapply(v, function(e) e$fallback, 1); cat("nns::gravity == R:", sum(core==r), "/", length(r), "  gravity_exact == R:", sum(ex==r), "/", length(r), "  fallback == R:", sum(fb==r), "\\n")'], capture_output=True, text=True).stdout)
