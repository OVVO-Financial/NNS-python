import json, subprocess, sys, numpy as np
import nns._reg_engine as eng
orig = eng.native_fn
def py(fn, *a):
    eng.native_fn = lambda name: None
    try: return fn(*a)
    finally: eng.native_fn = orig
D = "/home/user/audit/work/native/ab/mreg"
rng = np.random.default_rng(0); cases = []
for trial in range(150):
    n = int(rng.integers(2, 200)); p = int(rng.integers(1, 6)); m = int(rng.integers(1, 40))
    kind = trial % 5
    if kind == 0: X = rng.normal(size=(n, p))
    elif kind == 1: X = rng.integers(0, 4, size=(n, p)).astype(float)
    elif kind == 2: X = np.repeat(rng.normal(size=(n, 1)), p, axis=1)
    elif kind == 3: X = np.round(rng.normal(size=(n, p)), 1); X[:, 0] = 1.0
    else: X = rng.normal(size=(n, p)) * 10.0 ** rng.integers(-3, 4, size=p)
    yhat = rng.normal(size=n)
    xt = X[rng.integers(0, n, m)] + (0 if kind in (1, 2) else rng.normal(scale=0.1, size=(m, p)))
    for dist_code, dist in enumerate(("NNS", "L2", "L1", "FACTOR")):
        for is_class in (False, True):
            yh = np.round(yhat) if is_class else yhat
            for k in (1, 2, 5, n):
                cases.append(dict(X=X.tolist(), y=yh.tolist(), t=xt.tolist(), k=k, dist=dist_code, cls=is_class,
                                  nat=eng._mreg_predict(xt, X, yh, k, dist, X.min(0), X.max(0), is_class).tolist(),
                                  py=py(eng._mreg_predict, xt, X, yh, k, dist, X.min(0), X.max(0), is_class).tolist()))
        kmax = int(min(n, 20))
        cases.append(dict(X=X.tolist(), y=yhat.tolist(), t=xt.tolist(), k=kmax, dist=dist_code, cls=False, path=True,
                          nat=eng._mreg_predict_path(xt, X, yhat, kmax, dist, X.min(0), X.max(0)).ravel(order="F").tolist(),
                          py=py(eng._mreg_predict_path, xt, X, yhat, kmax, dist, X.min(0), X.max(0)).ravel(order="F").tolist()))
json.dump(cases, open(f"{D}/cases.json", "w"))
r = subprocess.run(["Rscript", "-e", f'''suppressMessages(library(NNS)); cs <- jsonlite::fromJSON("{D}/cases.json", simplifyVector=FALSE)
out <- lapply(cs, function(e) {{ X <- do.call(rbind, lapply(e$X, unlist)); t <- do.call(rbind, lapply(e$t, unlist)); y <- unlist(e$y)
 f <- if (isTRUE(e$path)) NNS:::NNS_mreg_predict_path_v2_cpp else NNS:::NNS_mreg_predict_v2_cpp
 as.numeric(f(X, y, t, as.integer(e$k), as.integer(e$dist), apply(X,2,min), apply(X,2,max), isTRUE(e$cls), 1L)) }})
writeLines(jsonlite::toJSON(out, digits=NA), "{D}/r_out.json")'''], capture_output=True, text=True)
if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
R = json.load(open(f"{D}/r_out.json"))
def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.nanmax(np.abs(a - b) / np.maximum(1, np.abs(b)))) if a.size else 0.0
nat = [rel(c["nat"], r) for c, r in zip(cases, R)]; pyd = [rel(c["py"], r) for c, r in zip(cases, R)]
print(f"cases {len(cases)}; native vs R: max rel {max(nat):.3g}, >1e-12 in {sum(d > 1e-12 for d in nat)}; "
      f"NumPy vs R: max rel {max(pyd):.3g}, >1e-12 in {sum(d > 1e-12 for d in pyd)}")
