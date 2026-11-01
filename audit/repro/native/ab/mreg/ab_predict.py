import numpy as np
import nns._reg_engine as eng
orig = eng.native_fn
def py(fn, *a):
    eng.native_fn = lambda name: None
    try: return fn(*a)
    finally: eng.native_fn = orig
rng = np.random.default_rng(0)
worst = {}; n_cases = 0
for trial in range(400):
    n = int(rng.integers(2, 300)); p = int(rng.integers(1, 6)); m = int(rng.integers(1, 80))
    kind = trial % 5
    if kind == 0: X = rng.normal(size=(n, p))
    elif kind == 1: X = rng.integers(0, 4, size=(n, p)).astype(float)        # heavy distance ties
    elif kind == 2: X = np.repeat(rng.normal(size=(n, 1)), p, axis=1)          # duplicated columns (1-D path)
    elif kind == 3: X = np.round(rng.normal(size=(n, p)), 1); X[:, 0] = 1.0     # a zero-range column
    else: X = rng.normal(size=(n, p)) * 10.0 ** rng.integers(-3, 4, size=p)
    yhat = rng.normal(size=n)
    xt = X[rng.integers(0, n, m)] + (0 if kind in (1, 2) else rng.normal(scale=0.1, size=(m, p)))
    mins, maxs = X.min(axis=0), X.max(axis=0)
    for dist in ("NNS", "L2", "L1", "FACTOR"):
        for is_class in (False, True):
            yh = np.round(yhat) if is_class else yhat
            for k in (1, 2, 5, n):
                a = eng._mreg_predict(xt, X, yh, k, dist, mins, maxs, is_class)
                b = py(eng._mreg_predict, xt, X, yh, k, dist, mins, maxs, is_class)
                d = float(np.nanmax(np.abs(a - b) / np.maximum(1, np.abs(b))))
                key = (dist, is_class, "k=1" if k == 1 else "k>1")
                worst[key] = max(worst.get(key, 0), d); n_cases += 1
        kmax = int(min(n, 30))
        a = eng._mreg_predict_path(xt, X, yhat, kmax, dist, mins, maxs)
        b = py(eng._mreg_predict_path, xt, X, yhat, kmax, dist, mins, maxs)
        d = float(np.nanmax(np.abs(a - b) / np.maximum(1, np.abs(b))))
        worst[(dist, "path")] = max(worst.get((dist, "path"), 0), d); n_cases += 1
print("cases", n_cases)
for k, v in sorted(worst.items(), key=str): print(k, f"{v:.3g}")
