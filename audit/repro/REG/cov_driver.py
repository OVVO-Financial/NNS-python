import sys, warnings, numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, "/home/user/audit/work/REG")
import run_py, os
run_py.OUT = "/home/user/audit/work/REG/out_py_cov"; os.makedirs(run_py.OUT, exist_ok=True)
run_py.main(sorted(os.listdir(run_py.ROOT)))
import nns
rng = np.random.default_rng(0); X = rng.standard_normal((200, 3)); y = X[:, 0] + X[:, 1] ** 2
nns.nns_stack(X[:150], y[:150], X[150:], folds=2, status=False, ncores=1)
nns.nns_boost(X[:150], y[:150], X[150:], epochs=3, learner_trials=3, status=False, folds=1) if hasattr(nns, "nns_boost") else None
nns.dy_d(X, y, 1, X[:5], messages=False); nns.dy_dx(X[:, 0], y, [0.0, 0.5]); nns.dy_dx(X[:, 0], y, "overall")
nns.nns_arma(np.sin(np.arange(120) / 5.0), h=5, seasonal_factor=12, method="nonlin", plot=False)
nns.nns_m_reg(X, y, point_est=X[:3])
nns.prepare_factor_predictors(np.array([["a", 1.0], ["b", 2.0]], dtype=object), factor_levels=(["a", "b"], None))
nns.nns_distance(np.column_stack([X, y]), X[0])
