import numpy as np, pandas as pd, os
rng = np.random.default_rng(20261031)
out = "/home/user/audit/work/DEP/cases"
cases = {}
for n in [10, 30, 100, 1000, 10000]:
    cases[f"indep_n{n}"] = (rng.normal(size=n), rng.normal(size=n))
n = 1000
x = rng.normal(size=n)
cases["y_eq_x"] = (x, x.copy())
cases["y_neg_x"] = (x, -x)
cases["y_x2"] = (x, x**2)
cases["y_negx3"] = (x, -x**3)
u = rng.uniform(-3, 3, size=n)
cases["y_sin"] = (u, np.sin(2*u))
th = rng.uniform(0, 2*np.pi, size=n)
cases["circle"] = (np.cos(th), np.sin(th))
cases["circle_noisy"] = (np.cos(th)+rng.normal(0,.05,n), np.sin(th)+rng.normal(0,.05,n))
cases["step"] = (u, (u > 0).astype(float))
cases["step3"] = (u, np.floor(u))
cases["linear_noise"] = (x, x + rng.normal(size=n))
cases["neg_linear_noise"] = (x, -x + rng.normal(size=n))
cases["exp_x_mono"] = (np.exp(x), x + rng.normal(size=n))
cases["x_mono_base"] = (x, x + rng.normal(size=n))  # same y as above? regenerate below
e = rng.normal(size=n); yy = x + e
cases["mono_base"] = (x, yy)
cases["mono_expx"] = (np.exp(x), yy)
cases["mono_x3"] = (x**3, yy)
cases["mono_expy"] = (x, np.exp(yy))
cases["mono_rank"] = (pd.Series(x).rank().values, pd.Series(yy).rank().values)
cases["indep_exp"] = (rng.exponential(size=n), rng.exponential(size=n))
cases["indep_exp_10k"] = (rng.exponential(size=10000), rng.exponential(size=10000))
cases["indep_t2"] = (rng.standard_t(2, size=n), rng.standard_t(2, size=n))
cases["discrete_indep"] = (rng.integers(0, 5, n).astype(float), rng.integers(0, 5, n).astype(float))
d = rng.integers(0, 5, n).astype(float)
cases["discrete_dep"] = (d, d + rng.integers(0, 2, n))
cases["binary_eq"] = ((d > 2).astype(float), (d > 2).astype(float))
cases["ties_heavy"] = (np.round(x, 0), np.round(yy, 0))
cases["len3"] = (np.array([1., 2, 3]), np.array([3., 1, 2]))
cases["len2"] = (np.array([1., 2]), np.array([2., 1]))
cases["len5"] = (np.array([1., 2, 3, 4, 5]), np.array([2., 1, 4, 3, 5]))
cases["const_y"] = (x[:50], np.ones(50))
cases["int_seq_rev"] = (np.arange(1, 101.), -np.arange(1, 101.))
cases["int_seq"] = (np.arange(1, 101.), np.arange(1, 101.))
cases["unsorted_quad"] = (u, u**2 + rng.normal(0, .1, n))
r = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)
cols = list(r.columns)
for a, b in [("SPY", "QQQ"), ("SPY", "TLT"), ("SPY", "GLD"), ("TLT", "GLD")]:
    if a in cols and b in cols:
        cases[f"ret_{a}_{b}"] = (r[a].values, r[b].values)
for k, (a, b) in cases.items():
    pd.DataFrame({"x": a, "y": b}).to_csv(f"{out}/{k}.csv", index=False)
print(len(cases), cols)
