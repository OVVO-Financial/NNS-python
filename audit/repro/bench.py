"""Before/after timing of the main public paths. Usage: python bench.py <label>"""
import json, sys, time, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
import nns
try:
    nns.set_num_threads(1)
except Exception:
    pass
rng = np.random.default_rng(42)
R = pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)
spy, qqq, tlt = R["SPY"].to_numpy(), R["QQQ"].to_numpy(), R["TLT"].to_numpy()
bos = pd.read_csv("/home/user/audit/data/boston.csv")
ap = pd.read_csv("/home/user/audit/data/airpassengers.csv")["y"].to_numpy()
X5 = rng.normal(size=(10000, 5)); y5 = np.sin(X5).sum(axis=1) + rng.normal(scale=0.1, size=10000)
x1 = rng.normal(size=10000); y1 = np.sin(x1) + rng.normal(scale=0.1, size=10000)
M = rng.normal(size=(4000, 2))
cases = {
    "lpm 1000 targets x 1e5": lambda: nns.lpm(2, np.linspace(-0.05, 0.05, 1000), rng.standard_t(3, 100000) * 0.01),
    "nns_dep SPY/TLT daily (n=4.2k)": lambda: nns.nns_dep(spy, tlt),
    "nns_dep n=1e5": lambda: nns.nns_dep(np.r_[spy, qqq][: 8000].repeat(12)[:100000], np.r_[tlt, spy][:8000].repeat(12)[:100000]),
    "nns_cdf univariate n=1e4 degree 1.5": lambda: nns.nns_cdf(x1, 1.5),
    "nns_cdf bivariate n=4000 degree 1": lambda: nns.nns_cdf(M, 1),
    "nns_reg univariate n=1e4": lambda: nns.nns_reg(x1, y1),
    "nns_reg multivariate n=1e4 p=5": lambda: nns.nns_reg(X5, y5),
    "nns_reg order='max' n=1e4 p=5": lambda: nns.nns_reg(X5, y5, order="max"),
    "nns_stack Boston (5-fold)": lambda: nns.nns_stack(bos.drop(columns="medv").to_numpy()[:400], bos["medv"].to_numpy()[:400], bos.drop(columns="medv").to_numpy()[400:]),
    "fsd_uni continuous n=1e4": lambda: nns.fsd_uni(x1, np.random.default_rng(7).normal(0.05, 1.0, 10000), type="continuous"),
    "sd_efficient_set 11 assets daily": lambda: nns.sd_efficient_set(R.to_numpy(), degree=1),
    "nns_arma_optim AirPassengers": lambda: nns.nns_arma_optim(ap[:132], h=12, seasonal_factor=[12, 6, 24]),
    "nns_seas n=1e4": lambda: nns.nns_seas(np.sin(np.arange(10000) * 2 * np.pi / 12) + rng.normal(size=10000)),
    "causal_matrix 10 cols n=1000": lambda: nns.causal_matrix(rng.normal(size=(1000, 10))),
}
out = {}
for name, fn in cases.items():
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        try:
            fn(); ts.append(time.perf_counter() - t)
        except Exception as e:
            ts = [float("nan")]; out[name + " (error)"] = str(e)[:80]; break
    out[name] = min(ts)
    print(f"{name:40s} {min(ts):8.3f} s", flush=True)
json.dump(out, open(f"/home/user/audit/work/bench/{sys.argv[1]}.json", "w"), indent=1)
