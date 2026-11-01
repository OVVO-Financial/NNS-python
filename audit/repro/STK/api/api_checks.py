import warnings, numpy as np, pandas as pd, traceback
warnings.simplefilter("ignore")
from nns import nns_stack, nns_stack_fit, nns_stack_predict, nns_boost
iris=pd.read_csv("/home/user/audit/data/iris.csv")
X=iris.iloc[:,:4]; y=iris.Species
def t(label, f):
    try: r=f(); print(f"[{label}] OK ->", r)
    except Exception as e: print(f"[{label}] {type(e).__name__}: {e}")
# 1 pandas DataFrame + Series
t("pandas DF/Series", lambda: nns_stack(X.iloc[:140], y.iloc[:140], X.iloc[140:], type="CLASS")["stack"])
# 2 pandas Categorical with explicit order (reverse) -> does Python honour category order?
yc=pd.Categorical(y, categories=["virginica","versicolor","setosa"])
t("Categorical reversed levels: class.levels", lambda: nns_stack(X.iloc[:140], yc[:140], X.iloc[140:], type="CLASS")["class.levels"])
# 3 class_levels kwarg on shuffled data
rs=np.random.RandomState(1); p=rs.permutation(150); p=np.r_[p[y.values[p]=="virginica"][:1], p[y.values[p]!="virginica"], p[y.values[p]=="virginica"][1:]]
Xs=X.values[p]; ys=y.values[p]
a=nns_stack(Xs[:135], ys[:135], Xs[135:], type="CLASS")
b=nns_stack(Xs[:135], ys[:135], Xs[135:], type="CLASS", class_levels=["setosa","versicolor","virginica"])
print("[class_levels ignored?] levels a", a["class.levels"], "b", b["class.levels"], "same stack:", np.array_equal(a["stack"], b["stack"]))
# 4 NaN in X
Xn=X.values.copy(); Xn[3,1]=np.nan
t("NaN in X", lambda: nns_stack(Xn[:140], y.values[:140], Xn[140:], type="CLASS")["stack"])
# 5 NaN in y regression
yn=X.values[:,0].copy(); yn[5]=np.nan
t("NaN in y", lambda: nns_stack(X.values[:140,1:], yn[:140], X.values[140:,1:])["stack"][:3])
# 6 integer-valued regression target with 2 unique values -> auto-class
t("numeric 2-valued y auto-class (R does same)", lambda: nns_stack(X.values[:140], (y.values[:140]=="setosa").astype(float)*10, X.values[140:])["class.levels"])
# 7 bool y
t("bool y", lambda: nns_stack(X.values[:140], (y.values[:140]=="virginica"), X.values[140:])["stack"])
# 8 dim.red.variables return type
r=nns_stack(X.iloc[:140], y.iloc[:140], X.iloc[140:], type="CLASS")
print("[dim.red.variables]", r["NNS.dim.red.variables"], type(r["NNS.dim.red.coefficients"]))
# 9 objective case/invalid
t("objective='MAX'", lambda: nns_stack(X.values[:100,1:], X.values[:100,0], X.values[100:,1:], objective="MAX", obj_fn=lambda p,a: -float(np.sum((p-a)**2)))["NNS.reg.n.best"])
# 10 method as string '1'
t("method='1'", lambda: nns_stack(X.values[:100,1:], X.values[:100,0], method="1")["NNS.reg.n.best"])
# 11 folds as float 3.0 / bool
t("folds=True", lambda: nns_stack(X.values[:100,1:], X.values[:100,0], folds=True)["NNS.reg.n.best"])
# 12 obj_fn returning numpy array
t("obj_fn returns vector", lambda: nns_stack(X.values[:100,1:], X.values[:100,0], obj_fn=lambda p,a: (p-a)**2)["NNS.reg.n.best"])
# 13 random_seed alias override
r1=nns_stack(X.values[:100,1:], X.values[:100,0], seed=123, random_seed=5)["OBJfn.reg"]; r2=nns_stack(X.values[:100,1:], X.values[:100,0], seed=5)["OBJfn.reg"]
print("[seed=123 explicit + random_seed=5 -> uses 5?]", r1==r2)
# 14 fit/predict equivalence
for bal in (False, True):
    kw=dict(type="CLASS", balance=bal)
    full=nns_stack(X.values[:140], y.values[:140], X.values[140:], **kw)
    fit=nns_stack_fit(X.values[:140], y.values[:140], **kw); pr=nns_stack_predict(fit, X.values[140:])
    print(f"[fit/predict balance={bal}] stack equal:", np.array_equal(full["stack"], pr["stack"]), "weights", full["weights"], pr["weights"])
Xr=X.values[:,1:]; yr=X.values[:,0]
full=nns_stack(Xr[:120], yr[:120], Xr[120:]); fit=nns_stack_fit(Xr[:120], yr[:120]); pr=nns_stack_predict(fit, Xr[120:])
print("[fit/predict regression] maxdiff", np.max(np.abs(full["stack"]-pr["stack"])))
# 15 boost pandas names
bb=nns_boost(X.iloc[:140], y.iloc[:140], X.iloc[140:], epochs=5, learner_trials=5, type="CLASS")
print("[boost names]", bb["feature.frequency"], "results dtype", np.asarray(bb["results"]).dtype)
# 16 boost numeric 2-class not auto (R same)
t("boost numeric binary w/o type", lambda: nns_boost(X.values[:140], (y.values[:140]=="setosa").astype(int), X.values[140:], epochs=5, learner_trials=5)["results"][:5])
# 17 boost string labels sorted vs stack first-appearance
bb=nns_boost(Xs[:135], ys[:135], Xs[135:], epochs=5, learner_trials=5, type="CLASS"); print("[boost levels shuffled]", bb["class.levels"])
# 18 ivs_test 1-row vector
t("1-row test vector", lambda: nns_stack(Xr[:100], yr[:100], Xr[100])["stack"])
# 19 ts_test with too-large folds
t("ts_test=60 folds=5 n=120", lambda: nns_stack(Xr[:120], yr[:120], Xr[120:], ts_test=60)["NNS.reg.n.best"])
# 20 constant predictor column
Xc=Xr[:100].copy(); Xc[:,1]=1.0
t("constant column", lambda: nns_stack(Xc, yr[:100], Xc[:5])["NNS.dim.red.coefficients"])
