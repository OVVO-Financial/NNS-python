import warnings, numpy as np, pandas as pd
import nns
warnings.simplefilter("ignore")
iris = pd.read_csv("/home/user/audit/data/iris.csv")
X = iris.iloc[:, :4].values; ys = iris.Species.values
def t(label, f):
    try:
        r = f(); print(label, "->", r)
    except Exception as e:
        print(label, "-> EXC", type(e).__name__, str(e)[:150])
# 1. nns_m_reg with string y, no type -> should auto classify (R: character y auto-class)
t("m_reg str y", lambda: (nns.nns_m_reg(X, ys)["R2"], nns.nns_m_reg(X, ys)["class.levels"]))
# 2. nns_m_reg with class_levels but no type -> codes become numeric regression?
t("m_reg class_levels no type", lambda: (lambda r: (r["R2"], r["class.levels"], np.unique(r["Fitted.xy"]["y.hat"])[:6]))(nns.nns_m_reg(X, ys, class_levels=["virginica", "versicolor", "setosa"])))
# 3. nns_reg class_levels ignored
r_a = nns.nns_reg(X, ys, type="CLASS", point_est=X[[0, 60, 120]])
r_b = nns.nns_reg(X, ys, type="CLASS", point_est=X[[0, 60, 120]], class_levels=["virginica", "versicolor", "setosa"])
print("nns_reg class_levels ignored:", r_a["Point.est"], r_b["Point.est"], r_b["class.levels"])
# 4. object-dtype numeric y
b = pd.read_csv("/home/user/audit/data/boston.csv")
yo = b.medv.values.astype(object)
t("object numeric y", lambda: (lambda r: (r["R2"], r["class.levels"][:5] if r["class.levels"] else None))(nns.nns_reg(b.lstat.values, yo)))
# 5. nullable Int64 with NA
yi = pd.Series([1, 2, 3, None] * 25, dtype="Int64")
t("Int64 with NA y", lambda: (lambda r: (r["R2"], r["class.levels"]))(nns.nns_reg(np.arange(100.0), yi)))
# 6. categorical x with missing values
xc = pd.Series(["a", "b", None, "a", "c"] * 20)
t("str x with None", lambda: nns.nns_reg(pd.DataFrame({"g": xc, "z": np.arange(100.0)}), np.arange(100.0))["RPM"].keys())
xc2 = pd.read_csv(__import__("io").StringIO("g\na\nb\n\na\n" * 25))["g"]
t("str x with NaN (csv blank)", lambda: list(nns.nns_reg(pd.DataFrame({"g": xc2, "z": np.arange(100.0)}), np.arange(100.0))["RPM"].keys()))
# 7. bool x
xb = np.arange(100) % 3 == 0
t("bool x univariate", lambda: (lambda r: (type(r).__name__, list(r.keys())[:4], r.get("regression.points", {}).get("x")))(nns.nns_reg(xb, np.arange(100.0))))
# 8. numeric strings
t("numeric-string x", lambda: list(nns.nns_reg(pd.DataFrame({"g": np.array(["1", "2", "10"] * 10, dtype=object), "z": np.arange(30.0)}), np.arange(30.0))["RPM"].keys()))
# 9. point_est DataFrame with permuted columns
mt = pd.read_csv("/home/user/audit/data/mtcars.csv")
Xm = mt[["wt", "hp"]]; pe_ok = Xm.iloc[[0, 1]]; pe_perm = pe_ok[["hp", "wt"]]
t("pe permuted cols", lambda: (nns.nns_reg(Xm, mt.mpg, point_est=pe_ok)["Point.est"], nns.nns_reg(Xm, mt.mpg, point_est=pe_perm)["Point.est"]))
# 10. pandas Categorical with custom order, factor_2_dummy=False
cyl = pd.Categorical(mt.cyl.map({4: "four", 6: "six", 8: "eight"}), categories=["four", "six", "eight"], ordered=True)
dfc = pd.DataFrame({"wt": mt.wt, "cyl": cyl})
t("categorical codes", lambda: np.unique(nns.nns_reg(dfc, mt.mpg, factor_2_dummy=False)["Fitted.xy"]["cyl"][[0, 2, 4]]))
# 11. column names with spaces
t("names with spaces", lambda: list(nns.nns_reg(pd.DataFrame({"my col": mt.wt, "2nd": mt.hp}), mt.mpg)["RPM"].keys()))
# 12. pandas Series x/y
t("Series in", lambda: nns.nns_reg(mt.wt, mt.mpg, point_est=pd.Series([3.0]))["Point.est"])
# 13. NaN in x
t("NaN x", lambda: nns.nns_reg(np.array([1, 2, np.nan, 4.0] * 5), np.arange(20.0)))
# 14. y length 1
t("len1", lambda: nns.nns_reg(np.array([1.0]), np.array([2.0])))
# 15. y 2D column
t("y 2D col", lambda: nns.nns_reg(mt.wt.values, mt.mpg.values.reshape(-1, 1))["R2"])
# 16. order float 2.0, order=0
t("order 2.0", lambda: nns.nns_reg(mt.wt.values, mt.mpg.values, order=2.0)["R2"])
t("order 0", lambda: nns.nns_reg(mt.wt.values, mt.mpg.values, order=0))
# 17. point_est 2D for univariate
t("pe 2D univariate", lambda: nns.nns_reg(mt.wt.values, mt.mpg.values, point_est=np.array([[2.0], [3.0]]))["Point.est"])
# 18. confidence_interval 95 (percent) error
t("ci=95", lambda: nns.nns_reg(mt.wt.values, mt.mpg.values, confidence_interval=95))
# 19. smooth with class
t("type xonly multivar", lambda: nns.nns_reg(Xm, mt.mpg, type="XONLY"))
# 20. noise_reduction mode_class (accepted in docs?)
t("nr mode_class", lambda: nns.nns_reg(mt.wt.values, mt.mpg.values, noise_reduction="mode_class"))
# 21. integer numpy y with 2 values not 0/1 -> regression
t("y in {1,2}", lambda: nns.nns_reg(mt.wt.values, (mt.am.values + 1))["class.levels"])
# 22. tau numeric for NNS.caus
t("tau=1 numeric", lambda: nns.nns_reg(Xm, mt.mpg, dim_red_method="NNS.caus", tau=1)["equation"])
