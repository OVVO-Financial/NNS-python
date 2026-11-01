from compare import *
iris=pd.read_csv("/home/user/audit/data/iris.csv")
X=iris.iloc[:,:4]; y=iris.Species.to_numpy()
te=np.arange(140,150); tr=np.arange(140)
R,P=case("iris_class", X.iloc[tr], y[tr], X.iloc[te], {"type":"CLASS"}, {"type":"CLASS"}, yfactor=True); report("iris CLASS", R,P); print("  levels", R.get("class.levels"), P.get("class.levels"))
R,P=case("iris_class_bal", X.iloc[tr], y[tr], X.iloc[te], {"type":"CLASS","balance":True}, {"type":"CLASS","balance":True}, yfactor=True); report("iris CLASS balance", R,P)
R,P=case("iris_class_nothr", X.iloc[tr], y[tr], X.iloc[te], {"type":"CLASS","optimize.threshold":False}, {"type":"CLASS","optimize_threshold":False}, yfactor=True); report("iris CLASS no thr opt", R,P)
R,P=case("iris_class_pi", X.iloc[tr], y[tr], X.iloc[te], {"type":"CLASS","pred.int":0.95}, {"type":"CLASS","pred_int":0.95}, yfactor=True); report("iris CLASS pi", R,P)
# shuffled so that virginica appears first
rs=np.random.RandomState(1); perm=rs.permutation(150); perm=np.r_[perm[y[perm]=="virginica"][:1], perm[y[perm]!="virginica"][:], perm[y[perm]=="virginica"][1:]]
Xs=X.iloc[perm].reset_index(drop=True); ys=y[perm]
R,P=case("iris_shuf", Xs.iloc[:135], ys[:135], Xs.iloc[135:], {"type":"CLASS"}, {"type":"CLASS"}, yfactor=True); report("iris shuffled (virginica first) CLASS", R,P); print("  levels R", R.get("class.levels"), "PY", P.get("class.levels"))
# binary numeric 0/1
yb=(y=="virginica").astype(int)
R,P=case("iris_bin01", X.iloc[tr], yb[tr], X.iloc[te], {}, {}); report("iris binary 0/1 auto", R,P); print("  levels", R.get("class.levels"), P.get("class.levels"))
