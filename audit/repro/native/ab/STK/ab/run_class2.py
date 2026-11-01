from compare import *
iris=pd.read_csv("/home/user/audit/data/iris.csv"); rs=np.random.RandomState(4); p=rs.permutation(150)
X=iris.iloc[p,:4].reset_index(drop=True); y=iris.Species.to_numpy()[p]
R,P=case("iris_cv03", X.iloc[:130], y[:130], X.iloc[130:], {"type":"CLASS","CV.size":0.3,"folds":3}, {"type":"CLASS","cv_size":0.3,"folds":3}, yfactor=True); report("iris CV.size=.3 folds=3",R,P,keys=("OBJfn.reg","NNS.reg.n.best","OBJfn.dim.red","probability.threshold","weights","stack"))
R,P=case("iris_ts", X.iloc[:130], y[:130], X.iloc[130:], {"type":"CLASS","ts.test":20,"folds":3}, {"type":"CLASS","ts_test":20,"folds":3}, yfactor=True); report("iris ts.test=20",R,P,keys=("OBJfn.reg","NNS.reg.n.best","OBJfn.dim.red","probability.threshold","weights","stack"))
R,P=case("iris_m1nostack", X.iloc[:130], y[:130], X.iloc[130:], {"type":"CLASS","method":1}, {"type":"CLASS","method":1}, yfactor=True); report("iris method1",R,P,keys=("OBJfn.reg","NNS.reg.n.best","probability.threshold","stack"))
