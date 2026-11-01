from compare import *
rs=np.random.RandomState(8); n=60
y=np.array(["A"]*29+["B"]*29+["C"]*2); x1=np.r_[rs.normal(0,1,29), rs.normal(3,1,29), [6.0,6.2]]; x2=rs.normal(size=n)
X=pd.DataFrame({"x1":x1,"x2":x2}); perm=rs.permutation(n); X=X.iloc[perm].reset_index(drop=True); y=y[perm]
te=pd.DataFrame({"x1":[0.0,3.0,6.1],"x2":[0,0,0]})
R,P=case("bal_singleton", X, y, te, {"type":"CLASS","balance":True,"method":1}, {"type":"CLASS","balance":True,"method":1}, yfactor=True)
report("balance with a 2-member class (fold has 1 member)", R, P, keys=("OBJfn.reg","NNS.reg.n.best","probability.threshold","stack"))
