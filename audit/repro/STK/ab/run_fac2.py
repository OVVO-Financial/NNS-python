from compare import *
rs=np.random.RandomState(3); n=240
g=rs.choice(["zeta","alpha","mid"], n); g[0]="zeta"; g[1]="mid"
eff={"zeta":2.0,"alpha":-1.0,"mid":0.5}
x1=rs.normal(size=n); x2=rs.uniform(size=n)
y=np.array([eff[v] for v in g])+0.5*x1+0.2*rs.normal(size=n)
X=pd.DataFrame({"x1":x1,"g":g,"x2":x2}); tr=np.arange(200); te=np.arange(200,240)
for nm,ra,pa in [("fac3_m12",{},{}),("fac3_m1",{"method":1},{"method":1}),("fac3_m2",{"method":2},{"method":2})]:
    R,P=case(nm,X.iloc[tr],y[tr],X.iloc[te],ra,pa,factor_cols=[2]); report(nm,R,P,keys=("OBJfn.reg","NNS.reg.n.best","OBJfn.dim.red","weights","reg","dim.red","stack"))
    print("   R coef", R.get("NNS.dim.red.coefficients"), "\n   PY coef", P.get("NNS.dim.red.coefficients"))
