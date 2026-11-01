from compare import *
rs=np.random.RandomState(5); n=200
X=pd.DataFrame({"a":rs.normal(size=n),"b":rs.normal(size=n),"c":rs.normal(size=n)}); y=rs.normal(size=n)
tr=np.arange(160); te=np.arange(160,200)
for nm,cols,ra,pa in [("noise_uni",["a"],{},{}),("noise_multi_m1",["a","b","c"],{"method":1},{"method":1}),("noise_multi",["a","b","c"],{},{})]:
    R,P=case(nm,X.iloc[tr][cols],y[tr],X.iloc[te][cols],ra,pa); report(nm,R,P,keys=("OBJfn.reg","NNS.reg.n.best","weights","reg","stack"))
    reg=np.asarray(R["reg"],float); print("   R reg sd:", reg.std(), " mean(ytrain)=", y[tr].mean(), " R reg[:3]", reg[:3])
