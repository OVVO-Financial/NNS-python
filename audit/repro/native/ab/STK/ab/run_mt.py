from compare import *
from boostcmp import bcase
mt=pd.read_csv("/home/user/audit/data/mtcars.csv").drop(columns="Unnamed: 0")
X=mt.drop(columns="mpg"); y=mt.mpg.to_numpy(); tr=np.arange(26); te=np.arange(26,32)
R,P=case("mt_num", X.iloc[tr], y[tr], X.iloc[te], {}, {}); report("mtcars mpg numeric",R,P,keys=("OBJfn.reg","NNS.reg.n.best","OBJfn.dim.red","weights","stack"))
Xf=X.copy(); Xf["cyl"]=Xf.cyl.astype(int).astype(str); Xf["gear"]=Xf.gear.astype(int).map({3:"three",4:"four",5:"five"})
R,P=case("mt_fac", Xf.iloc[tr], y[tr], Xf.iloc[te], {}, {}, factor_cols=[1,10]); report("mtcars factors cyl,gear",R,P,keys=("OBJfn.reg","NNS.reg.n.best","OBJfn.dim.red","weights","stack"))
R,P=case("mt_am_class", X.drop(columns="am").iloc[tr], mt.am.to_numpy()[tr], X.drop(columns="am").iloc[te], {}, {}); report("mtcars am (binary) auto-class",R,P,keys=("OBJfn.reg","NNS.reg.n.best","OBJfn.dim.red","probability.threshold","weights","stack"))
bcase("mt_boost", X.iloc[tr], y[tr], X.iloc[te], {"epochs":20,"learner.trials":20}, {"epochs":20,"learner_trials":20})
