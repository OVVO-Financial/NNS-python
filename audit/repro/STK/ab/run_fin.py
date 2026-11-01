from compare import *
r=pd.read_csv("/home/user/audit/data/returns_daily_equity_etf.csv", index_col=0)
feat=r[["SPY","QQQ","IWM","TLT","GLD","USO"]].shift(1).add_suffix("_l1")
feat["SPY_l2"]=r.SPY.shift(2); feat["SPY_5d"]=r.SPY.rolling(5).sum().shift(1)
df=pd.concat([feat, r.SPY.rename("target")],axis=1).dropna().iloc[-1000:]
X=df.drop(columns="target"); yret=df.target.to_numpy(); ysign=(yret>0).astype(int)
tr=np.arange(900); te=np.arange(900,1000)
R,P=case("fin_sign", X.iloc[tr], ysign[tr], X.iloc[te], {}, {}); report("SPY next-day sign (auto-class 0/1)",R,P,keys=("OBJfn.reg","NNS.reg.n.best","OBJfn.dim.red","probability.threshold","weights","stack","secs"))
if "stack" in R and "stack" in P:
    for s,o in (("R",R),("PY",P)):
        pred=np.asarray(o["stack"],float)-1; print(f"   {s} test hit-rate {np.mean(pred==ysign[te]):.3f}, frac up predicted {pred.mean():.2f}; base rate up {ysign[te].mean():.2f}")
R,P=case("fin_sign_ts", X.iloc[tr], ysign[tr], X.iloc[te], {"ts.test":100,"folds":3}, {"ts_test":100,"folds":3}); report("SPY sign ts.test=100 folds=3",R,P,keys=("OBJfn.reg","NNS.reg.n.best","OBJfn.dim.red","probability.threshold","weights","stack","secs"))
R,P=case("fin_ret", X.iloc[tr], yret[tr], X.iloc[te], {}, {}, obj="mae"); report("SPY next-day return (MAE)",R,P,keys=("OBJfn.reg","NNS.reg.n.best","OBJfn.dim.red","weights","stack","secs"))
if "stack" in R and "stack" in P:
    for s,o in (("R",R),("PY",P)):
        pred=np.asarray(o["stack"],float); print(f"   {s} sign hit {np.mean(np.sign(pred)==np.sign(yret[te])):.3f}  corr {np.corrcoef(pred,yret[te])[0,1]:.3f}")
