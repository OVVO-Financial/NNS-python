from compare import *
rs=np.random.RandomState(42); n=300
X=pd.DataFrame({"a":rs.uniform(-2,2,n),"b":rs.uniform(-2,2,n),"c":rs.normal(size=n),"d":rs.normal(size=n)})
y=np.sin(2*X.a)+X.b**2+0.1*rs.normal(size=n)
tr=np.arange(240); te=np.arange(240,300)
for nm,ra,pa,obj in [("sim_default",{},{},None),("sim_ts",{"ts.test":30},{"ts_test":30},None),("sim_ts_f2",{"ts.test":50,"folds":2},{"ts_test":50,"folds":2},None),
    ("sim_cv1",{"folds":1,"CV.size":0.3},{"folds":1,"cv_size":0.3},None),("sim_max",{"objective":"max"},{"objective":"max"},"negsse"),
    ("sim_mae",{},{},"mae"),("sim_numcoef",{"dim.red.method":[1,1,0,0]},{"dim_red_method":[1,1,0,0]},None),
    ("sim_m2pi",{"method":2,"pred.int":0.8},{"method":2,"pred_int":0.8},None),("sim_m1pi",{"method":1,"pred.int":0.8},{"method":1,"pred_int":0.8},None),
    ("sim_seedNULL_skip",{"seed":11},{"seed":11},None)]:
    R,P=case(nm,X.iloc[tr],y[tr],X.iloc[te],ra,pa,obj=obj); report(nm,R,P)
# univariate predictor
R,P=case("sim_uni",X.iloc[tr,[0]],y[tr],X.iloc[te,[0]],{},{}); report("sim_uni (1 predictor, method2 dropped)",R,P)
