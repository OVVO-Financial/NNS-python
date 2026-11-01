from boostcmp import *
df = pd.read_csv("../boston/boston_full.csv"); btr = np.loadtxt("../boston/train_idx.txt", dtype=int); bte=np.setdiff1d(np.arange(len(df)), btr)
X=df.iloc[:, [0,4,5,7,10,12]]; y=df.medv.to_numpy()
for nm,ra,pa in [("b_thr15",{"threshold":1.5},{"threshold":1.5}),("b_thrneg",{"threshold":-0.2},{"threshold":-0.2}),("b_extreme",{"extreme":True},{"extreme":True}),
                 ("b_ts",{"ts.test":40},{"ts_test":40}),("b_depthmax",{"depth":"max"},{"depth":"max"}),("b_L1",{"dist":"L1"},{"dist":"L1"}),("b_ep0",{"epochs":0},{"epochs":0}),
                 ("b_fonly",{"features.only":True},{"features_only":True}),("b_seed9",{"seed":9},{"seed":9})]:
    ra=dict(ra, epochs=ra.get("epochs",15), **{"learner.trials":15}); pa=dict(pa, epochs=pa.get("epochs",15), learner_trials=15)
    if nm=="b_fonly":
        R,P=bcase(nm, X.iloc[btr], y[btr], X.iloc[bte], ra, pa) if False else (None,None)
        continue
    bcase(nm, X.iloc[btr], y[btr], X.iloc[bte], ra, pa)
