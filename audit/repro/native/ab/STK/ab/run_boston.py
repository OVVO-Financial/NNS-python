from compare import *
df = pd.read_csv("../boston/boston_full.csv"); tr = np.loadtxt("../boston/train_idx.txt", dtype=int)
te = np.setdiff1d(np.arange(len(df)), tr)
X = df.iloc[:, :13]; y = df["medv"].to_numpy(float)
for nm, ra, pa in [("m1", {"method":1}, {"method":1}),
                   ("m2", {"method":2}, {"method":2}),
                   ("m12_nostack", {"stack":False}, {"stack":False}),
                   ("m12_pi", {"pred.int":0.9}, {"pred_int":0.9}),
                   ("m12_dep", {"dim.red.method":"NNS.dep"}, {"dim_red_method":"NNS.dep"}),
                   ("m12_caus", {"dim.red.method":"NNS.caus"}, {"dim_red_method":"NNS.caus"}),
                   ("m12_equal", {"dim.red.method":"equal"}, {"dim_red_method":"equal"}),
                   ("m12_all", {"dim.red.method":"all"}, {"dim_red_method":"all"}),
                   ("m12_folds3", {"folds":3}, {"folds":3}),
                   ("m12_folds1", {"folds":1}, {"folds":1}),
                   ("m12_seed7", {"seed":7}, {"seed":7}),
                   ("m12_order1", {"order":1}, {"order":1}),
                   ("m12_ordermax", {"order":"max"}, {"order":"max"}),
                   ("m12_L1", {"dist":"L1"}, {"dist":"L1"}),
                   ]:
    R,P = case("boston_"+nm, X.iloc[tr], y[tr], X.iloc[te], ra, pa, obj="rmse")
    report(nm, R, P)
