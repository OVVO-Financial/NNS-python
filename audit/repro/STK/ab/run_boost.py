from boostcmp import *
iris=pd.read_csv("/home/user/audit/data/iris.csv"); print(iris.columns.tolist())
te=np.arange(140,150); tr=np.arange(140)
X=iris.iloc[:,:4]; y=iris.iloc[:,4].to_numpy()
bcase("boost_iris", X.iloc[tr], y[tr], X.iloc[te], {"epochs":10,"learner.trials":10,"balance":True,"type":"CLASS"}, {"epochs":10,"learner_trials":10,"balance":True,"type":"CLASS"}, yfactor=True)
bcase("boost_iris_nobal", X.iloc[tr], y[tr], X.iloc[te], {"epochs":20,"learner.trials":15,"type":"CLASS"}, {"epochs":20,"learner_trials":15,"type":"CLASS"}, yfactor=True)
df = pd.read_csv("../boston/boston_full.csv"); btr = np.loadtxt("../boston/train_idx.txt", dtype=int); bte=np.setdiff1d(np.arange(len(df)), btr)
bcase("boost_boston", df.iloc[btr,:13], df.medv.to_numpy()[btr], df.iloc[bte,:13], {"epochs":30,"learner.trials":30}, {"epochs":30,"learner_trials":30})
bcase("boost_boston_pi", df.iloc[btr,:13], df.medv.to_numpy()[btr], df.iloc[bte,:13], {"epochs":20,"learner.trials":20,"pred.int":0.9,"CV.size":0.25}, {"epochs":20,"learner_trials":20,"pred_int":0.9,"cv_size":0.25}, obj="rmse")
bcase("boost_boston_fonly", df.iloc[btr,:13], df.medv.to_numpy()[btr], None, {"epochs":20,"learner.trials":20,"features.only":True}, {"epochs":20,"learner_trials":20,"features_only":True}) if False else None
mt=pd.read_csv("/home/user/audit/data/mtcars.csv"); print(mt.columns.tolist())
