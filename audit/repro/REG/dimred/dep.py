import numpy as np, pandas as pd
from nns import nns_dep
from nns.causation import _uni_caus
b = pd.read_csv("/home/user/audit/data/boston.csv"); y = b.medv.values; x = b.drop(columns="medv").values
print("dep ", " ".join("%.10f" % nns_dep(x[:, j], y, asym=True)["Dependence"] for j in range(13)))
print("caus", " ".join("%.10f" % _uni_caus(y, x[:, j], 0) for j in range(13)))
