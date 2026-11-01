"""NNS.CDF with NA (airquality$Ozone has 37 NA of 153): CDF never reaches 1 in R and in Python."""
import sys; sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, pandas as pd, nns
from rrun import rcall, arr
oz = pd.read_csv("/home/user/audit/data/airquality.csv")["Ozone"].to_numpy(dtype=float)
py = nns.nns_cdf(oz, 0)["Function"]["CDF"]; py1 = nns.nns_cdf(oz, 1)["Function"]["CDF"]
r = rcall("o <- NNS.CDF(airquality$Ozone, 0, plot=FALSE); o1 <- NNS.CDF(airquality$Ozone, 1, plot=FALSE); list(n=nrow(o$Function), last=tail(o$Function$CDF,1), last1=tail(o1$Function$CDF,1))")
print("python: rows", py.size, "max CDF deg0", py.max(), "deg1", py1.max())
print("R     :", r)
print("expected max CDF = 1; 116/153 =", 116 / 153)
try:
    nns.nns_cdf(oz, 0, target=30.0)
except Exception as e:
    print("python with target + NaN raises:", type(e).__name__, e)
