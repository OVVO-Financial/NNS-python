suppressPackageStartupMessages({library(NNS); library(jsonlite)})
X <- as.matrix(read.csv("/home/user/audit/work/TS/data/panel_nan.csv"))
r <- NNS.VAR(X, h=3, tau=2, status=FALSE, ncores=1)
cat(toJSON(list(interp=as.matrix(r$interpolated_and_extrapolated), uni=as.matrix(r$univariate), multi=as.matrix(r$multivariate), ens=as.matrix(r$ensemble)), digits=NA))
