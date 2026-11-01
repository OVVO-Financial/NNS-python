suppressPackageStartupMessages(library(NNS))
P <- as.matrix(read.csv("/home/user/audit/work/TS/data/panel_monthly.csv"))[,1:3]
h <- 6; r <- NNS.VAR(P, h=h, tau=1, status=FALSE, ncores=1)
cat("R multivariate:\n"); print(round(as.matrix(r$multivariate),4)); cat("last h observed:\n"); print(round(tail(P,h),4))
