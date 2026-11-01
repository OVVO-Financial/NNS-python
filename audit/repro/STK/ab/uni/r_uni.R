suppressPackageStartupMessages(library(NNS))
x <- scan("uni/xtr.txt",quiet=TRUE); y <- scan("uni/ytr.txt",quiet=TRUE); xt <- scan("uni/xte.txt",quiet=TRUE)
f <- NNS:::.nns_reg_univariate_fast(x, y, xt, order=NULL, noise.reduction="off", is.class=FALSE)$prediction
g <- NNS.reg(x, y, point.est=xt, plot=FALSE, residual.plot=FALSE, ncores=1, point.only=TRUE)$Point.est
g2 <- NNS.reg(x, y, point.est=xt, plot=FALSE, residual.plot=FALSE, ncores=1, point.only=FALSE)$Point.est
writeLines(sprintf("%.17g", f), "uni/r_fast.txt"); writeLines(sprintf("%.17g", g), "uni/r_reg.txt")
cat("fast vs reg(point.only) maxdiff", max(abs(f-g)), " reg point.only vs full", max(abs(g-g2)), "\n")
full <- NNS.reg(x, y, point.est=xt, plot=FALSE, residual.plot=FALSE, ncores=1)
print(nrow(full$regression.points)); write.csv(full$regression.points, "uni/r_rp.csv", row.names=FALSE)
