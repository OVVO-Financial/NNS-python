suppressPackageStartupMessages(library(NNS))
x <- seq(0, 2*pi, pi/100); y <- sin(x)
writeLines(c("x,y", sprintf("%.17g,%.17g", x, y)), "/home/user/audit/work/REG/repro/sinxy.csv")
m <- cbind(NNS.rescale(x,0,1), NNS.rescale(x,0,1), NNS.rescale(y,0,1))
cat("R: colMeans", sprintf("%.17g", colMeans(m)), " NNS.dep", NNS.dep(x,y,asym=TRUE)$Dependence, " NNS.copula", NNS.copula(m),
    " .nns_reg_dependence", NNS:::.nns_reg_dependence(x,y), " default order", NNS:::.nns_reg_default_order(x,y), "\n")
