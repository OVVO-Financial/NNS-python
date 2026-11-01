suppressPackageStartupMessages(library(NNS))
x <- scan("uni/xtr.txt",quiet=TRUE); y <- scan("uni/ytr.txt",quiet=TRUE)
for (o in list(NULL,1,2,3,4,5,"max")) { r <- NNS.reg(x, y, order=o, plot=FALSE, residual.plot=FALSE, ncores=1); cat("order", if(is.null(o)) "NULL" else o, "rp", nrow(r$regression.points), "\n") }
cat("dep:", NNS.dep(x,y)$Dependence, " n unique x", length(unique(x)), "n", length(x), "\n")
