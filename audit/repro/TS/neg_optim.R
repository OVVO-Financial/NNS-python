suppressPackageStartupMessages(library(NNS))
x <- scan("data/down.csv", quiet=TRUE)
r <- NNS.ARMA.optim(x, h=12, seasonal.factor=c(3,6,12,24), negative.values=TRUE, print.trace=FALSE, ncores=1)
cat("R periods", r$periods, r$method, r$obj.fn, "\nR results", round(r$results,3), "\n")
