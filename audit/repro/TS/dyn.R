suppressPackageStartupMessages(library(NNS))
x <- read.csv("data/ap.csv")$y
trace(NNS:::ARMA.seas.weighting, exit=quote(print(returnValue()$lag)), print=FALSE)
r <- NNS.ARMA(x, h=3, seasonal.factor=FALSE, best.periods=2, dynamic=TRUE, method="lin", plot=FALSE, seasonal.plot=FALSE)
print(r)
print(head(NNS.seas(x,plot=FALSE)$periods))
