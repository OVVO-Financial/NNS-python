suppressPackageStartupMessages(library(NNS))
x <- read.csv("data/ap.csv")$y
set.seed(1)
f <- function(...) tryCatch(NNS.ARMA(x, plot=FALSE, seasonal.plot=FALSE, ...), error=function(e) paste("ERR", conditionMessage(e)))
print(f(h=1, seasonal.factor=12, method="nonlin", pred.int=.95))
print(f(h=12, seasonal.factor=12, method="lin", pred.int=.95))
print(f(h=12, seasonal.factor=c(12,24), method="lin", pred.int=.95))
print(f(h=12, seasonal.factor=12, method="nonlin", pred.int=.95))
print(f(h=12, seasonal.factor=12, method="means", pred.int=.95))
