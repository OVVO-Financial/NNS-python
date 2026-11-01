suppressPackageStartupMessages(library(NNS))
x <- read.csv("data/ap.csv")$y
f <- function(l, e) { r <- tryCatch(e, error=function(err) paste("ERR", conditionMessage(err))); cat(l, ":", if(is.character(r)) r else round(head(as.numeric(r),4),3), "\n") }
f("sfTRUE bpNULL", NNS.ARMA(x, h=4, seasonal.factor=TRUE, best.periods=NULL, plot=FALSE))
f("sf=1 nonlin", NNS.ARMA(x, h=4, seasonal.factor=1, plot=FALSE))
f("sfFALSE bpNULL", NNS.ARMA(x, h=4, seasonal.factor=FALSE, best.periods=NULL, plot=FALSE))
f("h0", NNS.ARMA(x, h=0, seasonal.factor=12, plot=FALSE))
f("ts>n", NNS.ARMA(x, h=4, training.set=200, seasonal.factor=12, plot=FALSE))
f("sf 12.5", NNS.ARMA(x, h=4, seasonal.factor=12.5, method="lin", plot=FALSE))
f("weights single 0.5 lin", NNS.ARMA(x, h=4, seasonal.factor=12, weights=0.5, method="lin", plot=FALSE))
f("weights single 0.5 nonlin", NNS.ARMA(x, h=4, seasonal.factor=12, weights=0.5, method="nonlin", plot=FALSE))
f("modulo no-match", NNS.ARMA(x, h=4, modulo=7, plot=FALSE))
f("wn mean10 default", NNS.ARMA(rnorm(100,10), h=4, plot=FALSE))
