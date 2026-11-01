suppressPackageStartupMessages(library(NNS))
for (f in c("vix_d1000","vix_d3000")) { x <- read.csv(paste0("data/",f,".csv"))$y
 t <- system.time(r <- NNS.ARMA.optim(x, h=20, seasonal.factor=1:24, print.trace=FALSE, ncores=1))[3]
 cat(f, "optim R", t, "s periods", r$periods, r$method, r$obj.fn, "\n")
 t <- system.time(a <- NNS.ARMA(x, h=50, seasonal.factor=c(5,21), method="nonlin", plot=FALSE))[3]; cat(f,"ARMA nonlin h=50 R", t, "s", head(a,3), "\n") }
