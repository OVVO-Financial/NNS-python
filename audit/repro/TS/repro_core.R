# Compact R repros for TS findings (run: Rscript repro_core.R)
suppressPackageStartupMessages(library(NNS))
ap <- as.numeric(AirPassengers); set.seed(1)
cat("TS-means: method='means' ->", NNS.ARMA(ap, h=4, seasonal.factor=12, method="means", plot=FALSE), "\n")
t <- 1:240; s <- 3*sin(2*pi*t/12) + rnorm(240, 0, .5)
cat("TS-nonlin-clip: last 12 obs", round(tail(s,12),2), "\n  nonlin fcst (negative.values=TRUE):", round(NNS.ARMA(s, h=12, seasonal.factor=12, negative.values=TRUE, plot=FALSE),2), "\n")
cat("TS-weights: w=c(.5,.5) ->", round(NNS.ARMA(ap, h=3, seasonal.factor=c(12,24), method="lin", weights=c(.5,.5), plot=FALSE),1),
    " w=c(1,1) ->", round(NNS.ARMA(ap, h=3, seasonal.factor=c(12,24), method="lin", weights=c(1,1), plot=FALSE),1), "\n")
cat("TS-bpNULL: sf=TRUE,best.periods=NULL ->", round(NNS.ARMA(ap, h=3, seasonal.factor=TRUE, best.periods=NULL, plot=FALSE),3),
    " vs sf=1 ->", round(NNS.ARMA(ap, h=3, seasonal.factor=1, plot=FALSE),3), " best period:", NNS.seas(ap, plot=FALSE)$best.period, "\n")
wn <- read.csv("/home/user/audit/work/TS/data/synth.csv")$wn
cat("TS-zero: white noise mean 10, modulo=7 ->", NNS.ARMA(wn, h=4, modulo=7, plot=FALSE), "\n")
cat("TS-const: constant 5 ->", NNS.ARMA(rep(5,60), h=3, plot=FALSE), "| sf=12 ->", NNS.ARMA(rep(5,60), h=3, seasonal.factor=12, plot=FALSE), "\n")
cat("TS-h0: h=0 returns length", length(NNS.ARMA(ap, h=0, seasonal.factor=12, plot=FALSE)), "\n")
a <- NNS.ARMA.optim(ap[1:132], h=12, seasonal.factor=c(6,12,18,24), linear.approximation=FALSE, print.trace=FALSE, ncores=1)
b <- NNS.ARMA.optim(ap[1:132], h=12, seasonal.factor=c(6,12,18,24), lin.only=TRUE, print.trace=FALSE, ncores=1)
cat("TS-noapprox: linear.approximation=FALSE ->", a$method, a$periods, a$obj.fn, " | lin.only ->", b$method, b$periods, b$obj.fn, "\n")
