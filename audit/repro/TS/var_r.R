suppressPackageStartupMessages({library(NNS); library(jsonlite)})
P <- as.matrix(read.csv("data/panel_monthly.csv"))
args <- commandArgs(TRUE); k <- as.integer(args[1]); h <- as.integer(args[2]); tau <- as.integer(args[3]); drm <- args[4]
X <- P[, 1:k]
t0 <- Sys.time()
r <- NNS.VAR(X, h=h, tau=tau, dim.red.method=drm, status=FALSE, ncores=1)
el <- as.numeric(difftime(Sys.time(), t0, units="secs"))
out <- list(time=el, univariate=as.matrix(r$univariate), multivariate=as.matrix(r$multivariate), ensemble=as.matrix(r$ensemble),
  rv=lapply(as.data.frame(r$relevant_variables), function(z) as.character(na.omit(z))))
cat(toJSON(out, digits=NA, null="null"))
