suppressPackageStartupMessages(library(NNS)); a <- commandArgs(TRUE); n <- a[1]; p <- a[2]
X <- read.csv(sprintf("X_%s_%s.csv",n,p)); y <- read.csv(sprintf("y_%s_%s.csv",n,p))[[1]]
m <- floor(0.8*nrow(X)); tr <- 1:m; te <- (m+1):nrow(X)
t0 <- proc.time()[["elapsed"]]; r <- NNS.stack(X[tr,], y[tr], X[te,], status=FALSE, ncores=1); el <- proc.time()[["elapsed"]]-t0
writeLines(sprintf("%.17g", r$stack), sprintf("r_stack_%s_%s.txt",n,p))
cat(sprintf("R n=%s p=%s stack secs=%.2f nbest=%s w=%.2f\n", n, p, el, r$NNS.reg.n.best, r$weights[1]))
if (length(a)>2) { t0 <- proc.time()[["elapsed"]]; b <- NNS.boost(X[tr,], y[tr], X[te,], epochs=50, learner.trials=50, status=FALSE, feature.importance=FALSE); cat(sprintf("R boost(50/50) secs=%.2f\n", proc.time()[["elapsed"]]-t0)); writeLines(sprintf("%.17g", b$results), sprintf("r_boost_%s_%s.txt",n,p)) }
