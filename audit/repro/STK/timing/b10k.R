suppressPackageStartupMessages(library(NNS)); X <- read.csv("X_10000_5.csv"); y <- read.csv("y_10000_5.csv")[[1]]; tr <- 1:8000; te <- 8001:10000
t0 <- proc.time()[["elapsed"]]; b <- NNS.boost(X[tr,], y[tr], X[te,], epochs=20, learner.trials=20, status=FALSE, feature.importance=FALSE); cat(sprintf("R boost n=10000 p=5 (20/20) secs=%.2f\n", proc.time()[["elapsed"]]-t0))
writeLines(sprintf("%.17g", b$results), "r_b10k.txt")
