suppressMessages(library(NNS))
b <- read.csv("/home/user/audit/data/boston.csv"); y <- b$medv; X <- as.matrix(b[, setdiff(names(b),"medv")])
for (m in 1:2) { t <- system.time(r <- NNS.stack(X[1:400,], y[1:400], IVs.test=X[401:506,], method=m, folds=1, ncores=1))[["elapsed"]]; cat(sprintf("R NNS.stack method=%d: %.2fs\n", m, t)) }
