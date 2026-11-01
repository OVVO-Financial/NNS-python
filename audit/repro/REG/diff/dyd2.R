suppressPackageStartupMessages({library(NNS)})
set.seed(5); n <- 500; x <- cbind(runif(n,-2,2), runif(n,-2,2)); y <- 2*x[,1]
ep <- rbind(c(-1, 0), c(0, 1), c(1, -1))
r1 <- dy.d_(x, y, wrt = 1, eval.points = ep, messages = FALSE); r2 <- dy.d_(x, y, wrt = 2, eval.points = ep, messages = FALSE)
cat("y=2*x1: dy/dx1 (truth 2):", sprintf("%.4f", r1["First",][[1]]), "| dy/dx2 (truth 0):", sprintf("%.4f", r2["First",][[1]]), "\n")
