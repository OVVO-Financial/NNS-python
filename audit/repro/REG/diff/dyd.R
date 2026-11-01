suppressPackageStartupMessages({library(NNS)})
d <- read.csv("dyd.csv"); x <- as.matrix(d[,1:2]); y <- d$y
ep <- rbind(c(-1, 0), c(0, 0), c(1, 0.5), c(1.5, -1))
t0 <- proc.time()[[3]]
r1 <- dy.d_(x, y, wrt = 1, eval.points = ep, messages = FALSE)
r2 <- dy.d_(x, y, wrt = 2, eval.points = ep, messages = FALSE)
cat("time", proc.time()[[3]] - t0, "\n")
cat("R wrt1 First ", sprintf("%.5f", r1["First",][[1]]), "\n"); cat("R wrt1 Second", sprintf("%.5f", r1["Second",][[1]]), "\n")
cat("R wrt2 First ", sprintf("%.5f", r2["First",][[1]]), "\n")
r3 <- dy.d_(x, y, wrt = 1, eval.points = c(-1, 0, 1), messages = FALSE)
cat("R apd-vec wrt1 First", sprintf("%.5f", r3["First",][[1]]), "\n")
