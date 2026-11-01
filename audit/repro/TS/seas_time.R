suppressPackageStartupMessages(library(NNS))
set.seed(1)
for (n in c(1000, 10000, 100000)) { x <- 100 + sin(2*pi*(1:n)/12) + rnorm(n); t <- system.time(r <- NNS.seas(x, plot=FALSE))[3]; cat("R seas n=", n, " ", t, "s best", r$best.period, "\n") }
