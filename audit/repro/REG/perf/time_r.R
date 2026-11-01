suppressPackageStartupMessages(library(NNS))
tm <- function(expr, reps = 3) { e <- substitute(expr); pf <- parent.frame(); t <- sapply(seq_len(reps), function(i) system.time(eval(e, pf))[["elapsed"]]); min(t) }
pe <- seq(-3.5, 3.5, length.out = 100)
for (n in c(1000, 10000, 100000)) {
  d <- read.csv(sprintf("uni_%d.csv", n))
  cat(sprintf("uni n=%d default: %.3fs\n", n, tm(NNS.reg(d$x, d$y, plot = FALSE, point.est = pe, ncores = 1))))
  cat(sprintf("uni n=%d smooth: %.3fs\n", n, tm(NNS.reg(d$x, d$y, plot = FALSE, point.est = pe, smooth = TRUE, ncores = 1))))
  if (n <= 10000) cat(sprintf("uni n=%d order=max: %.3fs\n", n, tm(NNS.reg(d$x, d$y, plot = FALSE, point.est = pe, order = "max", ncores = 1))))
  cat(sprintf("uni n=%d ci: %.3fs\n", n, tm(NNS.reg(d$x, d$y, plot = FALSE, point.est = pe, confidence.interval = .95, ncores = 1))))
}
d <- read.csv("mv_10000.csv"); X <- as.matrix(d[, 1:5]); pe5 <- X[1:200, ]
cat(sprintf("mv n=1e4 p=5 default: %.3fs\n", tm(NNS.reg(X, d$y, plot = FALSE, point.est = pe5, ncores = 1), 2)))
cat(sprintf("mv n=1e4 p=5 n.best=1: %.3fs\n", tm(NNS.reg(X, d$y, plot = FALSE, point.est = pe5, n.best = 1, ncores = 1), 2)))
cat(sprintf("mv n=1e4 p=5 order=max: %.3fs\n", tm(NNS.reg(X, d$y, plot = FALSE, point.est = pe5, order = "max", ncores = 1), 2)))
cat(sprintf("mv n=1e4 p=5 dimred cor: %.3fs\n", tm(NNS.reg(X, d$y, plot = FALSE, point.est = pe5, dim.red.method = "cor", ncores = 1), 2)))
cat(sprintf("mv n=1e4 p=5 dimred NNS.dep: %.3fs\n", tm(NNS.reg(X, d$y, plot = FALSE, point.est = pe5, dim.red.method = "NNS.dep", ncores = 1), 2)))
cat(sprintf("mv n=1e4 p=5 dimred all: %.3fs\n", tm(NNS.reg(X, d$y, plot = FALSE, point.est = pe5, dim.red.method = "all", ncores = 1), 2)))
r <- NNS.reg(X, d$y, plot = FALSE, point.est = pe5, ncores = 1); cat("mv R2", r$R2, "n.best", r$n.best, "RPM rows", nrow(r$RPM), "\n")
