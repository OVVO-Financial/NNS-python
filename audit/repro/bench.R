suppressPackageStartupMessages(library(NNS)); RcppParallel::setThreadOptions(1)
set.seed(42)
R <- read.csv("/home/user/audit/data/returns_daily_equity_etf.csv", row.names = 1)
bos <- read.csv("/home/user/audit/data/boston.csv"); ap <- read.csv("/home/user/audit/data/airpassengers.csv")$y
X5 <- matrix(rnorm(5e4), ncol = 5); y5 <- rowSums(sin(X5)) + rnorm(1e4, sd = .1)
x1 <- rnorm(1e4); y1 <- sin(x1) + rnorm(1e4, sd = .1); M <- matrix(rnorm(8000), ncol = 2)
tm <- function(expr) { e <- substitute(expr); min(sapply(1:3, function(i) system.time(eval(e, parent.frame(3)))[["elapsed"]])) }
res <- list(
 "lpm 1000 targets x 1e5" = tm(LPM(2, seq(-.05, .05, length = 1000), rt(1e5, 3) * .01)),
 "nns_dep SPY/TLT daily (n=4.2k)" = tm(NNS.dep(R$SPY, R$TLT)),
 "nns_dep n=1e5" = tm(NNS.dep(rep(c(R$SPY, R$QQQ)[1:8000], each = 12), rep(c(R$TLT, R$SPY)[1:8000], each = 12))),
 "nns_cdf univariate n=1e4 degree 1.5" = tm(NNS.CDF(x1, 1.5, plot = FALSE)),
 "nns_cdf bivariate n=4000 degree 1" = tm(NNS.CDF(M, 1, plot = FALSE)),
 "nns_reg univariate n=1e4" = tm(NNS.reg(x1, y1, plot = FALSE, residual.plot = FALSE)),
 "nns_reg multivariate n=1e4 p=5" = tm(NNS.reg(X5, y5, plot = FALSE, residual.plot = FALSE)),
 "nns_reg order='max' n=1e4 p=5" = tm(NNS.reg(X5, y5, order = "max", plot = FALSE, residual.plot = FALSE)),
 "nns_stack Boston (5-fold)" = tm(NNS.stack(bos[1:400, -14], bos$medv[1:400], IVs.test = bos[401:506, -14], status = FALSE, ncores = 1)),
 "fsd_uni continuous n=1e4" = tm(NNS.FSD.uni(x1, rnorm(1e4, .05), type = "continuous")),
 "sd_efficient_set 11 assets daily" = tm(NNS.SD.efficient.set(as.matrix(R), degree = 1, status = FALSE)),
 "nns_arma_optim AirPassengers" = tm(NNS.ARMA.optim(ap[1:132], h = 12, seasonal.factor = c(12, 6, 24), print.trace = FALSE, plot = FALSE)),
 "nns_seas n=1e4" = tm(NNS.seas(sin((1:1e4) * 2 * pi / 12) + rnorm(1e4), plot = FALSE)),
 "causal_matrix 10 cols n=1000" = tm(NNS.caus(matrix(rnorm(1e4), ncol = 10)))
)
for (n in names(res)) cat(sprintf("%-40s %8.3f s\n", n, res[[n]]))
jsonlite::write_json(res, "/home/user/audit/work/bench/r.json", auto_unbox = TRUE, digits = NA)
