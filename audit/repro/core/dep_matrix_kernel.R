suppressMessages(library(NNS))
r <- read.csv("/home/user/audit/data/returns_daily_equity_etf.csv")
X <- as.matrix(r[1:1000, c("SPY","QQQ","TLT","GLD")])
a <- NNS.dep(X)$Dependence
b <- NNS:::NNS_dep_matrix_cpp(X)$Dependence
print(round(a,4)); print(round(b,4))
cat("max abs diff Dependence:", max(abs(a-b)), "\n")
# independent discrete data: tie handling floor
set.seed(1); x <- sample(0:2, 5000, TRUE); y <- sample(0:2, 5000, TRUE)
print(NNS.dep(x, y))
set.seed(2); x <- rnorm(5000); y <- rnorm(5000)
print(NNS.dep(x, y))
