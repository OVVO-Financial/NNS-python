suppressMessages(library(NNS))
set.seed(11); hit <- 0
for (i in 1:20000) {
  X <- matrix(sample(1:5, 30, TRUE), ncol = 3)
  pm <- PM.matrix(0, 0, colMeans(X), X, pop_adj = FALSE); u <- upper.tri(pm$cupm)
  s <- sum(pm$cupm[u]) + sum(pm$clpm[u])
  if (s == 1) { hit <- hit + 1; if (hit == 1) { cat("found d=3 X with discrete_Co_pm == 1 (independence level is 1.5):\n"); print(t(X)); cat("NNS.copula =", NNS.copula(X), "  NNS.copula(X+1e-9 jitter on one cell) =", {Y <- X; Y[1,1] <- Y[1,1] + 1e-9; NNS.copula(Y)}, "\n"); write.csv(X, "copula_eq1_X.csv", row.names = FALSE) } }
}
cat("hits:", hit, "/ 20000\n")
