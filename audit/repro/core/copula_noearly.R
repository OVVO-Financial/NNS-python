suppressMessages(library(NNS))
X <- as.matrix(read.csv("copula_eq1_X.csv")); n <- ncol(X); target <- colMeans(X)
d <- PM.matrix(0, 0, target, X, pop_adj = FALSE); u <- upper.tri(d$cupm)
dCo <- sum(d$cupm[u]) + sum(d$clpm[u])
c1 <- PM.matrix(1, 1, target, X, pop_adj = TRUE, norm = TRUE); cCo <- sum(c1$cupm[u]) + sum(c1$clpm[u])
ind <- .25 * (n^2 - n)
dd <- min(max(abs(dCo - ind)/ind, 0), 1); cd <- min(max(abs(cCo - ind)/ind, 0), 1)
iD <- 1 - 2 * 0.5^n
nd1 <- abs(DPM_nD(X, target, 0, TRUE) - iD)/iD; nd2 <- abs(DPM_nD(X, target, 1, TRUE) - iD)/iD
cat("discrete_Co_pm =", dCo, " -> early return 1; formula value would be", sqrt(mean(c(dd, cd, nd1, nd2))), "\n")
