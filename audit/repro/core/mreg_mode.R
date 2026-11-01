suppressMessages(library(NNS))
set.seed(4); X <- matrix(round(rnorm(600), 1), ncol = 2); y <- round(X[,1] + X[,2]^2 + rnorm(300, sd=.3), 1)
a <- NNS.reg(X, y, noise.reduction = "mode", plot = FALSE, point.only = FALSE)$RPM
options(NNS.native.mreg = FALSE)
b <- NNS.reg(X, y, noise.reduction = "mode", plot = FALSE)$RPM
cat("RPM rows native", nrow(a), " fallback", nrow(b), "; max |y.hat diff| =", max(abs(a$y.hat - b$y.hat)), "\n")
z <- c(0.1, 0.1, 0.2, 0.25, 0.3, 0.31, 0.32, 0.33)
cat("NNS internal mode(z)=", NNS:::mode(z), " NNS.mode(z)=", NNS.mode(z, multi=FALSE), " exact-most-frequent=0.1\n")
