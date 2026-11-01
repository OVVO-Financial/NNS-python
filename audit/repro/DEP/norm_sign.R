suppressMessages(library(NNS)); set.seed(5); x <- rnorm(500, 1); y <- x - 1.5 + rnorm(500, sd=.3)
cat("means", mean(x), mean(y), " cor", cor(x,y), "\n"); N <- NNS.norm(cbind(x,y)); print(head(N,3)); cat("cor after norm:", cor(N[,1],N[,2]), " scales:", N[1,1]/x[1], N[1,2]/y[1], "\n")
print(NNS.caus(x,y,tau=0)); print(NNS.caus(x, y+10, tau=0))
