suppressMessages(library(NNS))
set.seed(1); x <- sample(0:2, 5000, TRUE); y <- sample(0:2, 5000, TRUE)
dx <- x-mean(x); dy <- y-mean(y)
cat("mean x,y:", mean(x), mean(y), "\n")
d0co <- mean((dx>0&dy>0) | (dx<=0&dy<=0)); dpm0 <- mean(!((dx<0&dy<0)|(dx>0&dy>0)))
cat("d0_Co", d0co, " dpm_d0", dpm0, " sum", d0co+dpm0, "\n")
# exact tie case: center data so the mean is a data value
x2 <- x - mean(x) + 1; x2[x2 > 0.5 & x2 < 1.5] <- 1
set.seed(3); x <- sample(c(-1,0,1), 6000, TRUE); y <- sample(c(-1,0,1), 6000, TRUE)
x <- c(x, -x); y <- c(y, sample(y))   # make means exactly 0
cat("means", mean(x), mean(y), "\n")
dx <- x; dy <- y
d0co <- mean((dx>0&dy>0) | (dx<=0&dy<=0)); dpm0 <- mean(!((dx<0&dy<0)|(dx>0&dy>0)))
cat("tie case d0_Co", d0co, " (indep null used 0.5)  dpm_d0", dpm0, " (null used 0.5)  sum", d0co+dpm0, "\n")
print(NNS.dep(x, y)); print(NNS.copula(cbind(x,y)))
