suppressPackageStartupMessages(library(NNS))
x <- read.csv("data/ap.csv")$y
set.seed(1); t <- system.time(m <- NNS.MC(x))[3]; cat("R NNS.MC default (201 rhos x 30 reps):", t, "s; n reps", length(m$replicates), "\n")
m2 <- NNS.MC(x, reps=5, by=.5, exp=2); cat(names(m2$replicates), "\n")
cat("ens mean", mean(m$ensemble), "sd", sd(m$ensemble), "\n")
sp <- sapply(m$replicates, function(R) mean(apply(R,2,function(z) cor(z,x,method="spearman")))); print(round(sp[c(1,51,101,151,201)],3))
