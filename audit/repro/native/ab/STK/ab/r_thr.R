suppressPackageStartupMessages({library(NNS);library(jsonlite)})
spec <- fromJSON("cases/b_thr15/spec.json"); Xtr <- read.csv(spec$xtrain); ytr <- read.csv(spec$ytrain)[[1]]; Xte <- read.csv(spec$xtest)
ns <- asNamespace("NNS"); f <- get(".nns_boost_threshold", ns)
g <- function(threshold, objective, extreme, learner.results) { r <- f(threshold, objective, extreme, learner.results); writeLines(sprintf("%.17g", learner.results), "r_learner.txt"); cat("R cutoff", sprintf("%.17g", r$cutoff), "max", sprintf("%.17g", max(learner.results)), "\n"); r }
unlockBinding(".nns_boost_threshold", ns); assign(".nns_boost_threshold", g, ns)
b <- NNS.boost(Xtr, ytr, Xte, threshold=1.5, epochs=15, learner.trials=15, status=FALSE, feature.importance=FALSE)
print(b$feature.frequency)
