suppressPackageStartupMessages({library(NNS);library(jsonlite)})
run <- function(case, extra=list()) { spec <- fromJSON(file.path("cases",case,"spec.json"))
 Xtr <- read.csv(spec$xtrain); ytr <- read.csv(spec$ytrain)[[1]]; Xte <- read.csv(spec$xtest); if (isTRUE(spec$yfactor)) ytr <- factor(ytr)
 r <- do.call(NNS.stack, c(list(Xtr,ytr,Xte,status=FALSE,ncores=1), spec$args, extra)); c(reg=r$OBJfn.reg, nb=r$NNS.reg.n.best, dim=r$OBJfn.dim.red, w=r$weights[1], thr=r$probability.threshold) }
for (cs in c("iris_class_bal","iris_class_nothr","iris_bin01")) {
 cat(cs, "\n"); print(run(cs))
 options(NNS.native.univariate=FALSE); print(run(cs)); options(NNS.native.univariate=TRUE)
 options(NNS.native.stack=FALSE); print(run(cs)); options(NNS.native.stack=TRUE)
}
