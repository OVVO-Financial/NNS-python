suppressPackageStartupMessages({library(NNS);library(mlbench);library(caret);library(jsonlite)})
data("BostonHousing")
set.seed(12345)
in_train <- createDataPartition(y = BostonHousing$medv, p = 0.70, list = FALSE)
BostonHousing$chas <- as.numeric(as.character(BostonHousing$chas))
training <- BostonHousing[in_train, ]; testing <- BostonHousing[-in_train, ]
write.csv(BostonHousing, "boston_full.csv", row.names=FALSE)
writeLines(as.character(as.integer(in_train)-1), "train_idx.txt")
rmse <- function(p,a) sqrt(mean((p-a)^2))
run <- function(...) {
  t0 <- Sys.time()
  r <- NNS.stack(training[, -14], training[, 14], IVs.test = testing[, -14], status = FALSE,
     obj.fn = expression(sqrt(mean((predicted - actual)^2))), objective='min', ncores=1, ...)
  el <- as.numeric(Sys.time()-t0, units="secs")
  list(reg=rmse(r$reg,testing$medv), dim=rmse(r$dim.red,testing$medv), stack=rmse(r$stack,testing$medv),
       nbest=r$NNS.reg.n.best, thr=r$NNS.dim.red.threshold, w=r$weights, objreg=r$OBJfn.reg, objdim=r$OBJfn.dim.red,
       vars=r$NNS.dim.red.variables, head=head(r$stack,5), secs=el)
}
out <- list(default=run(), cv25=run(CV.size=0.25))
cat(toJSON(out, auto_unbox=TRUE, digits=NA))
