suppressPackageStartupMessages({library(NNS); library(jsonlite)})
C <- fromJSON("/home/user/audit/work/TS/arma_cfg.json", simplifyVector=TRUE, simplifyDataFrame=FALSE)
res <- list()
for (i in seq_along(C$cfgs)) {
  c <- C$cfgs[[i]]; x <- C$series[[c$series]]
  a <- list(variable=x, h=c$h, plot=FALSE, seasonal.plot=FALSE, method=c$method, seasonal.factor=c$seasonal_factor)
  if(!is.null(c$training_set)) a$training.set <- c$training_set
  if(!is.null(c$best_periods)) a$best.periods <- c$best_periods
  if(!is.null(c$weights)) a$weights <- c$weights
  if(!is.null(c$shrink)) a$shrink <- c$shrink
  if(!is.null(c$dynamic)) a$dynamic <- c$dynamic
  if(!is.null(c$modulo)) a$modulo <- c$modulo
  if(!is.null(c$negative_values)) a$negative.values <- c$negative_values
  t0 <- Sys.time()
  r <- tryCatch(as.numeric(do.call(NNS.ARMA, a)), error=function(e) paste("ERR:", conditionMessage(e)))
  res[[i]] <- list(out=r, t=as.numeric(Sys.time()-t0))
}
cat(toJSON(res, digits=NA, auto_unbox=TRUE, na="string"))
