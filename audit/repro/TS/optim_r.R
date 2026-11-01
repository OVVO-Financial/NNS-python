suppressPackageStartupMessages({library(NNS); library(jsonlite)})
C <- fromJSON("/home/user/audit/work/TS/optim_cfg.json", simplifyVector=TRUE, simplifyDataFrame=FALSE)
res <- list()
for (c in C) {
  a <- list(variable=c$x, seasonal.factor=c$seasonal_factor, print.trace=FALSE, ncores=1)
  for (k in c("h","training_set","lin_only","linear_approximation","objective")) if(!is.null(c[[k]])) a[[gsub("_",".",k)]] <- c[[k]]
  t0 <- Sys.time()
  r <- tryCatch(do.call(NNS.ARMA.optim, a), error=function(e) list(err=conditionMessage(e)))
  r$time <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  res[[c$name]] <- r
}
cat(toJSON(res, digits=NA, auto_unbox=TRUE, null="null", na="string"))
