suppressPackageStartupMessages({library(NNS); library(jsonlite)})
d <- "/home/user/audit/work/TS/data/"
S <- list(ap=read.csv(paste0(d,"ap.csv"))$y, vix=read.csv(paste0(d,"vix_monthly.csv"))$y, spyd=read.csv(paste0(d,"spy_daily_ret_1000.csv"))$y)
stats <- function(x, b, td=NULL) {
  R <- b$replicates
  sp <- apply(R, 2, function(z) cor(z, x, method="spearman"))
  pe <- apply(R, 2, function(z) cor(z, x))
  sl <- apply(R, 2, function(z) coef(lm(z ~ seq_along(z)))[2])
  list(sp_mean=mean(sp), sp_sd=sd(sp), pe_mean=mean(pe), colmean_mean=mean(colMeans(R)), colmean_sd=sd(colMeans(R)),
       colsd_mean=mean(apply(R,2,sd)), slope_mean=mean(sl), ens_mean=mean(b$ensemble), ens_sd=sd(b$ensemble),
       min=min(R), max=max(R), q01=unname(quantile(R,.01)), q99=unname(quantile(R,.99)))
}
out <- list()
set.seed(42)
for (nm in names(S)) {
  x <- S[[nm]]
  for (rho in c(-1,-0.5,0,0.5,1)) {
    t0 <- Sys.time(); b <- NNS.meboot(x, reps=300, rho=rho)[, 1]; el <- as.numeric(difftime(Sys.time(), t0, units="secs"))
    out[[paste(nm, rho)]] <- c(stats(x, b), list(time=el))
    if (rho == 0) out[[paste(nm, "det")]] <- list(xx=b$xx, z=b$z, dv=b$dv, dvtrim=b$dvtrim, xmin=b$xmin, xmax=b$xmax, desintxb=b$desintxb, ordxx=b$ordxx, x=x)
  }
  b <- NNS.meboot(x, reps=300, rho=0.5, drift=FALSE)[,1]; out[[paste(nm,"nodrift")]] <- stats(x,b)
  b <- NNS.meboot(x, reps=300, rho=0.5, target_drift=0)[,1]; out[[paste(nm,"td0")]] <- stats(x,b)
  b <- NNS.meboot(x, reps=300, rho=0.5, target_drift_scale=2)[,1]; out[[paste(nm,"tds2")]] <- stats(x,b)
  b <- NNS.meboot(x, reps=300, rho=0.5, xmin=min(x))[,1]; out[[paste(nm,"xmin")]] <- stats(x,b)
  b <- NNS.meboot(x, reps=300, rho=0.5, type="pearson")[,1]; out[[paste(nm,"pearson")]] <- stats(x,b)
}
x <- S$ap; t0<-Sys.time(); b <- NNS.meboot(x, reps=50, rho=0.5, type="nnsdep")[,1]; out[["ap nnsdep"]] <- c(stats(x,b), list(time=as.numeric(difftime(Sys.time(), t0, units="secs"))))
cat(toJSON(out, digits=NA, auto_unbox=TRUE))
