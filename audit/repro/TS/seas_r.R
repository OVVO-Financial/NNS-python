suppressPackageStartupMessages({library(NNS); library(jsonlite)})
d <- "/home/user/audit/work/TS/data/"
S <- read.csv(paste0(d,"synth.csv"))
series <- list(ap=read.csv(paste0(d,"ap.csv"))$y, vix=read.csv(paste0(d,"vix_monthly.csv"))$y,
  s12=S$s12, s7=S$s7, s12_trend=S$s12_trend, s_neg=S$s_neg, s_rw=S$s_rw, wn=S$wn,
  odd11=c(3,1,4,1,5,9,2,6,5,3,5), odd13=c(1,2,3,1,2,3,1,2,3,1,2,3,1), zero_mean=c(-1,1,-1,1,-1,1,-1,1,-1,1))
out <- list()
for (nm in names(series)) {
  x <- series[[nm]]
  r <- NNS.seas(x, plot=FALSE)
  r2 <- NNS.seas(x, modulo=c(12), plot=FALSE)
  r3 <- NNS.seas(x, modulo=c(7,12), mod.only=FALSE, plot=FALSE)
  out[[nm]] <- list(x=x, best=r$best.period, periods=r$periods, cv=r$all.periods$Coefficient.of.Variation,
     m12=r2$periods, m12cv=r2$all.periods$Coefficient.of.Variation, m712=r3$periods)
}
cat(toJSON(out, digits=NA, auto_unbox=TRUE))
