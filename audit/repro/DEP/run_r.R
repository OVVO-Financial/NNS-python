suppressMessages(library(NNS)); library(jsonlite)
fs <- sort(list.files("/home/user/audit/work/DEP/cases", full.names=TRUE, pattern="csv$"))
out <- list()
for (f in fs) {
  k <- sub(".csv$", "", basename(f)); d <- read.csv(f); x <- d$x; y <- d$y
  r <- list()
  tr <- function(name, expr) { v <- tryCatch(expr, error=function(e) paste("ERR", conditionMessage(e))); r[[name]] <<- v }
  tr("dep", NNS.dep(x, y)); tr("dep_yx", NNS.dep(y, x))
  tr("asym_xy", NNS.dep(x, y, asym=TRUE)); tr("asym_yx", NNS.dep(y, x, asym=TRUE))
  tr("copula", NNS.copula(cbind(x, y))); tr("copula_disc", NNS.copula(cbind(x, y), continuous=FALSE))
  tr("caus0", if (length(x) <= 3000) as.list(NNS.caus(x, y, tau=0)) else "skip")
  tr("caus3", if (length(x) > 10 && length(x) <= 3000) as.list(NNS.caus(x, y, tau=3)) else "skip")
  tr("pearson", suppressWarnings(cor(x, y)))
  out[[k]] <- r
}
write(toJSON(out, auto_unbox=TRUE, digits=NA, na="string"), "/home/user/audit/work/DEP/r_out.json")
