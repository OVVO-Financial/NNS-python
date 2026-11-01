suppressPackageStartupMessages(library(NNS))
x <- read.csv("/home/user/audit/work/TS/data/ap.csv")$y
f <- function(l, e) { r <- tryCatch(e, error=function(err) paste("ERR", conditionMessage(err))); cat(l, ":", if(is.character(r)) r else paste(class(r), paste(dim(r), collapse="x")), "\n") }
f("td vec", NNS.meboot(x, reps=5, rho=0, target_drift=c(1,7)))
f("reps1", NNS.meboot(x, reps=1, rho=.5))
f("n2", NNS.meboot(c(1,2), reps=5, rho=.5))
f("n3", NNS.meboot(c(1,2,4), reps=5, rho=.5))
f("const", NNS.meboot(rep(1,20), reps=5, rho=.5))
f("linear exact", NNS.meboot(0:19, reps=5, rho=.5))
f("mc wrong sign", NNS.MC(x, reps=3, lower_rho=1, upper_rho=-1, by=.5))
set.seed(1); r <- NNS.meboot(x, reps=5, rho=.5, xmin=0, force.clt=FALSE)[,1]; cat("xmin+force.clt=F colsd", apply(r$replicates,2,sd), "\n")
cat("dvtrim trim=.5 R", NNS.meboot(x, reps=3, rho=0, trim=.5)[,1]$dvtrim, "\n")
