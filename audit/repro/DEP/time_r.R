suppressMessages(library(NNS)); RcppParallel::setThreadOptions(numThreads=1)
tm <- function(expr, reps=3){ e<-substitute(expr); b<-Inf; for(i in 1:reps){t<-system.time(eval(e, parent.frame()))[["elapsed"]]; b<-min(b,t)}; b }
for (n in c(1000,10000,100000)){ d<-read.csv(sprintf("/home/user/audit/work/DEP/time_%d.csv",n)); x<-d$x; y<-d$y
  r <- c(n=n, dep=tm(NNS.dep(x,y)), dep_asym=tm(NNS.dep(x,y,asym=TRUE)), copula=tm(NNS.copula(cbind(x,y))),
     part_off=tm(NNS.part(x,y)), part_xonly=tm(NNS.part(x,y,type="XONLY")),
     part_mean=tm(NNS.part(x,y,noise.reduction="mean"),1), part_median=tm(NNS.part(x,y,noise.reduction="median"),1), part_mode=tm(NNS.part(x,y,noise.reduction="mode"),1),
     caus0=if(n<=10000) tm(NNS.caus(x,y,tau=0),1) else NA, caus3=if(n<=10000) tm(NNS.caus(x,y,tau=3),1) else NA)
  print(round(r,4)) }
M <- as.matrix(read.csv("/home/user/audit/work/DEP/time_mat10.csv"))
cat("dep matrix 10 cols n=5000:", tm(NNS.dep(M)), "\n")
cat("copula 10 cols n=5000:", tm(NNS.copula(M)), "\n")
cat("caus matrix 10 cols n=5000:", tm(NNS.caus(M),1), "\n")
cat("caus matrix 10 cols n=1000:", tm(NNS.caus(M[1:1000,]),1), "\n")
