suppressMessages(library(NNS))
fs <- sort(list.files("/home/user/audit/work/DEP/caus_cases", full.names=TRUE))
for (f in fs){ d<-read.csv(f); x<-d$x; y<-d$y
  r <- sapply(c(0,1,3), function(tau){ cp<-NNS.caus(x,y,tau=tau); sprintf("tau%d:%s=%.4f(%.4f,%.4f)", tau, names(cp)[3], cp[3], cp[1], cp[2])})
  cat(basename(f), r, "\n") }
# mechanism: lag "normalization" is just a scalar rescale of the contemporaneous series
d<-read.csv(fs[grep("lag_iid_0",fs)]); x<-d$x; tau<-3
xv <- sapply(0:tau, function(i) x[(tau-i+1):(length(x)-i)])
xn <- unlist(NNS.norm(xv)[,1]); cat("cor(x.norm.tau, x[t]) =", cor(xn, x[(tau+1):length(x)]), " ratio range:", range(xn/x[(tau+1):length(x)]), "\n")
