suppressMessages(library(NNS)); X <- as.matrix(read.csv("big_500_4232.csv", header=FALSE))
py <- NULL
for (d in 1:3) { t0<-proc.time()[[3]]; e <- NNS.SD.efficient.set(X,d,status=FALSE); cat(sprintf("R k=500 n=4232 deg=%d: %.0f ms |eff|=%d\n", d, (proc.time()[[3]]-t0)*1000, length(e))); }
