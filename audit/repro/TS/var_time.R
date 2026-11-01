suppressPackageStartupMessages(library(NNS))
for (f in c("daily5_500","daily5_1000")) { X <- as.matrix(read.csv(paste0("data/",f,".csv")))
 t <- system.time(r <- NNS.VAR(X, h=5, tau=2, status=FALSE, ncores=1))[3]; cat(f, "R VAR", t, "s\n") 
 write.csv(as.matrix(r$ensemble), paste0("var_ens_", f, ".csv"), row.names=FALSE) }
