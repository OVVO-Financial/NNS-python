suppressMessages(library(NNS))
for (f in c("cop_fin4","cop_fin11","cop_3","cop_3ind","cop_3disc","cop_3same","cop_3anti","cop_8ind")){ X<-as.matrix(read.csv(paste0("/home/user/audit/work/DEP/",f,".csv")))
 cat(f, sprintf("%.12f", c(NNS.copula(X), NNS.copula(X,continuous=FALSE), NNS.copula(X,target=rep(0,ncol(X))))), "\n")}
