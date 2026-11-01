suppressMessages(library(NNS)); X <- as.matrix(read.csv("eff/m39.csv",header=FALSE)); a<-X[,112]; b<-X[,64]
cat(NNS.FSD.uni(a,b,"continuous"), NNS.FSD.uni(b,a,"continuous"), NNS.FSD(a,b,"continuous",plot=FALSE), "\n")
g<-sort(c(a,b)); print(LPM.ratio(1,g,a)-LPM.ratio(1,g,b))
