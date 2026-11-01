suppressMessages(library(NNS)); set.seed(1); b <- rnorm(50)
X <- cbind(A=b+3, B=b+2, C=b+1, D=b, E=b-1)
str(NNS.SD.cluster(X, degree=1, min_cluster=2)$Clusters)
str(NNS.SD.cluster(X, degree=1, min_cluster=0)$Clusters)
