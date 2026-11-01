suppressMessages(library(NNS))
x <- c(1,2); y <- c(1,1,2,2)
cat("identical distributions, unequal n:", NNS.TSD.uni(x,y), NNS.TSD.uni(y,x), NNS.SSD.uni(x,y), NNS.SSD.uni(y,x), "| bi:", NNS.TSD(x,y,plot=FALSE), "\n")
x <- c(5,6,7); y <- c(0,1,2,3,4,5,6,7,8,9)  # y has larger tail beyond x's length
cat("x=5:7 vs y=0:9  FSD.uni(y,x) =", NNS.FSD.uni(y,x), " (bi:", NNS.FSD(x,y,plot=FALSE), ")\n")
cat("x=0 vs y=c(1,2,3,-100): FSD.uni(x,y)=", NNS.FSD.uni(0, c(1,2,3,-100)), " bi:", NNS.FSD(0, c(1,2,3,-100), plot=FALSE),"\n")
cat("x=c(-100,1,2,3) vs 0: FSD.uni(y,x) uses m=1 =>", NNS.FSD.uni(c(-100,1,2,3)[c(2,3,4,1)], 0), "\n")
