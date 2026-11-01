suppressMessages(library(NNS))
n <- 100; p <- ((1:n)-0.5)/n; x <- qnorm(p); y <- qnorm(p,0,2)
cat(sprintf("mean x %.3e mean y %.3e\n", mean(x), mean(y)))
g <- sort(c(x,y)); d1 <- LPM(1,g,x)-LPM(1,g,y); d2 <- LPM(2,g,x)-LPM(2,g,y)
cat(sprintf("max LPM1 diff %.3e at %d ; max LPM2 diff %.3e\n", max(d1), which.max(d1), max(d2)))
print(c(NNS.SSD(x,y,plot=FALSE), NNS.SSD.uni(x,y), NNS.TSD(x,y,plot=FALSE), NNS.TSD.uni(x,y)))
# translate both by constant 0.37 -- dominance should be invariant
print(c(NNS.SSD(x+0.37,y+0.37,plot=FALSE), NNS.SSD.uni(x+0.37,y+0.37), NNS.TSD(x+0.37,y+0.37,plot=FALSE)))
x2<-c(0.1,0.7); y2<-c(0,0.8); cat(sprintf("%.17g %.17g\n", mean(x2), mean(y2))); print(NNS.SSD(x2,y2,plot=FALSE)); print(NNS.SSD(x2*10,y2*10,plot=FALSE))
