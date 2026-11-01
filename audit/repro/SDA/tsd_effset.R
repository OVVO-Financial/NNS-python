suppressMessages(library(NNS))
x <- c(-0.85,-1.52,0.68); y <- c(-0.21,-0.33,-1.67); z <- c(-3, 0.16, 0.2)
print(NNS.SD.efficient.set(cbind(x=x,y=y), 3, status=FALSE))
print(NNS.SD.efficient.set(cbind(x=x,y=y,z=z), 3, status=FALSE))
