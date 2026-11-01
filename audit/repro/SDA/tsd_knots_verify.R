suppressMessages(library(NNS))
x <- c(-0.85,-1.52,0.68); y <- c(-0.21,-0.33,-1.67)
print(c(NNS.TSD(x,y,plot=FALSE), NNS.TSD.uni(x,y)))
tt <- seq(-1.67, 0.68, by=1e-4); d <- LPM(2,tt,x)-LPM(2,tt,y)
cat(sprintf("max LPM2_x - LPM2_y on dense grid = %.5f at t = %.4f\n", max(d), tt[which.max(d)]))
g <- sort(c(x,y)); cat("max at merged knots:", max(LPM(2,g,x)-LPM(2,g,y)), "\n")
# efficient set: third column with a value inside violation interval
X <- cbind(x=x, y=y)
print(NNS.SD.efficient.set(X, 3, status=FALSE))
z <- c(-1.45, 5, 6)  # z has knot at -1.45 inside the interval
X3 <- cbind(x=x, y=y, z=z)
print(NNS.SD.efficient.set(X3, 3, status=FALSE))
