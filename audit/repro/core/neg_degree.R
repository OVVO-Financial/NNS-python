suppressMessages(library(NNS))
cat("LPM(-1,0,c(-2,1)) =", LPM(-1, 0, c(-2,1)), " (mean(pmax(0-x,0)^-1 over positive part) = 0.25)\n")
cat("Co.LPM(-1, c(-2,1), c(-2,1), 0, 0) =", Co.LPM(-1, c(-2,1), c(-2,1), 0, 0), "\n")
