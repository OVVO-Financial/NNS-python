# R: LPM/UPM validate finiteness but LPM.ratio/UPM.ratio/Co.*/D.* do not; NA is counted in n.
suppressPackageStartupMessages(library(NNS))
x <- c(1, 2, 3, NA, 5)
print(try(LPM(0, 2, x), silent = TRUE))
cat("LPM.ratio(0,2,x)+UPM.ratio(0,2,x) =", LPM.ratio(0, 2, x) + UPM.ratio(0, 2, x), "\n")
cat("LPM.ratio(0,6,x) =", LPM.ratio(0, 6, x), " Co.LPM(0,x,x,6,6) =", Co.LPM(0, x, x, 6, 6), "\n")
cat("LPM(-1, 3, 1:5) =", LPM(-1, 3, 1:5), " LPM.VaR(0.05,-1,0:9) =", LPM.VaR(0.05, -1, 0:9), "\n")
cat("LPM(0,2,c(1,2,3),excess_ret=TRUE) =", LPM(0, 2, c(1, 2, 3), excess_ret = TRUE), " vs LPM(0,2,c(1,2,3)) =", LPM(0, 2, c(1, 2, 3)), "\n")
cat("LPM.ratio(1, NULL, 1:5) =", LPM.ratio(1, NULL, 1:5), " LPM(1, NULL, 1:5) length =", length(LPM(1, NULL, 1:5)), "\n")
