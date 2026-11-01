suppressMessages(library(NNS))
cat("LPM NaN target:", LPM(1, NaN, c(1,2,3)), " Inf target:", LPM(1, Inf, c(1,2,3)), UPM(1,-Inf,c(1,2,3)), "\n")
cat("LPM.ratio all equal:", LPM.ratio(1, 2, c(2,2,2)), " UPM.ratio:", UPM.ratio(1,2,c(2,2,2)), "\n")
cat("LPM.ratio deg0 tie:", LPM.ratio(0, 2, c(1,2,3)), "\n")
print(try(Co.LPM(1, c(1,NA,3,0), c(1,2,NA,0), 1, 1)))
cat("LPM frac deg 0.5:", LPM(0.5, 0, c(-1,-4, 3)), "\n")
cat("LPM deg 40 (beyond prefix 32) 50 targets:", LPM(40, seq(0,1,length.out=50), c(0.1,0.5,0.9))[50], "\n")
print(try(LPM(1, 0, c(-Inf, 1))))

