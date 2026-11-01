# R bug: *_RCPP wrappers test is<NumericVector>(target), which is FALSE for integer vectors,
# and then silently replace the target by mean(variable).
suppressPackageStartupMessages(library(NNS))
x <- c(1, 2, 3, 10)
cat("LPM.ratio(0, 2L, x) =", LPM.ratio(0, 2L, x), " (expected 0.5; LPM.ratio(0, 2, x) =", LPM.ratio(0, 2, x), ", at mean(x)=4:", LPM.ratio(0, 4, x), ")\n")
cat("LPM.ratio(0, 1:3, x) =", LPM.ratio(0, 1:3, x), " (expected 0.25 0.5 0.75)\n")
cat("UPM.ratio(1, 2L, x) =", UPM.ratio(1, 2L, x), " vs UPM.ratio(1, 2, x) =", UPM.ratio(1, 2, x), "\n")
cat("Co.LPM(1, x, x, 2L, 2L) =", Co.LPM(1, x, x, 2L, 2L), " vs doubles:", Co.LPM(1, x, x, 2, 2), "\n")
cat("D.LPM(1, 1, x, rev(x), 0L, 0L) =", D.LPM(1, 1, x, rev(x), 0L, 0L), " vs doubles:", D.LPM(1, 1, x, rev(x), 0, 0), "\n")
A <- cbind(x, rev(x))
cat("PM.matrix(1,1, c(0L,0L), A, FALSE)$clpm[1,2] =", PM.matrix(1, 1, c(0L, 0L), A, FALSE)$cupm[1, 2],
    " vs doubles:", PM.matrix(1, 1, c(0, 0), A, FALSE)$cupm[1, 2], "\n")
set.seed(1); p <- rpois(200, 3)   # integer vector
o <- NNS.CDF(p, 0, plot = FALSE)
cat("NNS.CDF(rpois(200,3)) unique CDF values:", unique(o$Function$CDF), " (expected ECDF; as.numeric gives",
    length(unique(NNS.CDF(as.numeric(p), 0, plot = FALSE)$Function$CDF)), "distinct values)\n")
cat("NNS.CDF(1:10)$Function$CDF =", NNS.CDF(1:10, 0, plot = FALSE)$Function$CDF, "\n")
cat("NNS.CDF(airquality$Ozone) CDF range =", range(NNS.CDF(airquality$Ozone, 0, plot = FALSE)$Function$CDF), "\n")
cat("LPM(0, 2L, x) (wrapper coerces) =", LPM(0, 2L, x), "\n")
cat("LPM.VaR(0.25, 1, 1:10) =", LPM.VaR(0.25, 1, 1:10), " LPM.ratio(1, that, 1:10) =", LPM.ratio(1, LPM.VaR(0.25, 1, 1:10), 1:10), "\n")
