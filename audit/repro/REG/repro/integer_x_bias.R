suppressPackageStartupMessages(library(NNS))
for (s in c(0, 0.5)) { x <- as.numeric(1:1000) + s; r <- NNS.reg(x, 2*x, plot = FALSE)
  cat("shift", s, "max|yhat - y|", max(abs(r$Fitted.xy$y.hat - 2*x)), " first rp:", head(r$regression.points$x, 3), "/", head(r$regression.points$y, 3), "\n") }
