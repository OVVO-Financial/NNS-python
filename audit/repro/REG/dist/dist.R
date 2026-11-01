suppressPackageStartupMessages(library(NNS))
set.seed(11); rpm <- data.frame(a = round(runif(30), 2), b = round(runif(30), 2), y.hat = sample(1:3, 30, TRUE))
writeLines(c("a,b,y.hat", sprintf("%.17g,%.17g,%.17g", rpm$a, rpm$b, rpm$y.hat)), "rpm.csv")
pts <- list(c(0.5, 0.5), c(0.1, 0.9), c(2, -1), c(rpm$a[3], rpm$b[3]))
for (p in pts) for (k in list(1, 2, 5, "all")) cat(sprintf("%.15g", NNS.distance(rpm, p, k = k)), sprintf("%.15g", NNS.distance(rpm, p, k = k, class = TRUE)), "\n")
# tie: two equidistant rows with different classes
r2 <- data.frame(a = c(0, 1, 5), b = c(0, 0, 5), y.hat = c(2, 1, 3))
cat("tie class k=2:", NNS.distance(r2, c(0.5, 0), k = 2, class = TRUE), " k=1:", NNS.distance(r2, c(0.5, 0), k = 1, class=TRUE), NNS.distance(r2, c(0.5, 0), k = 1), "\n")
