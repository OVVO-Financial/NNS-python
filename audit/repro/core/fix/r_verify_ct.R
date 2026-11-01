suppressMessages(library(NNS)); cat("lib:", find.package("NNS"), "\n")
cat("mode 3e9:", NNS.mode(c(3e9,3e9,3e9,1,2), discrete=TRUE), " -5e9:", NNS.mode(c(-5e9,-5e9,-5e9,1,2), discrete=TRUE), "\n")
z <- NNS.mode(c(-0.3,-0.2,0.1,0.2,5), discrete=TRUE); cat("signbit-safe zero:", 1/z, "\n")
set.seed(1); x <- c(5 + 1e-7*rnorm(1000), 6)
t1 <- system.time(g <- NNS.gravity(x))[["elapsed"]]; t2 <- system.time(m <- NNS.mode(x))[["elapsed"]]
cat(sprintf("tight IQR: gravity %.10f (%.3fs) mode %.10f (%.3fs)\n", g, t1, m, t2))
# A/B the 480 random cases: patched R must equal unpatched results stored in ct_cases.txt
L <- readLines("ct_cases.txt"); bad <- 0
p <- function(s) as.numeric(strsplit(s, ",")[[1]])
for (ln in L) { f <- strsplit(ln, ";")[[1]]; x <- p(f[1])
  got <- list(NNS.gravity(x), NNS.mode(x, multi=FALSE), NNS.mode(x, multi=TRUE), NNS.gravity(x, discrete=TRUE), NNS.mode(x, discrete=TRUE, multi=FALSE), NNS.mode(x, discrete=TRUE, multi=TRUE))
  for (k in 1:6) if (!identical(got[[k]], p(f[k+1]))) bad <- bad + 1 }
cat("patched vs unpatched on", length(L), "cases: mismatches", bad, "\n")
print(tryCatch(NNS:::generate.lin.vectors(1:4, 0L), error=function(e) conditionMessage(e)))
str(NNS:::generate.vectors(c(1,2,3), 7L))
x6 <- c(1,3,2,5,4,6); print(tryCatch(NNS.ARMA(x6, h=2, seasonal.factor=9, method="lin", plot=FALSE), error=function(e) conditionMessage(e)))
print(tryCatch(LPM(-1, 0, c(-2,1)), error=function(e) conditionMessage(e)))
print(tryCatch(Co.LPM(-1, c(-2,1), c(-2,1), 0, 0), error=function(e) conditionMessage(e)))
