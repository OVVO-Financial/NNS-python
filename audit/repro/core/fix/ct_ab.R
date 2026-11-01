# Write random vectors and R 13.3 gravity/mode results for bit-exact A/B.
suppressMessages(library(NNS))
set.seed(42)
out <- file("ct_cases.txt", "w")
gen <- list(
  function(n) rnorm(n),
  function(n) rcauchy(n),                       # huge bin counts, long empty runs
  function(n) round(rnorm(n) * 3),              # discrete ties
  function(n) c(rnorm(n - 3), 50, -80, 1e3),    # outliers
  function(n) rexp(n)^3,
  function(n) c(rnorm(n/2, -3), rnorm(n/2, 3)), # bimodal
  function(n) sample(c(1, 2, 2, 3, 10, 50), n, TRUE),
  function(n) rt(n, 1.5) * 1e-3 + 100
)
k <- 0
for (rep in 1:60) for (g in gen) {
  n <- sample(c(4:12, 20, 50, 200, 1000, 3000), 1)
  x <- g(n)
  gv <- NNS.gravity(x); m1 <- NNS.mode(x, multi = FALSE); mm <- NNS.mode(x, multi = TRUE)
  gd <- NNS.gravity(x, discrete = TRUE); md <- NNS.mode(x, discrete = TRUE, multi = FALSE)
  mdm <- NNS.mode(x, discrete = TRUE, multi = TRUE)
  f <- function(v) paste(sprintf("%.17g", v), collapse = ",")
  writeLines(paste(f(x), f(gv), f(m1), f(mm), f(gd), f(md), f(mdm), sep = ";"), out)
  k <- k + 1
}
close(out)
cat("cases:", k, "\n")
