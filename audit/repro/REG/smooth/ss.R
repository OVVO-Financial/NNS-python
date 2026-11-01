x <- c(0, 0.7, 1.3, 2.0, 3.1, 4.0, 5.5, 6.2)
y <- c(0.1, 0.8, 0.9, 0.95, 0.1, -0.7, -0.75, -0.1)
for (s in c(0.3, 0.6, 0.75)) {
  f <- smooth.spline(x, y, spar = s)
  cat(s, sprintf("%.12f", predict(f, c(-1, 0, 1, 3, 6.2, 7))$y), " lambda=", f$lambda, " nknots=", f$fit$nk, "\n")
}
set.seed(1); x2 <- sort(runif(80, 0, 10)); y2 <- sin(x2) + rnorm(80, 0, .1)
f <- smooth.spline(x2, y2, spar = 0.7); cat("n80", sprintf("%.12f", predict(f, c(1, 5, 9))$y), "nk", f$fit$nk, "lambda", f$lambda, "\n")
write.csv(data.frame(x=x2,y=y2), "/home/user/audit/work/REG/smooth/x80.csv", row.names=FALSE)
