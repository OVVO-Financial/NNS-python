suppressPackageStartupMessages({library(NNS); library(jsonlite)})
x <- seq(0, 2*pi, pi/100); y <- sin(x)
set.seed(7); xu <- runif(500, -2, 2); yu <- xu^3
write.csv(data.frame(x=xu, y=yu), "cubic.csv", row.names=FALSE)
res <- list(
 sin1 = dy.dx(x, y, eval.point = 1.75),
 sin2 = dy.dx(x, y, eval.point = c(1.75, 2.5)),
 sin_bound = dy.dx(x, y, eval.point = c(0, 2*pi)),
 sin_overall = dy.dx(x, y, eval.point = "overall"),
 cubic = dy.dx(xu, yu, eval.point = c(-1.5, 0, 0.5, 1.5)),
 cubic_sorted = dy.dx(sort(xu), yu[order(xu)], eval.point = c(-1.5, 0, 0.5, 1.5))
)
write_json(lapply(res, function(z) if (is.data.frame(z)) as.list(z) else z), "dydx_r.json", digits = NA, auto_unbox = TRUE)
print(res)
