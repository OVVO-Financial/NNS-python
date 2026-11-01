suppressPackageStartupMessages({library(NNS); library(jsonlite)})
fs <- list(sin1 = list(function(x) sin(x), 1), sq3 = list(function(x) x^2, 3), exp0 = list(function(x) exp(x), 0),
           cub2 = list(function(x) x^3, 2), log5 = list(function(x) log(x), 5), abs1 = list(function(x) abs(x), -1),
           lin = list(function(x) 2*x + 1, 4), sq0 = list(function(x) x^2, 0), tanh05 = list(function(x) tanh(x), 0.5),
           neg = list(function(x) x^4 - 3*x, -2))
out <- lapply(fs, function(p) { r <- NNS.diff(p[[1]], p[[2]]); setNames(as.numeric(r), rownames(r)) })
write_json(out, "nnsdiff_r.json", digits = NA, auto_unbox = TRUE, na = "null")
