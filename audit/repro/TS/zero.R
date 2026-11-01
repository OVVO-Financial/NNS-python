suppressPackageStartupMessages({library(NNS); library(jsonlite)})
C <- fromJSON("/home/user/audit/work/TS/zero_cases.json")
for (k in names(C)) { a <- list(variable=C[[k]]$x, h=4, plot=FALSE); if (!is.null(C[[k]]$kw$modulo)) a$modulo <- C[[k]]$kw$modulo
  cat(k, "R:", round(do.call(NNS.ARMA, a),3), " Py:", round(C[[k]]$py,3), "\n") }
