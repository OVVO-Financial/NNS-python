suppressPackageStartupMessages({library(NNS); library(jsonlite)})
root <- "/home/user/audit/work/REG/cases"; out <- list()
for (cs in list.files(root)) {
  meta <- fromJSON(file.path(root, cs, "args.json"))
  if (!isTRUE(meta$univariate) || length(meta$x_factor_cols)) next
  x <- read.csv(file.path(root, cs, "x.csv"))[[1]]; y <- read.csv(file.path(root, cs, "y.csv"))$y
  if (!is.numeric(y)) next
  out[[cs]] <- c(NNS:::.nns_reg_dependence(x, y), NNS:::.nns_reg_default_order(x, y))
}
write_json(out, "/home/user/audit/work/REG/dep_r.json", digits = NA)
