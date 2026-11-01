suppressPackageStartupMessages({library(NNS); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
root <- "/home/user/audit/work/REG/cases"
out_root <- "/home/user/audit/work/REG/out_r"
dir.create(out_root, showWarnings = FALSE)
cases <- if (length(args)) args else list.files(root)

to_list <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.data.frame(v)) return(lapply(as.list(v), function(z) if (is.factor(z)) as.character(z) else z))
  if (is.matrix(v)) return(lapply(seq_len(ncol(v)), function(j) v[, j]))
  if (is.list(v)) return(lapply(v, to_list))
  v
}

for (cs in cases) {
  d <- file.path(root, cs)
  meta <- fromJSON(file.path(d, "args.json"), simplifyVector = TRUE)
  x <- read.csv(file.path(d, "x.csv"), stringsAsFactors = FALSE, check.names = FALSE)
  y <- read.csv(file.path(d, "y.csv"), stringsAsFactors = FALSE)$y
  if (isTRUE(meta$y_factor)) y <- factor(y)
  for (fc in meta$x_factor_cols) x[[fc]] <- factor(x[[fc]])
  pe <- NULL
  if (isTRUE(meta$has_pe)) {
    pe <- read.csv(file.path(d, "pe.csv"), stringsAsFactors = FALSE, check.names = FALSE)
    for (fc in meta$x_factor_cols) pe[[fc]] <- factor(pe[[fc]], levels = levels(x[[fc]]))
  }
  if (isTRUE(meta$univariate)) {
    x <- x[[1]]
    if (!is.null(pe)) pe <- pe[[1]]
  }
  a <- meta$args
  call_args <- c(list(x = x, y = y, point.est = pe, plot = FALSE, ncores = 1), a)
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch(do.call(NNS.reg, call_args), error = function(e) list(error = conditionMessage(e)))
  el <- proc.time()[["elapsed"]] - t0
  out <- lapply(res, to_list)
  out$elapsed <- el
  write_json(out, file.path(out_root, paste0(cs, ".json")), digits = NA, auto_unbox = TRUE, null = "null", na = "null")
  cat(cs, if (!is.null(res$error)) paste("ERROR:", res$error) else "ok", sprintf("%.3fs", el), "\n")
}
