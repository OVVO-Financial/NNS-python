args <- commandArgs(TRUE)
suppressMessages({library(testthat); library(NNS)})
cat("lib:", find.package("NNS"), "\n")
for (f in list.files("testthat", pattern = "^test.*\\.R$", full.names = TRUE)) {
  r <- tryCatch(as.data.frame(test_file(f, reporter = "silent", stop_on_failure = FALSE)), error = function(e) NULL)
  if (is.null(r)) { cat(basename(f), "ERROR\n"); next }
  cat(sprintf("%-45s tests %3d failed %2d errors %d skipped %d\n", basename(f), sum(r$nb), sum(r$failed), sum(r$error), sum(r$skipped)))
}
