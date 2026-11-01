suppressMessages({library(NNS); library(jsonlite)})
cases <- fromJSON("/home/user/audit/work/core/ab_cases.json", simplifyVector = FALSE)
simp <- function(z) { if (is.list(z) && length(z) && all(vapply(z, function(e) is.list(e), TRUE))) { if (all(vapply(z, length, 1L) == length(z[[1]])) && all(vapply(z, function(e) all(vapply(e, is.numeric, TRUE)), TRUE))) return(do.call(rbind, lapply(z, unlist))); return(lapply(z, simp)) }; if (is.list(z) && all(vapply(z, function(e) is.numeric(e) || is.logical(e), TRUE)) && all(vapply(z, length, 1L)==1)) return(unlist(z)); z }
mat <- function(m) { m <- as.matrix(m); storage.mode(m) <- "double"; m }
cmp <- function(py, r) {
  py <- unlist(py); r <- unlist(r)
  if (length(py) != length(r)) return(c(Inf, Inf, length(py), length(r)))
  d <- abs(py - r); d[is.na(py) & is.na(r)] <- 0
  rel <- d / pmax(abs(r), 1e-300)
  c(max(d), max(rel[d > 0], 0), length(py), length(r))
}
res <- list()
for (i in seq_along(cases)) {
  cs <- cases[[i]]; nm <- cs$name; f <- cs$func; a <- lapply(cs$args, simp); py <- cs$py
  if (is.list(py) && !is.null(names(py))) py <- lapply(py, simp) else py <- simp(py)
  r <- tryCatch(switch(f,
    co_lpm = Co.LPM(a$dx, a$x, a$y, a$tx, a$ty, degree_y = a$dy),
    co_upm = Co.UPM(a$dx, a$x, a$y, a$tx, a$ty, degree_y = a$dy),
    d_lpm = D.LPM(a$dx, a$dy, a$x, a$y, a$tx, a$ty),
    d_upm = D.UPM(a$dx, a$dy, a$x, a$y, a$tx, a$ty),
    lpm_v = LPM(a$deg, a$t, a$x), upm_v = UPM(a$deg, a$t, a$x),
    lpm_ratio_v = LPM.ratio(a$deg, a$t, a$x), upm_ratio_v = UPM.ratio(a$deg, a$t, a$x),
    co_lpm_v = Co.LPM(a$dx, a$x, a$y, a$tx, a$ty, degree_y = a$dy),
    d_upm_v = D.UPM(a$dx, a$dy, a$x, a$y, a$tx, a$ty),
    clpm_nd = Co.LPM_nD(mat(a$X), a$t, a$deg, a$norm),
    cupm_nd = Co.UPM_nD(mat(a$X), a$t, a$deg, a$norm),
    dpm_nd = DPM_nD(mat(a$X), a$t, a$deg, a$norm),
    clpm_nd_batch = NNS:::Co.LPM_nD.batch(mat(a$X), mat(a$T), a$deg, a$norm),
    pm_matrix = { p <- PM.matrix(a$dl, a$du, a$t, mat(a$X), a$pop, a$norm); list(p$cupm, p$dupm, p$dlpm, p$clpm, p$cov.matrix) },
    fast_lm = { z <- NNS:::fast_lm(a$x, a$y); list(z$coef, z$fitted.values) },
    fast_lm_mult = { z <- NNS:::fast_lm_mult(mat(a$X), a$y); list(z$coefficients, z$r.squared) },
    gravity = NNS.gravity(a$x, a$d),
    mode = NNS.mode(a$x, discrete = a$d, multi = a$m),
    generate_vectors = { z <- NNS:::generate.vectors(a$x, as.integer(a$l)); list(z$Component.series, z$Component.index) },
    generate_lin_vectors = { z <- NNS:::generate.lin.vectors(a$x, as.integer(a$l), as.integer(a$h)); list(z$Component.series, z$Component.index, z$forecast.values, z$forecast.index) },
    stoch_sup = { z <- NNS:::stoch_superiority_cpp(a$x, a$y); list(z$p_gt, z$p_tie, z$p_star) },
    is_discrete = as.numeric(NNS:::is.discrete(a$x)),
    stop("unknown ", f)), error = function(e) paste("R ERROR:", conditionMessage(e)))
  if (is.character(r)) { res[[nm]] <- c(NA, NA, NA, NA); cat(sprintf("%-40s %s\n", nm, r)); next }
  if (f == "pm_matrix") py <- list(py$cupm, py$dupm, py$dlpm, py$clpm, py$`cov.matrix`)
  if (f == "fast_lm") py <- list(py$coef, py$fitted)
  if (f == "fast_lm_mult") py <- list(py$coef, py$r2)
  if (f == "generate_vectors") py <- list(py$series, py$index)
  if (f == "generate_lin_vectors") py <- list(py$series, py$index, py$forecast_values, py$forecast_index)
  if (f == "stoch_sup") py <- list(py$p_gt, py$p_tie, py$p_star)
  if (f == "is_discrete") py <- as.numeric(py)
  res[[nm]] <- cmp(py, r)
}
out <- do.call(rbind, res)
colnames(out) <- c("max_abs", "max_rel", "n_py", "n_r")
bad <- out[!is.na(out[, 1]) & (out[, 1] > 1e-12 & out[, 2] > 1e-9), , drop = FALSE]
cat("\n== cases:", nrow(out), " mismatches (abs>1e-12 & rel>1e-9):", nrow(bad), "\n")
print(signif(bad, 4))
cat("\n== tiny fp-level diffs (nonzero but within tol):\n")
tiny <- out[!is.na(out[, 1]) & out[, 1] > 0 & !(rownames(out) %in% rownames(bad)), , drop = FALSE]
print(signif(tiny, 3))
