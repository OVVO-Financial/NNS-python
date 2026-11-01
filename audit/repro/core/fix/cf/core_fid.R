suppressMessages(library(NNS))
for (d in c("ret", "disc", "ap")) {
  x <- scan(sprintf("x_%s.txt", d), quiet = TRUE); y <- scan(sprintf("y_%s.txt", d), quiet = TRUE)
  L <- strsplit(readLines(sprintf("core_%s.out", d)), " ")
  core <- setNames(lapply(L, `[`, -1), vapply(L, `[`, "", 1))
  num <- function(k) as.numeric(core[[k]])
  res <- c()
  for (noise in c("off", "mean", "median", "mode")) for (xo in 0:1) {
    p <- NNS:::NNS_part_cpp(x, y, type = if (xo) "XONLY" else NULL, order_in = NULL, obs_req = 8L,
                            min_obs_stop = FALSE, noise_reduction = noise)
    key <- sprintf("part_%s_%d", noise, xo)
    q_ok <- identical(as.character(p$dt$quadrant), core[[paste0(key, "_quad")]])
    o_ok <- p$order == num(paste0(key, "_order"))
    rpx <- num(paste0(key, "_rpx")); rpy <- num(paste0(key, "_rpy"))
    rp_ok <- length(rpx) == nrow(p$regression.points) &&
      max(abs(c(rpx - p$regression.points$x, rpy - p$regression.points$y)), 0) < 1e-10
    res[key] <- sprintf("order %s quad %s rp %s", o_ok, q_ok, rp_ok)
  }
  s <- NNS:::NNS_seas_cpp(x)
  res["seas"] <- sprintf("periods %s cv %s best %s",
    identical(as.numeric(s$all.periods$Period), num("seas_periods")),
    isTRUE(all.equal(as.numeric(s$all.periods$Coefficient.of.Variation), num("seas_cv"), tolerance = 1e-10)),
    as.numeric(s$best.period) == num("seas_best"))
  res["fsd_d"] <- NNS:::NNS_FSD_uni_cpp(x, y, "discrete") == num("fsd_d")
  res["fsd_c"] <- NNS:::NNS_FSD_uni_cpp(x, y, "continuous") == num("fsd_c")
  res["ssd"] <- NNS:::NNS_SSD_uni_cpp(x, y) == num("ssd")
  res["tsd"] <- NNS:::NNS_TSD_uni_cpp(x, y) == num("tsd")
  res["fsd_yx"] <- NNS:::NNS_FSD_uni_cpp(y, x, "discrete") == num("fsd_yx_d")
  res["ssd_yx"] <- NNS:::NNS_SSD_uni_cpp(y, x) == num("ssd_yx")
  res["tsd_yx"] <- NNS:::NNS_TSD_uni_cpp(y, x) == num("tsd_yx")
  res["rescale"] <- isTRUE(all.equal(as.numeric(NNS.rescale(x, 5, 10)), num("rescale"), tolerance = 1e-12))
  cat("==", d, "\n"); print(res, quote = FALSE)
}
