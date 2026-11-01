suppressMessages(library(NNS))
rpm <- as.matrix(read.csv("mp_rpm.csv", header=FALSE)); y <- scan("mp_y.csv", quiet=TRUE); xt <- as.matrix(read.csv("mp_xt.csv", header=FALSE))
mins <- apply(rpm,2,min); maxs <- apply(rpm,2,max)
t <- system.time(P <- NNS:::NNS_mreg_predict_path_v2_cpp(rpm, y, xt, 30L, 0L, mins, maxs, FALSE, 1L))[["elapsed"]]
t2 <- system.time(S <- NNS:::NNS_mreg_predict_v2_cpp(rpm, y, xt, 12L, 0L, mins, maxs, FALSE, 1L))[["elapsed"]]
Pp <- as.matrix(read.csv("mp_py_path.csv", header=FALSE)); Sp <- scan("mp_py_single.csv", quiet=TRUE)
cat(sprintf("R path %.3fs single %.3fs ; max|diff| path %.3g single %.3g\n", t, t2, max(abs(P-Pp)), max(abs(S-Sp))))
