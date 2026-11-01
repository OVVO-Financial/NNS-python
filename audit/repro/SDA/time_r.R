suppressMessages(library(NNS))
set.seed(1)
tm <- function(expr, reps) { t0 <- proc.time()[["elapsed"]]; for (i in 1:reps) force(eval(expr)); (proc.time()[["elapsed"]]-t0)/reps*1000 }
for (n in c(1e3,1e4,3e4,1e5)) { y <- rnorm(n); x <- y + 0.1
  reps <- if (n==1e5) 3 else 20
  cat(sprintf("n=%g FSD %.2f SSD %.2f TSD %.2f FSDc %.2f | FSD.uni %.3f SSD.uni %.3f TSD.uni %.3f ms\n", n,
   tm(quote(NNS.FSD(x,y,plot=FALSE)),reps), tm(quote(NNS.SSD(x,y,plot=FALSE)),reps), tm(quote(NNS.TSD(x,y,plot=FALSE)),reps),
   tm(quote(NNS.FSD(x,y,type="continuous",plot=FALSE)),reps),
   tm(quote(NNS.FSD.uni(x,y)),reps), tm(quote(NNS.SSD.uni(x,y)),reps), tm(quote(NNS.TSD.uni(x,y)),reps)))
}
R <- read.csv("/home/user/audit/data/returns_daily_equity_etf.csv"); X <- as.matrix(R[,-1])
for (d in 1:3) cat(sprintf("eff set 11 assets n=%d deg %d: %.2f ms\n", nrow(X), d, tm(quote(NNS.SD.efficient.set(X,d,status=FALSE)),20)))
cat(sprintf("eff set deg1 continuous: %.2f ms\n", tm(quote(NNS.SD.efficient.set(X,1,type="continuous",status=FALSE)),20)))
cat(sprintf("SD.cluster deg 2: %.2f ms\n", tm(quote(NNS.SD.cluster(X,degree=2)),10)))
Y <- matrix(rnorm(1000*200, rep(rnorm(200,0,.05),each=1000)), 1000, 200)
for (d in 1:3) cat(sprintf("eff set 200 cols n=1000 deg %d: %.2f ms\n", d, tm(quote(NNS.SD.efficient.set(Y,d,status=FALSE)),5)))
write.csv(Y, "Y200.csv", row.names=FALSE)
