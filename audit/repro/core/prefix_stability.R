suppressMessages(library(NNS))
set.seed(1)
direct_upm <- function(d,t,x) sapply(t, function(tt) mean(pmax(x-tt,0)^d))
direct_lpm <- function(d,t,x) sapply(t, function(tt) mean(pmax(tt-x,0)^d))
# Case A: targets near the max, degree 2 (upper tail)
x <- rnorm(1e4, mean=100, sd=1)
t <- sort(x)[(1e4-99):1e4] - 1e-9   # 100 targets just below top order stats
u <- UPM(2, t, x); ud <- direct_upm(2, t, x)
cat("A deg2 UPM near max: min(R)=", min(u), " max rel err=", max(abs(u-ud)/pmax(ud,1e-300)), " any<0:", any(u<0), "\n")
# Case B: degree 3..8 heavy tails
r <- read.csv("/home/user/audit/data/returns_daily_equity_etf.csv")
y <- na.omit(r[[2]])
tt <- quantile(y, seq(0.001,0.999,length.out=200))
for (d in c(3,4,6,8,12)) {
  l <- LPM(d, tt, y); ld <- direct_lpm(d, tt, y)
  uu <- UPM(d, tt, y); udd <- direct_upm(d, tt, y)
  cat(sprintf("B deg%d returns: LPM max rel err=%.3g (neg:%d)  UPM max rel err=%.3g (neg:%d)\n", d,
     max(abs(l-ld)/pmax(abs(ld),1e-300)), sum(l<0), max(abs(uu-udd)/pmax(abs(udd),1e-300)), sum(uu<0)))
}
# Case C: price levels (offset) degree 4 with targets in far tail
p <- read.csv("/home/user/audit/data/prices_daily.csv")
z <- na.omit(p$SPY)
tz <- seq(min(z)-50, max(z)+50, length.out=100)
for (d in c(2,4,8)) {
  l <- LPM(d, tz, z); ld <- direct_lpm(d, tz, z)
  uu <- UPM(d, tz, z); udd <- direct_upm(d, tz, z)
  cat(sprintf("C deg%d SPY: LPM max rel err=%.3g (neg:%d)  UPM max rel err=%.3g (neg:%d)\n", d,
     max(abs(l-ld)/pmax(abs(ld),1e-300)), sum(l<0), max(abs(uu-udd)/pmax(abs(udd),1e-300)), sum(uu<0)))
}
# scalar vs vector consistency
cat("scalar vs vector (deg 4, SPY, target 5th):", LPM(4, tz[5], z), LPM(4, tz, z)[5], "\n")
