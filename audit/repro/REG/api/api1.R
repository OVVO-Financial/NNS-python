suppressPackageStartupMessages(library(NNS))
mt <- read.csv("/home/user/audit/data/mtcars.csv")
cyl <- factor(c("four","six","eight")[match(mt$cyl, c(4,6,8))], levels = c("four","six","eight"))
r <- NNS.reg(data.frame(wt = mt$wt, cyl = cyl), mt$mpg, factor.2.dummy = FALSE, plot = FALSE)
cat("codes rows 1,3,5:", r$Fitted.xy$cyl[c(1,3,5)], "R2", r$R2, "\n")
xb <- (0:99) %% 3 == 0
r <- NNS.reg(xb, as.numeric(0:99), plot = FALSE, point.est = c(TRUE, FALSE)); cat("bool x R2", r$R2, "names", names(r$RPM), "pe", r$Point.est, "\n")
r <- NNS.reg(data.frame(g = rep(c("1","2","10"), 10), z = as.numeric(0:29)), as.numeric(0:29), plot = FALSE); cat("numstr names", names(r$RPM), "R2", r$R2, "\n")
X <- mt[, c("wt","hp")]; pe <- X[1:2, ]
cat("pe ok", NNS.reg(X, mt$mpg, point.est = pe, plot = FALSE)$Point.est, " perm", NNS.reg(X, mt$mpg, point.est = pe[, c("hp","wt")], plot = FALSE)$Point.est, "\n")
cat("names:", names(NNS.reg(data.frame("my col" = mt$wt, "2nd" = mt$hp, check.names = FALSE), mt$mpg, plot = FALSE)$RPM), "\n")
r <- tryCatch(NNS.reg(data.frame(g = c("a","b",NA,"a","c")[rep(1:5, 20)], z = as.numeric(1:100)), as.numeric(1:100), plot = FALSE), error = function(e) conditionMessage(e)); cat("NA cat:", if (is.character(r)) r else "no error", "\n")
