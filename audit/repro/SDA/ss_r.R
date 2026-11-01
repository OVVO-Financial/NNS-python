suppressMessages(library(NNS)); library(jsonlite)
R <- read.csv("/home/user/audit/data/returns_daily_equity_etf.csv")[,-1]
out <- list()
for (i in 1:ncol(R)) for (j in 1:ncol(R)) out[[paste(i,j)]] <- unlist(NNS.SS(R[,i],R[,j]))
set.seed(1); a <- sample(1:5,100,TRUE); b <- sample(1:5,77,TRUE)
out[["ties"]] <- unlist(NNS.SS(a,b)); write.csv(data.frame(a=c(a,rep(NA,0))), "ss_a.csv", row.names=FALSE); write.csv(data.frame(b=b), "ss_b.csv", row.names=FALSE)
out[["nan"]] <- unlist(NNS.SS(c(1,NA,3), c(2,NA)))
out[["inf"]] <- unlist(NNS.SS(c(1,Inf,3), c(-Inf,2,Inf)))
set.seed(2); ci <- NNS.SS(R$SPY[1:500], R$TLT[1:500], confidence.interval=TRUE, reps=199)
out[["ci"]] <- c(ci$p_star, ci$lower, ci$upper, mean(ci$boot_vals), sd(ci$boot_vals))
t0 <- proc.time()[[3]]; for (k in 1:50) NNS.SS(R$SPY, R$TLT); out[["t_ms"]] <- (proc.time()[[3]]-t0)/50*1000
x <- rnorm(1e6); y <- rnorm(1e6); t0 <- proc.time()[[3]]; NNS.SS(x,y); out[["t1e6_ms"]] <- (proc.time()[[3]]-t0)*1000
t0 <- proc.time()[[3]]; NNS.SS(R$SPY[1:1000], R$TLT[1:1000], confidence.interval=TRUE, reps=999); out[["tci_ms"]] <- (proc.time()[[3]]-t0)*1000
write_json(out, "ss_r.json", digits=NA)
