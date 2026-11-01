suppressMessages(library(NNS)); set.seed(1)
ctrl <- c(rnorm(50), rnorm(950, 5)); trt <- rnorm(50)   # control has mean ~4.75; only first 50 obs look like treatment
r <- NNS.ANOVA(ctrl, trt, robust=TRUE, plot=FALSE)
cat(sprintf("base Certainty %.4f ; Robust Certainty Estimate %.4f [%.4f, %.4f]\n", r$Certainty, r$`Robust Certainty Estimate`, r$`Lower 95% CI`, r$`Upper 95% CI`))
t0<-proc.time()[[3]]; F <- as.matrix(read.csv("/home/user/audit/data/returns_daily_equity_etf.csv")[,-1])
NNS.ANOVA(F, plot=FALSE); cat("multi 11 assets ms", (proc.time()[[3]]-t0)*1000, "\n")
t0<-proc.time()[[3]]; NNS.ANOVA(F, pairwise=TRUE, plot=FALSE); cat("pairwise 11 assets ms", (proc.time()[[3]]-t0)*1000, "\n")
t0<-proc.time()[[3]]; for(i in 1:20) NNS.ANOVA(F[,1],F[,2], plot=FALSE); cat("binary w/ CI ms", (proc.time()[[3]]-t0)*1000/20, "\n")
t0<-proc.time()[[3]]; NNS.ANOVA(F[,1],F[,2], robust=TRUE, plot=FALSE); cat("robust ms", (proc.time()[[3]]-t0)*1000, "\n")
