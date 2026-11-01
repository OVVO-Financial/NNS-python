suppressMessages(library(NNS)); library(jsonlite)
P11 <- as.matrix(read.csv("prices11.csv",row.names=1)); P5 <- as.matrix(read.csv("prices5.csv",row.names=1))
out <- list()
t0<-proc.time()[[3]]; out$p11_nl <- unname(as.matrix(NNS.norm(P11))); out$t11nl <- (proc.time()[[3]]-t0)*1000
out$p11_l <- unname(as.matrix(NNS.norm(P11, linear=TRUE)))
t0<-proc.time()[[3]]; out$p5_nl <- unname(as.matrix(NNS.norm(P5))); out$t5nl <- (proc.time()[[3]]-t0)*1000
out$p5_l <- unname(as.matrix(NNS.norm(P5, linear=TRUE)))
out$list_uneq <- NNS.norm(list(c(1,2,3,4,5,6,7), c(10,20,30,40,50,60), c(.5,.6,.7,.8,.9,1,1.1,1.2,1.3)))
out$list_eq <- unname(as.matrix(NNS.norm(list(P5[,1],P5[,2],P5[,3]))))
out$names11 <- colnames(NNS.norm(P11))
out$zero_mean <- unname(as.matrix(NNS.norm(cbind(a=c(-1,0,1), b=c(1,2,3)))))
out$dep11 <- unname(NNS.dep(P11)$Dependence)
write_json(out, "norm_r.json", digits=NA)
