X <- as.matrix(read.csv("cases/boston_m12_pi/xtr.csv")); y <- read.csv("cases/boston_m12_pi/ytr.csv")[[1]]
mn <- apply(X,2,min); rg <- apply(X,2,max)-mn; D <- sweep(sweep(X,2,mn,"-"),2,rg,"/")
co <- sapply(1:ncol(D), function(j) cor(D[,j], y, method="spearman"))
writeLines(sprintf("%.17g", co), "dump/rcor.txt")
a <- rep(0,13); o <- order(abs(co), decreasing=TRUE, method="radix")[1:3]; a[o] <- co[o]
writeLines(sprintf("%.17g", as.numeric(D %*% a / 3)), "dump/rproj.txt")
writeLines(sprintf("%.17g", as.numeric(D)), "dump/rD.txt")
