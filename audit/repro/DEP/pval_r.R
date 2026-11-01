suppressMessages(library(NNS)); set.seed(42)
reps <- 200; pc <- pd <- numeric(reps); obs_below <- 0
for (r in 1:reps){ x<-rnorm(100); y<-rnorm(100); o <- NNS.dep(x,y,p.value=TRUE); pc[r]<-o[["Correlation p.value"]]; pd[r]<-o[["Dependence p.value"]] }
cat("Null: P(cor p<0.05) =", mean(pc<0.05), " P(dep p<0.05) =", mean(pd<0.05), " P(dep p==0) =", mean(pd==0), "\n")
# strongly dependent y = x^2 : p-value for Dependence
x<-rnorm(100); o<-NNS.dep(x,x^2,p.value=TRUE); str(o)
# show that Dependence p.value is small when observed dep is LOWER than permutations: perfect anti-dep? use constant-ish
