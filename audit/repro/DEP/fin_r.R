suppressMessages(library(NNS)); X<-as.matrix(read.csv("/home/user/audit/work/DEP/fin5.csv"))
D<-NNS.dep(X); A<-NNS.dep(X,asym=TRUE); C<-NNS.caus(X[(nrow(X)-999):nrow(X),], tau=0); C3<-NNS.caus(X[(nrow(X)-999):nrow(X),], tau=3)
library(jsonlite); write(toJSON(list(cor=D$Correlation,dep=D$Dependence,acor=A$Correlation,adep=A$Dependence,caus=C,caus3=C3,pearson=cor(X)),digits=NA),"/home/user/audit/work/DEP/fin_r.json")
print(round(D$Correlation,3)); print(round(D$Dependence,3)); print(round(cor(X),3)); print(round(C,3))
