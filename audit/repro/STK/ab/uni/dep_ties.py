import numpy as np, subprocess
from nns import nns_dep
rs=np.random.RandomState(0); x0=rs.rand(60); y0=np.round(x0*2+rs.rand(60))+0.123456789
idx=rs.randint(0,60,120); x=x0[idx]; y=y0[idx]
np.savetxt("uni/dx.txt",x,fmt="%.17g"); np.savetxt("uni/dy.txt",y,fmt="%.17g")
print("py dup:", nns_dep(x,y)["Dependence"], " py nodup:", nns_dep(x0,y0)["Dependence"])
print(subprocess.run(["Rscript","-e",'suppressMessages(library(NNS));x<-scan("uni/dx.txt",quiet=T);y<-scan("uni/dy.txt",quiet=T);cat("R dup:",NNS.dep(x,y)$Dependence,"\n")'],capture_output=True,text=True).stdout)
np.savetxt("uni/dx0.txt",x0,fmt="%.17g"); np.savetxt("uni/dy0.txt",y0,fmt="%.17g")
print(subprocess.run(["Rscript","-e",'suppressMessages(library(NNS));x<-scan("uni/dx0.txt",quiet=T);y<-scan("uni/dy0.txt",quiet=T);cat("R nodup:",NNS.dep(x,y)$Dependence,"\n")'],capture_output=True,text=True).stdout)
