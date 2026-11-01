import numpy as np, json, subprocess
from nns._native import native_fn as nf
rng=np.random.default_rng(0); vecs=[]
for i in range(3000):
    n=int(rng.integers(4,300)); k=rng.integers(0,4)
    x = rng.normal(size=n) if k==0 else (rng.integers(0,5,n).astype(float) if k==1 else (np.round(rng.standard_t(2,size=n),2) if k==2 else rng.exponential(size=n)*1e3+1e6))
    vecs.append({'x':x.tolist(),'core':nf('gravity')(x,False),'exact':nf('gravity_exact')(x)})
json.dump(vecs, open('grav_vecs.json','w'))
print(subprocess.run(["Rscript","-e",'suppressMessages(library(NNS)); v<-jsonlite::fromJSON("grav_vecs.json", simplifyVector=FALSE); r<-vapply(v, function(e) NNS.gravity(unlist(e$x)), 1); core<-vapply(v, function(e) e$core, 1); ex<-vapply(v, function(e) e$exact, 1); cat("nns::gravity == R:", sum(core==r), "/", length(r), "  gravity_exact == R:", sum(ex==r), "/", length(r), "\\n")'],capture_output=True,text=True).stdout)
