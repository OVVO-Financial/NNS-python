import numpy as np, nns
x=np.array([-0.85,-1.52,0.68]); y=np.array([-0.21,-0.33,-1.67]); z=np.array([-3,0.16,0.2])
def junk(k): return np.column_stack([np.array([-10.0-i,-9.0-i,-8.0-i]) for i in range(k)])
for k in (71,72):
    X=np.c_[x,y,z,junk(k)]
    print(X.shape[1],"columns: efficient set (deg 3) =", nns.sd_efficient_set(X,3), "cluster1 =", nns.nns_sd_cluster(X,degree=3)["Clusters"]["Cluster_1"][:3])
print("3 columns:", nns.sd_efficient_set(np.c_[x,y,z],3), " pair only:", nns.sd_efficient_set(np.c_[x,y],3), " tsd_uni(x,y)=", nns.tsd_uni(x,y))
