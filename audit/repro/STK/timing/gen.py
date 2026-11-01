import numpy as np, pandas as pd, sys
n=int(sys.argv[1]); p=int(sys.argv[2]); rs=np.random.RandomState(n+p)
X=rs.normal(size=(n,p)); y=np.sin(X[:,0])+X[:,1]**2*0.5+0.3*X[:,2]*X[:,3]+0.3*rs.normal(size=n)
pd.DataFrame(X,columns=[f"v{j}" for j in range(p)]).to_csv(f"X_{n}_{p}.csv",index=False); pd.DataFrame({"y":y}).to_csv(f"y_{n}_{p}.csv",index=False)
