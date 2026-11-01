import numpy as np, nns
print("about to call nns_copula on empty arrays", flush=True)
nns.nns_copula(np.array([]), np.array([]))
print("survived")
