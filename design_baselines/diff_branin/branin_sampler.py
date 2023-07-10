import numpy as np 
import random 
from trainer import Branin

np.random.seed(42)
mean = np.array([3.141592, 2.275])
cov = np.array([[1.0, 0.0], [0.0, 1.0]])

X = np.random.multivariate_normal(mean=mean, cov=cov, size=(5500))

print(X.shape)
X_inside = X[(X[:, 0] > -5) & (X[:, 0] < 10) & (X[:, 1] > 0) & (X[:, 1] < 15)]
print(X_inside.shape)

branin = Branin(path="dataset/branin_gaussian_5k.p")
Ys = branin.predict(X_inside)

print(X_inside, Ys)

# a*f(x) + b
al = [1,2,3,4,5]
bl = [1,2,3,4,5]

import pickle as pkl
for i in range(len(al)):
    with open(f"dataset/branin_gaussian_5k_{al[i]}_{bl[i]}.p", "wb") as f:
        pkl.dump([X_inside, al[i]*Ys+bl[i]], f)
