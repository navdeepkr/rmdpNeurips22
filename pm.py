import numpy as np
import matplotlib.pyplot as plt

# parameters
def kernel(S,A):
    p = np.random.rand(S,A,S)
    for s in range(S):
        for a in range(A): 
            summ = np.sum(p[s,a])
            p[s,a] = p[s,a]/summ
    return p
   
S = 100
A = 20
R = np.random.randn(S,A)
v0 = np.random.randn(S)
gamma = 0.9
alphaSA = 0.1*np.ones((S,A))
betaSA = 0.1*np.ones((S,A))
alphaS = 0.1*np.ones(S)
betaS = 0.1*np.ones(S)
n =100
P = kernel(S,A)

