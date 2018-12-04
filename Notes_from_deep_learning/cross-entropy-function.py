import numpy as np
Y = [1,2,3]
P = [4,5,6]
Y = np.array([1,0,1])
P = np.array([0.2,0.6,0.8])
# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    sum ((Y*-np.log(P)) + ((1-Y)*np.log(1-P)))
    pass
