import numpy as np
Y = [1,2,3]
P = [4,5,6]
Y = np.array([1,0,1])
P = np.array([0.2,0.6,0.8])
np.float_(Y)
# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
import numpy as np

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
