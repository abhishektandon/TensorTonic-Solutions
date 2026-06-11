import numpy as np
import math

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x = np.array(x)
    y = np.array(y)
    euc_dist = math.sqrt(np.sum((x - y) ** 2))
    return euc_dist