import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x = np.array(x, dtype=int)
    y = np.array(y, dtype=int)
    man_dist = np.sum(np.abs(x - y))
    return int(man_dist)