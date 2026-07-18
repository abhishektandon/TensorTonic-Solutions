import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.array(x)
    d = {}
    for i in x:
        if i not in d.keys():
            d[i] = 1
        else:
            d[i] = d[i] + 1

    min_k, max_v = np.inf, 0
    for k, v in d.items():
        if v > max_v:
            min_k = k 
            max_v = v 

    return np.mean(x), np.median(x), min_k
    