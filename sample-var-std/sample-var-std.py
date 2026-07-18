import numpy as np
import math

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    n = len(x)
    x = np.array(x)
    mu = np.mean(x)

    var = np.sum((x - mu) ** 2)/(n-1)
    std = math.sqrt(var)
    
    return var, std