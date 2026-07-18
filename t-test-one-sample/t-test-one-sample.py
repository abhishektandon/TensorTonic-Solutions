import numpy as np
import math

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    n = len(x)
    x_mu = np.mean(x)
    std = math.sqrt(np.sum((x - x_mu) ** 2)/(n-1))
    t = math.sqrt(n) * (x_mu - mu0)/std
    return t