import numpy as np
import math 

def layer_norm(inp: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    
    x = np.array(inp, ndmin=3)
    gamma = np.array(gamma, ndmin=3)
    beta = np.array(beta, ndmin=3)

    mu = np.mean(x, axis=-1).T
    var = (np.std(x, axis=-1) ** 2).T

    lyr_norm = gamma * (x - mu)/np.sqrt(var + eps) + beta

    if inp.ndim == 2:
        lyr_norm = np.squeeze(lyr_norm, axis=0)
    
    return lyr_norm