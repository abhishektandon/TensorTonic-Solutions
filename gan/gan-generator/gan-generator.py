import numpy as np

def exp(x):
    return np.exp(x)

def generator(z, W, b):
    """
    Returns: np.ndarray of shape (batch, output_dim) with tanh-activated values rounded to 4 decimals
    """
    
    z = np.asarray(z)
    W = np.asarray(W)
    b = np.asarray(b)
    
    x = z @ W + b
    gen = (exp(x) - exp(-x))/(exp(x) + exp(-x))    
    return gen