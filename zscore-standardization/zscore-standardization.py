import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    x = np.array(X)
    mu = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    if axis == 1:
        mu = mu[:, np.newaxis]
        
    z = (x - mu)/(std + eps)
    
    return z