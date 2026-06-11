import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.array(X, ndmin=2)
    
    n, m = X.shape
    if n<2:
        return None
        
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    covar = (X_centered.T @ X_centered)/(n-1)
    
    return covar