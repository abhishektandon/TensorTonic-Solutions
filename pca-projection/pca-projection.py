import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    
    x = np.array(X)
    n = x.shape[0]
    mu = np.mean(x, axis=0)
    
    x_cen = x - mu
    covar = (x_cen.T @ x_cen)/(n-1)
    eigvals, eigvecs = np.linalg.eig(covar)
    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx[:k]]
    x_proj = x_cen @ W

    return x_proj