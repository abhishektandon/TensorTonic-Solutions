import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    x = np.array(X)
    std = np.std(x, axis=0)
    mean = np.mean(x, axis=0)
    x_c = x - mean
    covar = (x_c.T @ x_c)/x.shape[0] # population covariance
    corr_mtx = covar/np.outer(std, std)
    return corr_mtx