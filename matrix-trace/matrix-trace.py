import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A = np.array(A)
    n = A.shape[0]
    mask = np.eye(n, dtype=bool)
    trace = np.sum(A[mask])
    return trace
    
