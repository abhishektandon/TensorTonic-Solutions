import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    A = np.array(A)
    h, w = A.shape
    if h!=w:
        return None
    elif np.linalg.det(A) == 0:
        return None

    return np.linalg.inv(A)