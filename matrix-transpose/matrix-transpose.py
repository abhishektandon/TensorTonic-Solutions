import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    h, w = A.shape
    transpose = np.zeros((w, h))
    for idx, i in enumerate(A):
        transpose[:, idx] = i

    return transpose
        