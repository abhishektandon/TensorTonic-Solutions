import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    x = np.array(matrix)
    
    if x.ndim != 2:
        return None

    if norm_type not in ["l1", "l2", "max"]:
        return None

    if axis not in [0, 1, -1, None]:
        return None
    
    if norm_type == "l2":
        n = np.sqrt(np.sum(x ** 2, axis=axis))
    elif norm_type == "l1":
        n = np.sum(np.abs(x), axis=axis)
    else:
        n = np.max(np.abs(x), axis=axis)

    if axis == 1:
        n = n[:, np.newaxis]
        
    return np.divide(x, n+1e-15)
    