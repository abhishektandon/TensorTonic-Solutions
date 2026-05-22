import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        matrix = np.array(matrix)
        h, w = matrix.shape
    except:
        return None
        
    if h!=w:
        return None
    evs = np.linalg.eig(matrix)
    
    return evs[0]
        