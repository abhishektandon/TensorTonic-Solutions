import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.array(a)
    b = np.array(b)
    
    cos_sim = np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b) + 1e-15)

    return cos_sim