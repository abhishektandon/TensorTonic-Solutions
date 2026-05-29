import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    o1 = x@W1 + b1
    o1[o1 < 0] = 0
    x = o1 @ W2 + b2
    return x