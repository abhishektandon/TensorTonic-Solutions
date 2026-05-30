import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def discriminator(x, W):
    """
    Returns: np.ndarray of shape (batch, 1) with probabilities rounded to 4 decimals
    """
    x = np.asarray(x)
    W = np.asarray(W)

    prob = sigmoid(x @ W)

    return prob

    