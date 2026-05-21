import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Returns: float scalar KL divergence averaged over the batch
    """
    num = mu.shape[0]
    var = np.exp(log_var)
    kld = (-0.5 * np.sum(1 + log_var - mu * mu - var))/num
    return kld
