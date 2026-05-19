import numpy as np

def compute_ddpm_loss(x_0, betas, t_values, epsilon, epsilon_pred):
    """
    Returns: float scalar MSE loss between true noise and predicted noise
    """
    n = np.prod(list(np.array(x_0).shape))

    mse = np.sum((np.array(epsilon) - np.array(epsilon_pred))**2)/n
    
    return mse