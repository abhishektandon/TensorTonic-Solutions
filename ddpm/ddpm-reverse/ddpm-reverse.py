import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    
    x_t = np.array(x_t)
    epsilon_pred = np.array(epsilon_pred)
    betas = np.array(betas)
    
    if z is not None:
        z = np.array(z)
    else:
        z = 0
    
    alpha_t = 1 - betas[t - 1]
    sigma_t = np.sqrt(betas[t - 1])
    alpha_t_bar = np.prod(1 - betas[:t])

    x_t_minus_1 = (x_t - ((1 - alpha_t) / (np.sqrt(1 - alpha_t_bar))) * epsilon_pred) / np.sqrt(alpha_t) + sigma_t * z

    return x_t_minus_1.tolist()