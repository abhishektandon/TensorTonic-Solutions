import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    betas = np.array(betas)
    one_minus_betas_t = 1 - betas
    alpha_bar = np.round(np.cumprod(one_minus_betas_t), 6)
    
    return alpha_bar

def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """

    x_0 = np.array(x_0)
    epsilon = np.array(epsilon)

    alpha_t_bar = get_alpha_bar(betas)[t-1]
    sq_alpha_t_bar = np.sqrt(alpha_t_bar)
    sq_one_minus_alpha_t_bar = np.sqrt(1 - alpha_t_bar)

    x_t = sq_alpha_t_bar * x_0 + sq_one_minus_alpha_t_bar * epsilon

    return x_t.tolist()