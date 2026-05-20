import numpy as np

def ddpm_sample(x_T, betas, epsilon_preds, z_values):
    """
    Returns: np.ndarray of the final denoised sample
    """
    
    epsilon_preds = np.array(epsilon_preds)
    z_values = np.array(z_values)
    betas = np.array(betas)
    
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    T = len(betas)
    x_t = np.array(x_T)
    
    for t in range(T-1, -1, -1):
        print(t)
        if t > 0:
            z = z_values[T-t-1]
        else:
            z = np.zeros_like(x_t)

        one_by_sq_alpha_t = 1/np.sqrt(alphas[t])
        one_minus_alpha_t = 1 - alphas[t]
        sq_one_minus_alpha_t_bar = np.sqrt(1 - alpha_bars[t])
        
        x_t_minus_1 = one_by_sq_alpha_t * (x_t - one_minus_alpha_t/sq_one_minus_alpha_t_bar * epsilon_preds[T-t-1]) + np.sqrt(betas[t]) * z
        
        x_t = x_t_minus_1

    return x_t