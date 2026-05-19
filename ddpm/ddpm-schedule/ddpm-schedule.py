import numpy as np

def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    """
    Linear noise schedule from beta_1 to beta_T.
    Returns list of floats rounded to 6 decimals.
    """
    betas_linear = np.round([beta_1 + (t-1)/(T-1 + 1e-10) * (beta_T - beta_1) for t in range(1, T+1)], 6)
    
    return betas_linear

def cosine_alpha_bar_schedule(T, s=0.008):
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """

    f_t = np.array(np.round([np.power(np.cos((t/T + s)/(1 + s) * (np.pi/2)), 2) for t in range(0, T+1)], 6))
    alpha_t_bar_cosine = f_t/f_t[0]
    alpha_t_bar_cosine = np.round(np.clip(alpha_t_bar_cosine, 0.0001, 0.9999), 6)

    return alpha_t_bar_cosine[1:]
    

def alpha_bar_to_betas(alpha_bars):
    """
    Convert alpha_bar schedule to beta schedule.
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """

    a_0 = [1.0]
    alpha_bars = np.array(alpha_bars)
    alpha_bars_shifted = np.array(np.append(a_0, alpha_bars))
    betas = np.clip(1 - alpha_bars/alpha_bars_shifted[:-1], 0.0001, 0.9999)

    betas = np.round(betas, 6)
    
    return betas
