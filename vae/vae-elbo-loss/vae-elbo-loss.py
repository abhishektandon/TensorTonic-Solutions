import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    num = x.shape[0]
    var = np.exp(log_var)
    
    kl = (-0.5 * np.sum(1 + log_var - mu**2 - var))/num
    recon = np.sum((x - x_recon)**2)/num
    total = kl + recon

    dict = {
            "total": float(total), 
            "recon": float(recon), 
            "kl": float(kl)
           }
    
    return dict
    
