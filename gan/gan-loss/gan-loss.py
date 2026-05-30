import numpy as np

def clip(probs):
    
    return np.clip(probs, 1e-8, 1-1e-8)

def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy.
    Returns: Loss value rounded to 4 decimals."""
    
    real_probs = clip(np.asarray(real_probs))
    fake_probs = clip(np.asarray(fake_probs))
    
    ld = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))    
    return np.round(ld, 4)

def generator_loss(fake_probs):
    """Compute non-saturating generator loss.
    Returns: Loss value rounded to 4 decimals."""
    
    fake_probs = clip(np.asarray(fake_probs))
    
    lg = -np.mean(np.log(fake_probs))

    return np.round(lg, 4)
