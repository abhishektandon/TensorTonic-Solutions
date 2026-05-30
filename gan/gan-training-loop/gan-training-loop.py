import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def train_gan_step(real_data, fake_data, D_W):
    """
    Returns: dict with "d_loss" and "g_loss" as float values
    """
    D_W = np.asarray(D_W)
    real_data = np.asarray(real_data)
    fake_data = np.asarray(fake_data)

    real_probs = np.clip(sigmoid(real_data @ D_W), 1e-8, 1-1e-8)
    fake_probs = np.clip(sigmoid(fake_data @ D_W), 1e-8, 1-1e-8)

    d_loss = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
    g_loss = -np.mean(np.log(fake_probs))

    return {"d_loss": d_loss, "g_loss": g_loss}

    