import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def generate(self, z):
        """
        Generate fake samples from noise z using tanh(z @ G_W).
        Returns list of lists, rounded to 4 decimals.
        """
        gen = np.round(self.tanh(z @ self.G_W), 4)
        
        return gen
    
    def discriminate(self, x):
        """
        Classify samples using sigmoid(x @ D_W).
        Returns list of lists, rounded to 4 decimals.
        """
        x = x @ self.D_W
        out = np.round(self.sigmoid(x), 4)

        return out
    
    def train_step(self, real_data, z):
        """
        Compute d_loss and g_loss for one training step.
        Returns dict with "d_loss" and "g_loss", rounded to 4 decimals.
        """
        
        real_probs = np.clip(self.discriminate(real_data), 1e-8, 1-1e-8)
        fake_probs = np.clip(self.discriminate(self.generate(z)), 1e-8, 1-1e-8)
        d_loss = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
        g_loss = -np.mean(np.log(fake_probs))
        
        return {"d_loss":d_loss, "g_loss":g_loss}
        