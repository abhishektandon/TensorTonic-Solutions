import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos_emb = np.zeros((seq_length, d_model))
    pos = np.arange(seq_length)
    
    for i in range(d_model):
        if i%2 == 0:
            pos_emb[:, i] = np.sin(pos / (1e4**(2*(i//2)/d_model)))
        else:
            pos_emb[:, i] = np.cos(pos / (1e4**(2*(i//2)/d_model)))

    return pos_emb