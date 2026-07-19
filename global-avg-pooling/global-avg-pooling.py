import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.array(x)
    if x.ndim < 3:
        raise ValueError()
    *dims, h, w = x.shape
    x = x.reshape(*dims, h * w)
    gap = np.mean(x, axis=-1)

    return gap
    