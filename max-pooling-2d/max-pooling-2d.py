import numpy as np 
import math 

def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    # Write code here
    image = np.array(X)
    m, n = image.shape
    m_out, n_out = m//pool_size, n//pool_size

    output = np.zeros((m_out, n_out))
    for i in range(m_out):
        for j in range(n_out):
            roi = image[i*pool_size:i*pool_size+pool_size, 
                        j*pool_size:j*pool_size+pool_size]
            output[i, j] = np.max(roi)

    return output.tolist()
    