import numpy as np
import math

def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    
    pad_width = 1
    image = np.array(image)
    m, n = image.shape
    
    image = np.pad(image, pad_width=pad_width, mode='constant')
    k1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    k2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    ksize, _ = k1.shape

    m_out = math.floor(m + 2*pad_width - ksize) + 1
    n_out = math.floor(n + 2*pad_width - ksize) + 1
    
    gx = np.zeros((m_out, n_out))
    gy = np.zeros((m_out, n_out))
    
    for i in range(m_out):
        for j in range(n_out):
            roi = image[i:i+ksize, j:j+ksize]
            gx[i, j] = np.sum(roi * k1)
            gy[i, j] = np.sum(roi * k2)

    g = np.sqrt(gx**2 + gy**2)
    
    return g.tolist()