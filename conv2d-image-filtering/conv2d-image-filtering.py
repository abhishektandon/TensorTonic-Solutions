import numpy as np
import math

def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    
    image = np.array(image)
    kernel = np.array(kernel)
    
    m, n = image.shape
    mk, nk = kernel.shape
    
    p = 2 * padding
    
    if padding !=0:
        z = np.zeros((m + p, n + p))
        z[padding:-padding, padding:-padding] = image
        image = z

    m_out = math.floor((m + p - mk)/stride) + 1
    n_out = math.floor((n + p - nk)/stride) + 1
    
    output = np.zeros((m_out, n_out))

    for i in range(m_out):
        for j in range(n_out):
            roi = image[i*stride: i*stride + mk, j*stride: j*stride + nk]
            output[i, j] = np.sum(kernel * roi)

    return output.tolist()