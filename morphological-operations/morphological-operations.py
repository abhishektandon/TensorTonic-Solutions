import math
import numpy as np

def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """
    
    image = np.array(image)
    m, n = image.shape
    kernel = np.array(kernel)
    ksize, _ = kernel.shape
    pad = ksize//2

    image = np.pad(image, pad, mode='constant')

    m_out = math.floor(m + 2*pad - ksize) + 1
    n_out = math.floor(n + 2*pad - ksize) + 1

    output = np.zeros((m_out, n_out))
    for i in range(m_out):
        for j in range(n_out):
            roi = image[i: i+ksize, j: j+ksize]
            
            if operation == 'erode':
                res = roi & kernel
                if np.all(roi[kernel == 1] == 1):
                    output[i, j] = 1
            elif operation == 'dilate':
                res = roi | kernel
                if np.any(roi[kernel == 1] == 1):
                    output[i, j] = 1


    return output.tolist()
    