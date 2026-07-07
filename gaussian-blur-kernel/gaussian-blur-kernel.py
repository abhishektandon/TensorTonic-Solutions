import numpy as np

def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    # Write code here
    center = size//2
    kernel = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2)/ (2 * sigma**2))

    kernel = kernel/np.sum(kernel)

    return kernel.tolist()