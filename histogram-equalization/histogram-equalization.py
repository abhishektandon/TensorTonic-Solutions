import numpy as np

def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    # Write code here
    hist = np.zeros(256, dtype=int)
    image = np.array(image)
    h, w = image.shape
    n_pixels = h * w
    
    for row in image:
        for val in row:
            hist[val] += 1

    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0][0]

    mapping = np.round((cdf - cdf_min)/(n_pixels - cdf_min) * 255).astype(np.uint8)
    eq = mapping[image]
    
    return eq.tolist()