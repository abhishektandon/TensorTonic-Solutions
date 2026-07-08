import numpy as np

def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    image = np.array(image)
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b 

    return gray.tolist()