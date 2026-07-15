import numpy as np

def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    hist = np.zeros(256, dtype=int)
    image = np.array(image)
    
    for i in image:
        for j in i:
            hist[j] = hist[j] + 1

    return hist.tolist()