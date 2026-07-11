import numpy as np

def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    image = np.array(image)
    h, w = image.shape

    output = np.zeros((new_h, new_w))

    y_scale = 0 if new_h == 1 else (h - 1) / (new_h - 1)
    x_scale = 0 if new_w == 1 else (w - 1) / (new_w - 1)

    for i in range(new_h):
        for j in range(new_w):
            srcy = i * y_scale
            srcx = j * x_scale

            y0 = int(np.floor(srcy))
            x0 = int(np.floor(srcx))
            dy = srcy - y0
            dx = srcx - x0

            y1 = min(y0 + 1, h - 1)
            x1 = min(x0 + 1, w - 1)

            output[i, j] = image[y0, x0] * (1 - dy) * (1 - dx) + image[y1, x0] * dy * (1 - dx) + image[y0, x1] * (1 - dy) * dx + image[y1, x1] * dy * dx 


    return output.tolist()