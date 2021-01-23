
import numpy as np
from scipy.ndimage import binary_dilation
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel

def gaussian_smooth(data, kernel_stddev_px):
    return convolve_fft(
        data,
        Gaussian2DKernel(kernel_stddev_px),
        boundary='wrap'
    )

def rough_peak_in_box(data, initial_guess, box_size):
    '''
    Parameters
    ----------
    data : ndarray
    initial_guess : (init_y, init_x) int
    box_size : (box_height, box_width) int
    
    Returns
    -------
    location : (y, x) int
        pixel location in original array coordinates/indices
        if found, `initial_guess` otherwise
    found : bool
        Whether the peak pixel was interior to the box
        or lay on a border (indicating we didn't have a
        feature inside)
    '''
    height, width = data.shape
    box_height, box_width = box_size
    init_y, init_x = initial_guess
    max_y = height - 1
    max_x = width - 1
    
    x_start, x_end = max(0, init_x - box_width // 2), min(max_x, init_x + box_width // 2)
    y_start, y_end = max(0, init_y - box_height // 2), min(max_y, init_y + box_height // 2)
    cutout = data[y_start:y_end,x_start:x_end]

    y, x = np.unravel_index(np.argmax(cutout), cutout.shape)
    found = True
    if y == 0 or y == cutout.shape[0] - 1:
        found = False
    if x == 0 or x == cutout.shape[1] - 1:
        found = False

    if found:
        true_x, true_y = x_start + x, y_start + y
        return (true_y, true_x), found
    else:
        return initial_guess, False
