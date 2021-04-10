
import numpy as np
from scipy.ndimage import binary_dilation
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel

from .. import core


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
    cutout = data[y_start:y_end, x_start:x_end]

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


def unwrap_cube(cube, good_pix_mask):
    '''Unwrap a shape (planes, m, n) `cube` and transpose into
    a (pix, planes) matrix, where `pix` is the number of *True*
    entries in a (m, n) `mask` (i.e. False entries are removed)

    Parameters
    ----------
    cube : array (planes, m, n)
    good_pix_mask : array (m, n)
        Pixels to include in `matrix`

    Returns
    -------
    matrix : array (pix, planes)
        Vectorized images, one per column
    x_indices : array (pix,)
        The x indices into the original image that correspond
        to each entry in the vectorized image
    y_indices : array (pix,)
        The y indices into the original image that correspond
        to each entry in the vectorized image
    '''
    xp = core.get_array_module(cube)
    yy, xx = xp.indices(cube.shape[1:])
    x_indices = xx[good_pix_mask]
    y_indices = yy[good_pix_mask]
    return cube[:, good_pix_mask].T, x_indices, y_indices


def unwrap_image(image, good_pix_mask):
    '''Unwrap a shape (m, n) `image` and transpose into a (pix,)
    vector, where `pix` is the number of *True* entries in a (m, n)
    `mask` (i.e. False entries are removed)

    Parameters
    ----------
    image : array (m, n)
    good_pix_mask : array (m, n)
        Pixels to include in `vector`

    Returns
    -------
    vector : array (pix,)
        Vectorized image
    x_indices : array (pix,)
        The x indices into the original image that correspond
        to each entry in the vectorized image
    y_indices : array (pix,)
        The y indices into the original image that correspond
        to each entry in the vectorized image
    '''
    xp = core.get_array_module(image)
    cube, x_indices, y_indices = unwrap_cube(image[xp.newaxis, :, :], good_pix_mask)
    return cube[:, 0], x_indices, y_indices


def wrap_matrix(matrix, shape, x_indices, y_indices):
    '''Wrap a (planes, pix) matrix into a shape `shape`
    data cube, where pix is the number of entries in `x_indices`
    and `y_indices`

    Parameters
    ----------
    matrix
    shape
    x_indices
    y_indices

    Returns
    -------
    cube
    '''
    xp = core.get_array_module(matrix)
    cube = xp.zeros(shape)
    cube[:, y_indices, x_indices] = matrix.T
    return cube


def wrap_vector(image_vec, shape, x_indices, y_indices):
    '''Wrap a (pix,) vector into a shape `shape` image,
    where pix is the number of entries in `x_indices`
    and `y_indices`

    Parameters
    ----------
    vector
    shape
    x_indices
    y_indices

    Returns
    -------
    vector
    '''
    xp = core.get_array_module(image_vec)
    matrix = image_vec[:, xp.newaxis]
    cube = wrap_matrix(matrix, (1,) + shape, x_indices, y_indices)
    return cube[0]
