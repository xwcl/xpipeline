from typing import Tuple
from functools import partial
import numpy as np
import logging
from numpy.core.numeric import count_nonzero
from scipy.ndimage import binary_dilation
import skimage.transform
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel

from .. import core
cp = core.cupy
da = core.dask_array
torch = core.torch

log = logging.getLogger(__name__)


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
    good_pix_mask = good_pix_mask == 1
    yy, xx = np.indices(cube.shape[1:])
    x_indices = xx[good_pix_mask]
    y_indices = yy[good_pix_mask]
    cube_to_vecs = lambda cube, good_pix_mask: cube[:,good_pix_mask].T
    if xp is da:
        chunk_size = cube.shape[0] // cube.numblocks[0]
        # import IPython
        # IPython.embed()
        image_vecs = cube.map_blocks(
            cube_to_vecs,
            good_pix_mask,
            dtype=cube.dtype,
            chunks=(chunk_size, np.count_nonzero(good_pix_mask == 1),),
            drop_axis=2
        )
    else:
        image_vecs = cube_to_vecs(cube, good_pix_mask)
    return image_vecs, x_indices, y_indices


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
    cube, x_indices, y_indices = unwrap_cube(image[core.newaxis, :, :], good_pix_mask)
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
    if xp is da:
        chunk_size = matrix.shape[0] // matrix.numblocks[0]
        return matrix.map_blocks(
            wrap_matrix,
            shape,
            x_indices,
            y_indices,
            chunks=(chunk_size,) + shape[1:],
            new_axis=2,
            dtype=matrix.dtype
        )
    if matrix.shape[1] != shape[0]:
        # handling a chunk from map_blocks. Since matrix is transposed before assignment
        # we check that the number of *columns* is equal to the expected number of planes in the cube
        shape = (matrix.shape[1],) + shape[1:]
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
    matrix = image_vec[:, core.newaxis]
    cube = wrap_matrix(matrix, (1,) + shape, x_indices, y_indices)
    return cube[0]

def quick_derotate(cube, angles):
    outimg = np.zeros(cube.shape[1:])

    for i in range(cube.shape[0]):
        image = cube[i]
        if core.get_array_module(image) is da:
            image = image.get()
        outimg += skimage.transform.rotate(image, -angles[i])

    return outimg

def mask_arc(center: Tuple[float, float],
             data_shape: Tuple[int, int],
             from_radius: float, to_radius: float,
             from_radians: float, to_radians: float,
             overall_rotation_radians: float=0) -> np.ndarray:
    '''Mask an arc beginning `from_radius` pixels from `center`
    and going out to `to_radius` pixels, beginning at `from_radians`
    from the +X direction (CCW when 0,0 at lower left) and going
    to `to_radians`. For cases where it's easier to adjust the overall
    rotation than the bounds, `overall_rotation_radians` can be set to
    offset the `from_radians` and `to_radians` values

    Parameters
    ----------
    center: tuple[float, float]
        x, y pixel coordinates of the center of the grid
    data_shape: tuple[int, int]
        height, width shape (Python / NumPy order)
    from_radius: float
        pixel distance from center where mask `True` region
        should start
    to_radius: float
        pixel distance from center where mask `True` region
        should end
    from_radians: float
        angle in radians from +X where mask `True` region
        should start
    to_radians: float
        angle in radians from +X where mask `True` region
        should end
    overall_rotation_radians: float (default: 0)
        amount to rotate coordinate grid from +X
    '''
    rho, phi = polar_coords(center, data_shape)
    phi = (phi + overall_rotation_radians) % (2 * np.pi)
    mask = (from_radius <= rho) & (rho <= to_radius)
    from_radians %= (2 * np.pi)
    to_radians %= (2 * np.pi)
    if from_radians != to_radians:
        mask &= (from_radians <= phi) & (phi <= to_radians)
    return mask

def cartesian_coords(center: Tuple[float, float],
                     data_shape: Tuple[int, int]) -> np.ndarray:
    '''center in x,y order; data_shape in (h, w); returns coord arrays xx, yy of data_shape

    Matrix layout ((0,0) at upper left) for (2, 2) matrix

       ----------+X---------->

    |  +---------+ +---------+
    |  |  [0,0]  | |  [0,1]  |
   +Y  +---------+ +---------+
    |  +---------+ +---------+
    |  |  [1,0]  | |  [1,1]  |
    V  +---------+ +---------+

    With sample locations defined as the centers of pixels, [0,0]
    represents the value from x = -0.5 to x = 0.5;
    [0,1] from x = 0.5 to x = 1.5.

    Placing the center at (0.5, 0.5) puts it at the corner of all four
    pixels. The coordinates of the centers in the new system are then

       ---------------+X---------------->

    |  +--------------+ +--------------+
    |  | (-0.5, -0.5) | |  (0.5, -0.5) |
   +Y  +--------------+ +--------------+
    |  +--------------+ +--------------+
    |  | (-0.5, 0.5)  | |  (0.5, 0.5)  |
    |  +--------------+ +--------------+
    V

    Which means coordinate arrays of

    xx = [[-0.5, 0.5], [-0.5, 0.5]]
    yy = [[-0.5, -0.5], [0.5, 0.5]]
    '''
    yy, xx = np.indices(data_shape, dtype=float)
    center_x, center_y = center
    yy -= center_y
    xx -= center_x
    return xx, yy


def polar_coords(center: Tuple[float, float],
                 data_shape: Tuple[int, int]) -> np.ndarray:
    '''center in x,y order; data_shape in (h, w); returns coord arrays rho, phi of data_shape'''
    xx, yy = cartesian_coords(center, data_shape)
    rho = np.sqrt(yy**2 + xx**2)
    phi = np.arctan2(yy, xx)
    return rho, phi


def max_radius(center: Tuple[float, float],
               data_shape: Tuple[int, int]) -> float:
    '''Given an (x, y) center location and a data shape of
    (height, width) return the largest circle radius from that center
    that is completely within the data bounds'''
    if center[0] > (data_shape[1] - 1) or center[1] > (data_shape[0] - 1):
        raise ValueError("Coordinates for center are outside data_shape")
    bottom_left = np.sqrt(center[0]**2 + center[1]**2)
    data_height, data_width = data_shape
    top_right = np.sqrt(
        (data_height - center[0])**2 + (data_width - center[1])**2)
    return min(bottom_left, top_right)
