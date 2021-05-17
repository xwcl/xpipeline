import warnings
from typing import Tuple, Union, List
from functools import partial
import numpy as np
import logging
from numpy.core.numeric import count_nonzero
from scipy.ndimage import binary_dilation
import skimage.transform
import skimage.registration
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel
from dataclasses import dataclass
import distributed.protocol
from .. import core

cp = core.cupy
da = core.dask_array
torch = core.torch

log = logging.getLogger(__name__)


def gaussian_smooth(data, kernel_stddev_px):
    return convolve_fft(data, Gaussian2DKernel(kernel_stddev_px), boundary="wrap")


def center(arr_or_shape):
    """Center coordinates for a 2D image (or shape) using the
    convention that indices are the coordinates of the centers
    of pixels, which run from (idx - 0.5) to (idx + 0.5)
    """
    shape = getattr(arr_or_shape, "shape", arr_or_shape)
    if len(shape) != 2:
        raise ValueError("Only do this on 2D images")
    return (shape[1] - 1) / 2, (shape[0] - 1) / 2


def rough_peak_in_box(data, initial_guess, box_size):
    """
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
    """
    height, width = data.shape
    box_height, box_width = box_size
    init_y, init_x = initial_guess
    max_y = height - 1
    max_x = width - 1

    x_start, x_end = max(0, init_x - box_width // 2), min(
        max_x, init_x + box_width // 2
    )
    y_start, y_end = max(0, init_y - box_height // 2), min(
        max_y, init_y + box_height // 2
    )
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


def _dask_ndcube_to_rows(ndcube_list, good_pix_mask):
    parts = []
    if isinstance(ndcube_list[0], list):
        for ndcube_list_entry in ndcube_list:
            parts.append(_dask_ndcube_to_rows(ndcube_list_entry, good_pix_mask))
    else:
        for ndcube in ndcube_list:
            parts.append(ndcube[:, good_pix_mask])
    return np.concatenate(parts)


def unwrap_cube(ndcube, good_pix_mask):
    """Unwrap a shape (planes, *idxs) `ndcube` and transpose into
    a (pix, planes) matrix, where `pix` is the number of *True*
    entries in a (*idxs) `mask` (i.e. False entries are removed)

    Parameters
    ----------
    ndcube : array (planes, *idxs)
        N-dimensional sequence where *idxs are the last (N-1) dimensions
    good_pix_mask : array (*idxs)
        Indices from a single slice to include in `matrix`, matching
        last (N-1) dimensions of ndcube

    Returns
    -------
    matrix : array (pix, planes)
        Vectorized images, one per column
    subset_idxs : array of shape (N-1, pix)
        The z, y, x (etc) indices into the original image that correspond
        to each entry in the vectorized image
    """
    if len(good_pix_mask.shape) != len(ndcube.shape) - 1:
        raise ValueError(
            f"To mask a {len(ndcube.shape)}-D cube, {len(ndcube.shape) - 1}-D masks are needed"
        )
    xp = core.get_array_module(ndcube)
    good_pix_mask = good_pix_mask == 1
    n_good_pix = np.count_nonzero(good_pix_mask)

    def ndcube_to_rows(cube, good_pix_mask):
        res = cube[:, good_pix_mask]
        return res

    if xp is da:
        # chunk_size = ndcube.shape[0] // ndcube.numblocks[0]
        # axes_to_drop = tuple(range(2, len(ndcube.shape)))
        # image_vecs = ndcube.map_blocks(
        #     ndcube_to_rows,
        #     good_pix_mask,
        #     dtype=ndcube.dtype,
        #     chunks=(chunk_size, n_good_pix,),
        #     drop_axis=axes_to_drop
        # )
        extra_axes = tuple(range(2, 2 + len(ndcube.shape[1:])))
        image_vecs = da.blockwise(
            _dask_ndcube_to_rows,
            (0, 1),
            ndcube,
            (0,) + extra_axes,
            good_pix_mask,
            None,
            new_axes={1: n_good_pix},
            dtype=ndcube.dtype,
        )
        image_vecs = image_vecs.T
    else:
        image_vecs = ndcube_to_rows(ndcube, good_pix_mask).T

    all_idxs = np.indices(ndcube.shape[1:])
    subset_idxs = all_idxs[:, good_pix_mask]
    return image_vecs, subset_idxs


def unwrap_image(image, good_pix_mask):
    """Unwrap a shape (*idxs) `image` and transpose into a (pix,)
    vector, where `pix` is the number of *True* entries in
    `good_pix_mask` (i.e. False entries are removed)

    Parameters
    ----------
    image : array (*idxs)
    good_pix_mask : array (*idxs)
        Pixels to include in `vector`

    Returns
    -------
    vector : array (pix,)
        Vectorized image
    subset_idxs : array of shape (N-1, pix)
        The z, y, x (etc) indices into the original image that correspond
        to each entry in the vectorized image
    """
    xp = core.get_array_module(image)
    indexer = (core.newaxis,) + tuple(slice(None, None) for _ in image.shape)
    cube, subset_idxs = unwrap_cube(image[indexer], good_pix_mask)
    return cube[:, 0], subset_idxs


def _dask_wrap_matrix(blocks, shape, subset_idxs, fill_value):
    # Because the columns axis is removed and two new axes added instead
    # the function invoked by blockwise() gets a list of arrays rather than
    # a single array
    res = []
    for block in blocks:
        output = np.ones((block.shape[0],) + shape[1:]) * fill_value
        indexer = (slice(None, None),) + tuple(x for x in subset_idxs)
        output[indexer] = block
        res.append(output)
    return np.concatenate(res)


def wrap_matrix(matrix, shape, subset_idxs, fill_value=np.nan):
    """Wrap a (N, pix) matrix into a shape `shape`
    data cube using the indexes from `subset_idxs`

    Parameters
    ----------
    matrix : array (N, pix)
    shape : tuple of length dims + 1
    subset_idxs : array (dims,) + dims * (pix,)
        pixel indices to map each vector entry to
        for each of `dims` dimensions
    fill_value : float
        default value for pixels without corresponding
        `subset_idxs` entries

    Returns
    -------
    cube : array of shape ``shape``
    """
    xp = core.get_array_module(matrix)
    if xp is da:
        plane_shape = shape[1:]

        new_axes_idxs = tuple(range(2, 2 + len(plane_shape)))
        new_axes = {k: plane_shape[idx] for idx, k in enumerate(new_axes_idxs)}
        log.debug(f"{new_axes=}")
        log.debug(f"{matrix.T.shape=}")
        result = da.blockwise(
            _dask_wrap_matrix,
            (0,) + new_axes_idxs,
            matrix.T,
            (0, 1),
            shape,
            None,
            subset_idxs,
            None,
            fill_value=fill_value,
            new_axes=new_axes,
            dtype=matrix.dtype,
        )
        return result
    cube = fill_value * xp.ones(shape)
    indexer = (slice(None, None),) + tuple(x for x in subset_idxs)
    cube[indexer] = matrix.T
    return cube


def wrap_vector(image_vec, shape, subset_idxs, fill_value=np.nan):
    """Wrap a (pix,) vector into a shape `shape` image using the
    indexes from `subset_idxs`

    Parameters
    ----------
    image_vec : array (pix,)
    shape : tuple of length dims
    subset_idxs : array (dims,) + dims * (pix,)
        pixel indices to map each vector entry to
        for each of `dims` dimensions
    fill_value : float
        default value for pixels without corresponding
        `subset_idxs` entries

    Returns
    -------
    image : array of shape ``shape``
    """
    xp = core.get_array_module(image_vec)
    matrix = image_vec[:, core.newaxis]
    cube = wrap_matrix(matrix, (1,) + shape, subset_idxs, fill_value=fill_value)
    return cube[0]


def quick_derotate(cube, angles):
    """Rotate each plane of `cube` by the corresponding entry
    in `angles`, interpreted as deg E of N when N +Y and E +X
    (CCW when 0, 0 at lower left)

    Parameters
    ----------
    cube : array (planes, xpix, ypix)
    angles : array (planes,)

    Returns
    -------
    outimg : array (xpix, ypix)
    """
    outimg = np.zeros(cube.shape[1:])
    for i in range(cube.shape[0]):
        image = cube[i]
        if core.get_array_module(image) is da:
            image = image.compute()
        # n.b. skimage rotates CW by default, so we negate
        outimg += skimage.transform.rotate(image, -angles[i])

    return outimg


def mask_arc(
    center: Tuple[float, float],
    data_shape: Tuple[int, int],
    from_radius: float,
    to_radius: float,
    from_radians: float,
    to_radians: float,
    overall_rotation_radians: float = 0,
) -> np.ndarray:
    """Mask an arc beginning ``from_radius`` pixels from ``center``
    and going out to ``to_radius`` pixels, beginning at ``from_radians``
    from the +X direction (CCW when 0,0 at lower left) and going
    to ``to_radians``. For cases where it's easier to adjust the overall
    rotation than the bounds, ``overall_rotation_radians`` can be set to
    offset the ``from_radians`` and ``to_radians`` values

    Parameters
    ----------
    center : tuple[float, float]
        x, y pixel coordinates of the center of the grid
    data_shape : tuple[int, int]
        height, width shape (Python / NumPy order)
    from_radius : float
        pixel distance from center where mask `True` region
        should start
    to_radius : float
        pixel distance from center where mask `True` region
        should end
    from_radians : float
        angle in radians from +X where mask `True` region
        should start
    to_radians : float
        angle in radians from +X where mask `True` region
        should end
    overall_rotation_radians : float (default: 0)
        amount to rotate coordinate grid from +X
    """
    rho, phi = polar_coords(center, data_shape)
    phi = (phi + overall_rotation_radians) % (2 * np.pi)
    mask = (from_radius <= rho) & (rho <= to_radius)
    from_radians %= 2 * np.pi
    to_radians %= 2 * np.pi
    if from_radians != to_radians:
        mask &= (from_radians <= phi) & (phi <= to_radians)
    return mask


def cartesian_coords(
    center: Tuple[float, float], data_shape: Tuple[int, int]
) -> np.ndarray:
    """center in x,y order; data_shape in (h, w); returns coord arrays xx, yy of data_shape

    Matrix layout ((0,0) at upper left) for (2, 2) matrix::

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
    pixels. The coordinates of the centers in the new system are then::

            ---------------+X---------------->

            |  +--------------+ +--------------+
            |  | (-0.5, -0.5) | |  (0.5, -0.5) |
           +Y  +--------------+ +--------------+
            |  +--------------+ +--------------+
            |  | (-0.5, 0.5)  | |  (0.5, 0.5)  |
            |  +--------------+ +--------------+
            V

    Which means coordinate arrays of::

        xx = [[-0.5, 0.5], [-0.5, 0.5]]
        yy = [[-0.5, -0.5], [0.5, 0.5]]
    """
    yy, xx = np.indices(data_shape, dtype=float)
    center_x, center_y = center
    yy -= center_y
    xx -= center_x
    return np.stack([xx, yy])


def polar_coords(
    center: Tuple[float, float], data_shape: Tuple[int, int]
) -> np.ndarray:
    """center in x,y order; data_shape in (h, w); returns coord arrays rho, phi of data_shape"""
    xx, yy = cartesian_coords(center, data_shape)
    rho = np.sqrt(yy ** 2 + xx ** 2)
    phi = np.arctan2(yy, xx)
    return np.stack([rho, phi])


def max_radius(center: Tuple[float, float], data_shape: Tuple[int, int]) -> float:
    """Given an (x, y) center location and a data shape of
    (height, width) return the largest circle radius from that center
    that is completely within the data bounds"""
    if center[0] > (data_shape[1] - 1) or center[1] > (data_shape[0] - 1):
        raise ValueError("Coordinates for center are outside data_shape")
    bottom_left = np.sqrt(center[0] ** 2 + center[1] ** 2)
    data_height, data_width = data_shape
    top_right = np.sqrt((data_height - center[0]) ** 2 + (data_width - center[1]) ** 2)
    return min(bottom_left, top_right)


def ft_shift2(
    image: np.ndarray,
    dx: float,
    dy: float,
    flux_tol: Union[None, float] = 1e-15,
    output_shape=None,
):
    """
    Fast Fourier subpixel shifting

    Parameters
    ----------
    dx : float
        Translation in +X direction (i.e. a feature at (x, y) moves to (x + dx, y))
    dy : float
        Translation in +Y direction (i.e. a feature at (x, y) moves to (x, y + dy))
    flux_tol : float
        Fractional flux change permissible
        ``(sum(output) - sum(image)) / sum(image) < flux_tol``
        (default: 1e-15)
    output_shape : tuple
        shape of output array (default: same as input)
    """
    if output_shape is None:
        output_shape = image.shape
    xfreqs = np.fft.fftfreq(output_shape[1])
    yfreqs = np.fft.fftfreq(output_shape[0])
    xform = np.fft.fft2(image, s=output_shape)
    if output_shape is not None:
        # compute center-to-center displacement such that
        # supplying dx == dy == 0.0 will be a no-op (aside
        # from changing shape)
        orig_ctr_x, orig_ctr_y = (image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2
        new_ctr_x, new_ctr_y = (output_shape[1] - 1) / 2, (output_shape[0] - 1) / 2
        base_dx, base_dy = new_ctr_x - orig_ctr_x, new_ctr_y - orig_ctr_y
    else:
        base_dx = base_dy = 0
    modified_xform = xform * np.exp(
        2j
        * np.pi
        * (
            (-(dx + base_dx) * xfreqs)[np.newaxis, :]
            + (-(dy + base_dy) * yfreqs)[:, np.newaxis]
        )
    )
    new_image = np.fft.ifft2(modified_xform).real
    frac_diff_flux = (np.sum(image) - np.sum(new_image)) / np.sum(image)
    if flux_tol is not None and frac_diff_flux > flux_tol:
        raise RuntimeError(
            f"Flux conservation violated by {frac_diff_flux} fractional difference (more than {flux_tol})"
        )
    return new_image


def shift2(image, dx, dy, how="bicubic", output_shape=None):
    """Wraps scikit-image for 2D shifts with bicubic or bilinear interpolation

    Direction convention: feature at (0, 0) moves to (dx, dy)
    """
    if output_shape is not None:
        # compute center-to-center displacement such that
        # supplying dx == dy == 0.0 will be a no-op (aside
        # from changing shape)
        orig_ctr_x, orig_ctr_y = (image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2
        new_ctr_x, new_ctr_y = (output_shape[1] - 1) / 2, (output_shape[0] - 1) / 2
        base_dx, base_dy = new_ctr_x - orig_ctr_x, new_ctr_y - orig_ctr_y
    else:
        base_dx = base_dy = 0
    order = {"bicubic": 3, "bilinear": 1}
    tform = skimage.transform.AffineTransform(
        translation=(-(dx + base_dx), -(dy + base_dy))
    )
    output = skimage.transform.warp(
        image, tform, order=order[how], output_shape=output_shape
    )
    return output


def mask_box(shape, center, size, rotation=0):
    try:
        width, height = size
    except TypeError:
        width = height = size
    y, x = np.indices(shape)
    center_x, center_y = center
    if rotation != 0:
        r = np.hypot(x - center_x, y - center_y)
        phi = np.arctan2(y - center_y, x - center_x)
        y = r * np.sin(phi + np.deg2rad(rotation)) + center_y
        x = r * np.cos(phi + np.deg2rad(rotation)) + center_x
    return (
        (y >= center_y - height / 2)
        & (y <= center_y + height / 2)
        & (x >= center_x - width / 2)
        & (x <= center_x + width / 2)
    )


def f_test(npix):
    """Create a square npix x npix array of zeros and draw a capital F
    that is upright and facing right when plotted with (0,0) at lower left
    as regions of ones"""
    f_test = np.zeros((npix, npix))
    mid = npix // 2
    stem = (
        slice(mid // 8, npix - mid // 8),
        slice((mid - mid // 4) - mid // 8, (mid - mid // 4) + mid // 8),
    )
    f_test[stem] = 1
    bottom = (
        slice(mid - mid // 8, mid + mid // 8),
        slice((mid - mid // 4) - mid // 8, (mid - mid // 4) + 2 * mid // 3),
    )
    f_test[bottom] = 1
    top = (
        slice(npix - mid // 8 - mid // 4, npix - mid // 8),
        slice((mid - mid // 4) - mid // 8, (mid - mid // 4) + mid),
    )
    f_test[top] = 1
    return f_test


def combine_paired_cubes(cube_1, cube_2, mask_1, mask_2, fill_value=np.nan):
    log.debug(
        f"combine_paired_cubes({cube_1=}, {cube_2=}, {mask_1=}, {mask_2=}, {fill_value=})"
    )
    xp = core.get_array_module(cube_1)
    if cube_1.shape != cube_2.shape:
        raise ValueError("cube_1 and cube_2 must be the same shape")
    if mask_1.shape != cube_1.shape[1:] or mask_1.shape != mask_2.shape:
        raise ValueError(
            "mask_1 and mask_2 must be the same shape as the last dimensions of cube_1"
        )
    if xp is da:
        output = da.blockwise(
            combine_paired_cubes,
            "ijk",
            cube_1,
            "ijk",
            cube_2,
            "ijk",
            mask_1,
            "jk",
            mask_2,
            "jk",
            fill_value=fill_value,
            dtype=cube_1.dtype,
        )
    else:
        output = fill_value * xp.ones_like(cube_1)
        output[:, mask_1] = cube_1[:, mask_1]
        output[:, mask_2] = cube_2[:, mask_2]
    log.debug(f"{output}")
    return output


def derotate_cube(cube, derotation_angles):
    xp = core.get_array_module(cube)
    if cube.shape[0] != derotation_angles.shape[0]:
        raise ValueError("Number of cube planes and derotation angles must match")
    if xp is da:
        return da.blockwise(derotate_cube, "ijk", cube, "ijk", derotation_angles, "i")
    output = np.zeros_like(cube)
    for idx in range(cube.shape[0]):
        # n.b. skimage rotates CW by default, so we negate
        output[idx] = skimage.transform.rotate(cube[idx], -derotation_angles[idx])
    return output


@dataclass
class BBox:
    origin: tuple[int, int]
    extent: tuple[int, int]

    @classmethod
    def from_center(cls, center_coords, extent):
        cy, cx = center_coords
        return cls(origin=(cy - extent[0] / 2, cx - extent[1] / 2), extent=extent)

    @property
    def center(self):
        oy, ox = self.origin
        dy, dx = self.extent
        return oy + (dy - 1) / 2, ox + (dx - 1) / 2

    @center.setter
    def center(self, value):
        cy, cx = self.center
        cy_prime, cx_prime = value
        dy, dx = cy_prime - cy, cx_prime - cx
        oy, ox = self.origin
        self.origin = oy + dy, ox + dx

    @property
    def slices(self):
        oy, ox = self.origin
        dy, dx = self.extent
        start_y, end_y = oy, oy + dy
        if int(start_y) != start_y or int(end_y) != end_y:
            warnings.warn(
                f"Coercing {start_y=} to {int(start_y)}, {end_y=} to {int(end_y)}"
            )
        start_x, end_x = ox, ox + dx
        if int(start_x) != start_x or int(end_x) != end_x:
            warnings.warn(
                f"Coercing {start_x=} to {int(start_x)}, {end_x=} to {int(end_x)}"
            )
        return slice(start_y, end_y), slice(start_x, end_x)


distributed.protocol.register_generic(BBox)


@dataclass
class CutoutTemplateSpec(BBox):
    template: np.ndarray
    name: str


distributed.protocol.register_generic(CutoutTemplateSpec)


def gauss2d(shape, center, sigma):
    """Evaluate Gaussian distribution in 2D on an array of shape
    `shape`, centered at `(center_y, center_x)`, and with a sigma
    of `(sigma_y, sigma_x)`
    """
    mu_y, mu_x = center
    sigma_y, sigma_x = sigma
    yy, xx = np.indices(shape, dtype=float)

    return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-(
            (xx - mu_x) ** 2 / (2 * sigma_x ** 2) +
            (yy - mu_y) ** 2 / (2 * sigma_y ** 2)
    ))

def pad_to_match(arr_a : np.ndarray, arr_b : np.ndarray):

    a_height, a_width = arr_a.shape
    b_height, b_width = arr_b.shape
    if a_height > b_height:
        arr_b = np.pad(arr_b, [(0, a_height - b_height), (0, 0)])
    elif b_height > a_height:
        arr_a = np.pad(arr_a, [(0, b_height - a_height), (0, 0)])

    if a_width > b_width:
        arr_b = np.pad(arr_b, [(0, 0), (0, a_width - b_width)])
    elif b_width > a_width:
        arr_a = np.pad(arr_a, [(0, 0), (0, b_width - a_width)])
    assert arr_a.shape == arr_b.shape
    return arr_a, arr_b


def aligned_cutout(sci_arr: np.ndarray, spec: CutoutTemplateSpec, upsample_factor: int = 100):
    # cut out bbox
    rough_cutout = sci_arr[spec.slices]
    yy, xx = np.indices(rough_cutout.shape)
    interp_cutout = regrid_image(
        rough_cutout,
        x_prime=xx,
        y_prime=yy,
        method='cubic'
    )
    template = spec.template
    # pad to match shapes
    rough_cutout, template = pad_to_match(rough_cutout, template)
    # xcorr
    shifts, error, phasediff = skimage.registration.phase_cross_correlation(
        reference_image=template,
        moving_image=interp_cutout,
        upsample_factor=upsample_factor,
    )
    # add offset + xcorr displacement
    yy, xx = np.indices(template.shape, dtype=float)
    yy -= spec.origin[0]
    yy -= shifts[0]
    xx -= spec.origin[1]
    xx -= shifts[1]
    # regrid
    out_image = regrid_image(sci_arr, xx, yy, method='cubic')
    return out_image

from scipy import interpolate
def make_grid(shape,
              rotation, rotation_x_center, rotation_y_center,
              scale_x, scale_y, scale_x_center, scale_y_center,
              x_shift, y_shift):
    '''
    Given the dimensions of a 2D image, compute the pixel center coordinates
    for a rotated/scaled/shifted grid.

    1. Rotate about (rotation_x_center, rotation_y_center)
    2. Scale about (scale_x_center, scale_y_center)
    3. Shift by (x_shift, y_shift)

    Returns
    -------

    xx, yy : 2D arrays
        x and y coordinates for pixel centers
        of the shifted grid
    '''
    yy, xx = np.indices(shape)
    if rotation != 0:
        r = np.hypot(xx - rotation_x_center, yy - rotation_y_center)
        phi = np.arctan2(yy - rotation_y_center, xx - rotation_x_center)
        yy_rotated = r * np.sin(phi + rotation) + rotation_y_center
        xx_rotated = r * np.cos(phi + rotation) + rotation_x_center
    else:
        yy_rotated, xx_rotated = yy, xx
    if scale_y != 1:
        yy_scaled = (yy_rotated - scale_y_center) / scale_y + scale_y_center
    else:
        yy_scaled = yy_rotated
    if scale_x != 1:
        xx_scaled = (xx_rotated - scale_x_center) / scale_x + scale_x_center
    else:
        xx_scaled = xx_rotated
    if y_shift != 0:
        yy_shifted = yy_scaled + y_shift
    else:
        yy_shifted = yy_scaled
    if x_shift != 0:
        xx_shifted = xx_scaled + x_shift
    else:
        xx_shifted = xx_scaled
    return xx_shifted, yy_shifted

def regrid_image(image, x_prime, y_prime, method='cubic', mask=None, fill_value=0.0):
    '''Given a 2D image and correspondingly shaped mask,
    as well as 2D arrays of transformed X and Y coordinates,
    interpolate a transformed image.

    Parameters
    ----------
    image
        2D array holding an image
    x_prime
        transformed X coordinates in the same shape as image.shape
    y_prime
        tranformed Y coordinates
    method : optional, default 'cubic'
        interpolation method passed to `scipy.interpolate.griddata`
    mask
        boolean array of pixels to keep
        ('and'-ed with the set of finite/non-NaN pixels)
    fill_value
        value for points outside the convex hull of True mask entries
    '''
    if mask is not None:
        mask = mask.copy()
        mask &= np.isfinite(image)
    else:
        mask = np.isfinite(image)
    yy, xx = np.indices(image.shape)
    xx_sub = xx[mask]
    yy_sub = yy[mask]
    zz = image[mask]
    new_image = interpolate.griddata(
        np.stack((xx_sub.flat, yy_sub.flat), axis=-1),
        zz.flatten(),
        (x_prime, y_prime),
        fill_value=fill_value,
        method=method).reshape(x_prime.shape)
    return new_image

def encircled_energy_and_profile(
    data,
    center,
    arcsec_per_px=None,
    normalize=None,
    display=False,
    ee_ax=None,
    profile_ax=None,
    label=None,
):
    """Compute encircled energies and profiles

    Returns
    -------
    ee_rho_steps
    encircled_energy_at_rho
    profile_bin_centers_rho
    profile_value_at_rho
    """
    rho, phi = polar_coords(center, data.shape)
    max_radius_px = int(max_radius(center, data.shape))

    ee_rho_steps = []
    profile_bin_centers_rho = []

    encircled_energy_at_rho = []
    profile_value_at_rho = []

    for n in np.arange(1, max_radius_px):
        interior_mask = rho < n - 1
        exterior_mask = rho < n
        ring_mask = exterior_mask & ~interior_mask

        # EE
        ee_npix = np.count_nonzero(exterior_mask)
        if ee_npix > 0:
            ee = np.nansum(data[exterior_mask])
            encircled_energy_at_rho.append(ee)
            ee_rho_steps.append(n * arcsec_per_px if arcsec_per_px is not None else n)

        # profile
        profile_npix = np.count_nonzero(ring_mask)
        if profile_npix > 0:
            profile_bin_centers_rho.append(
                (n - 0.5) * arcsec_per_px if arcsec_per_px is not None else n - 0.5
            )
            profile_value = np.nansum(data[ring_mask]) / profile_npix
            profile_value /= profile_npix
            profile_value_at_rho.append(profile_value)

    (
        ee_rho_steps,
        encircled_energy_at_rho,
        profile_bin_centers_rho,
        profile_value_at_rho,
    ) = (
        np.asarray(ee_rho_steps),
        np.asarray(encircled_energy_at_rho),
        np.asarray(profile_bin_centers_rho),
        np.asarray(profile_value_at_rho),
    )
    if normalize is not None:
        if normalize is True:
            ee_normalize_at_mask = profile_normalize_at_mask = None
        else:
            ee_normalize_at_mask = ee_rho_steps < normalize
            profile_normalize_at_mask = profile_bin_centers_rho < normalize
        encircled_energy_at_rho /= np.nanmax(
            encircled_energy_at_rho[ee_normalize_at_mask]
        )
        profile_value_at_rho /= np.nanmax(
            profile_value_at_rho[profile_normalize_at_mask]
        )

    if display:
        import matplotlib.pyplot as plt

        if ee_ax is None or profile_ax is None:
            _, (ee_ax, profile_ax) = plt.subplots(figsize=(8, 6), nrows=2)
        xlabel = r"$\rho$ [arcsec]" if arcsec_per_px is not None else r"$\rho$ [pixel]"
        ee_ax.set_xlabel(xlabel)
        ee_ax.set_ylabel("Encircled Energy")
        ee_ax.plot(ee_rho_steps, encircled_energy_at_rho, label=label)
        ee_ax.axvline(0, ls=":")
        profile_ax.set_xlabel(xlabel)
        profile_ax.set_ylabel("Radial Profile")
        profile_ax.set_yscale("log")
        profile_ax.plot(profile_bin_centers_rho, profile_value_at_rho, label=label)
        profile_ax.axvline(0, ls=":")
        plt.tight_layout()
    return (
        ee_rho_steps,
        encircled_energy_at_rho,
        profile_bin_centers_rho,
        profile_value_at_rho,
    )
