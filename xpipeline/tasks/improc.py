import warnings
from typing import Tuple, Union, List, Optional
import math
from functools import partial
import numpy as np
import logging
from numba import njit, jit
import numba
from numpy.core.numeric import count_nonzero
from scipy import interpolate
try:
    from pyfftw.interfaces import numpy_fft as fft
except ImportError:
    fft = np.fft

from scipy.ndimage import binary_dilation
import skimage.transform
import skimage.registration
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel
from dataclasses import dataclass
import distributed.protocol
from .. import core, constants

log = logging.getLogger(__name__)


def gaussian_smooth(data, kernel_stddev_px):
    return convolve_fft(data, Gaussian2DKernel(kernel_stddev_px), boundary="wrap")


def arr_center(arr_or_shape_or_extent):
    """Center coordinates in (y, x) order for a 2D image (or shape) using the
    convention that indices are the coordinates of the centers
    of pixels, which run from (idx - 0.5) to (idx + 0.5)
    """
    if isinstance(arr_or_shape_or_extent, PixelExtent):
        return (arr_or_shape_or_extent.height - 1) / 2, (arr_or_shape_or_extent.width - 1) / 2
    shape = getattr(arr_or_shape_or_extent, "shape", arr_or_shape_or_extent)
    if len(shape) != 2:
        raise ValueError("Only do this on 2D images")
    return (shape[0] - 1) / 2, (shape[1] - 1) / 2

def center_point(arr_or_shape_or_extent):
    ycoord, xcoord = arr_center(arr_or_shape_or_extent)
    return Point(y=ycoord, x=xcoord)

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

def _ensure_mask_bool(mask):
    if mask.dtype == bool:
        return mask
    return mask == 1

def unwrap_cube(cube, good_pix_mask):
    """Unwrap a shape (planes, ypix, xpix) `cube` and transpose into
    a (pix, planes) matrix, where `pix` is the number of *True*
    entries in a (ypix, xpix) `good_pix_mask` (i.e. False entries are omitted)

    Parameters
    ----------
    cube : array (planes, ypix, xpix)
        Data cube
    good_pix_mask : array (ypix, xpix)
        Indices from a single slice to include in `matrix`, matching
        last (N-1) dimensions of ndcube

    Returns
    -------
    matrix : array (pix, planes)
        Vectorized images, one per column
    """
    if len(good_pix_mask.shape) != 2 or len(cube.shape) != 3:
        raise ValueError(f"Tried to unwrap {cube.shape=} with {good_pix_mask.shape=}")
    good_pix_mask = _ensure_mask_bool(good_pix_mask)
    image_vecs = cube[:, good_pix_mask].T
    return image_vecs


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
    good_pix_mask = _ensure_mask_bool(good_pix_mask)
    cube = unwrap_cube(image[xp.newaxis, :, :], good_pix_mask)
    return cube[:, 0]


def wrap_matrix(matrix, good_pix_mask, fill_value=np.nan):
    """Wrap a (N, pix) matrix into a shape `shape`
    data cube using the indexes from `subset_idxs`

    Parameters
    ----------
    matrix : array (N, pix)
    good_pix_mask : array (pix2, pix2)
        Pixels to fill from columns of `matrix`
    fill_value : float
        default value for pixels not specified in the mask

    Returns
    -------
    cube : array of shape ``(N, pix2, pix2)``
    """
    xp = core.get_array_module(matrix)
    n_obs = matrix.shape[1]
    good_pix_mask = _ensure_mask_bool(good_pix_mask)
    cube = fill_value * xp.ones((n_obs,) + good_pix_mask.shape)
    cube[:, good_pix_mask] = matrix.T
    return cube


def wrap_vector(image_vec, good_pix_mask, fill_value=np.nan):
    """Wrap a (pix,) vector into a shape `shape` image using the
    indexes from `subset_idxs`

    Parameters
    ----------
    image_vec : array (N, pix)
    good_pix_mask : array (pix2, pix2)
        Pixels to fill from columns of `matrix`
    fill_value : float
        default value for pixels not specified in the mask

    Returns
    -------
    image : array of shape ``shape``
    """
    xp = core.get_array_module(image_vec)
    good_pix_mask = _ensure_mask_bool(good_pix_mask)
    matrix = image_vec[:, xp.newaxis]
    cube = wrap_matrix(matrix, good_pix_mask, fill_value=fill_value)
    return cube[0]



def mask_arc(
    center: Tuple[float, float],
    data_shape: Tuple[int, int],
    from_radius: float = 0,
    to_radius: float = None,
    from_radians: float = 0,
    to_radians: float = 0,
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
        y, x pixel coordinates of the center of the grid
    data_shape : tuple[int, int]
        height, width shape (Python / NumPy order)
    from_radius : float
        pixel distance from center where mask `True` region
        should start (default: 0)
    to_radius : float or None
        pixel distance from center where mask `True` region
        should end (default: None, goes to edges of array)
    from_radians : float
        angle in radians from +X where mask `True` region
        should start (default: 0)
    to_radians : float
        angle in radians from +X where mask `True` region
        should end (default: 0)
    overall_rotation_radians : float (default: 0)
        amount to rotate coordinate grid from +X
    """
    rho, phi = polar_coords(center, data_shape)
    phi = (phi + overall_rotation_radians) % (2 * np.pi)
    mask = (from_radius <= rho)
    if to_radius is not None:
        mask &= (rho <= to_radius)
    from_radians %= 2 * np.pi
    to_radians %= 2 * np.pi
    if from_radians != to_radians:
        mask &= (from_radians <= phi) & (phi <= to_radians)
    return mask

def cartesian_coords(
    center: Tuple[float, float], data_shape: Tuple[int, int]
) -> np.ndarray:
    """center in y,x order; data_shape in (h, w); returns coord arrays yy, xx of data_shape

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
    center_y, center_x = center
    yy -= center_y
    xx -= center_x
    return np.stack([yy, xx])


def polar_coords(
    center: Tuple[float, float], data_shape: Tuple[int, int]
) -> np.ndarray:
    """center in y,x order; data_shape in (h, w); returns coord arrays rho, phi of data_shape"""
    yy, xx = cartesian_coords(center, data_shape)
    rho = np.sqrt(yy ** 2 + xx ** 2)
    phi = np.arctan2(yy, xx)
    return np.stack([rho, phi])

def pa_coords(
    center: Tuple[float, float], data_shape: Tuple[int, int]
) -> np.ndarray:
    """center in y,x order; data_shape in (h, w); returns coord arrays rho, pa of data_shape
    where pa == 0 is aligned with the +Y direction, increasing CCW when 0,0 at lower left"""
    rho, phi = polar_coords(center, data_shape)
    return rho, np.rad2deg((phi + 3 * np.pi / 2) % (2 * np.pi))

def downsampled_grid_r_pa(mask, downsample):
    '''Given a boolean mask and downsampling factor,
    return the r_px, pa_deg, x, y arrays with coordinates corresponding
    to every `downsample`th pixel, filtered to pixels where
    `mask` is True
    '''
    yc, xc = arr_center(mask)
    rho, pa_deg = pa_coords((yc, xc), mask.shape)
    rho[~mask] = np.nan
    pa_deg[~mask] = np.nan
    yy, xx = np.indices(mask.shape)
    yy_downsamp = yy[::downsample, ::downsample]
    xx_downsamp = xx[::downsample, ::downsample]
    rho_downsamp, pa_deg_downsamp = rho[yy_downsamp, xx_downsamp], pa_deg[yy_downsamp, xx_downsamp]
    notnan = ~np.isnan(rho_downsamp)
    return rho_downsamp[notnan], pa_deg_downsamp[notnan], xx_downsamp[notnan], yy_downsamp[notnan]

def max_radius(center: Tuple[float, float], data_shape: Tuple[int, int]) -> float:
    """Given an (x, y) center location and a data shape of
    (height, width) return the largest circle radius from that center
    that is completely within the data bounds"""
    if center[0] > (data_shape[1] - 1) or center[1] > (data_shape[0] - 1):
        raise ValueError("Coordinates for center are outside data_shape")
    data_height, data_width = data_shape
    dx, dy = center
    odx, ody = data_width - dx, data_height - dy
    return min(dx, dy, odx, ody)

def min_radius(center: Tuple[float, float], data_shape: Tuple[int, int], data: np.ndarray) -> float:
    max_radius_int = max_radius(center, data_shape)
    rho, _ = polar_coords(center, data_shape)
    for i in range(max_radius_int):
        mask = i - 0.5 < rho < i + 0.5
        if np.all(np.isfinite(data[mask])):
            return i
    raise ValueError("At no radius is there a 1 pixel ring of finite values")

def ft_shift2(image: np.ndarray, dy: float, dx: float, flux_tol: Union[None, float] = 1e-15, output_shape=None):
    """
    Fast Fourier subpixel shifting

    Parameters
    ----------
    dy : float
        Translation in +Y direction (i.e. a feature at (x, y) moves to (x, y + dy))
    dx : float
        Translation in +X direction (i.e. a feature at (x, y) moves to (x + dx, y))
    flux_tol : float
        Fractional flux change permissible
        ``(sum(output) - sum(image)) / sum(image) < flux_tol``
        (default: 1e-15)
    output_shape : tuple
        shape of output array (default: same as input)
    """
    if output_shape is None:
        output_shape = image.shape
    xfreqs = fft.fftfreq(output_shape[1])
    yfreqs = fft.fftfreq(output_shape[0])
    xform = fft.fft2(image, s=output_shape)
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
    new_image = fft.ifft2(modified_xform).real
    frac_diff_flux = (np.sum(image) - np.sum(new_image)) / np.sum(image)
    if flux_tol is not None and frac_diff_flux > flux_tol:
        raise RuntimeError(
            f"Flux conservation violated by {frac_diff_flux} fractional difference (more than {flux_tol})"
        )
    return new_image


def mask_box(center: tuple[float, float], shape: tuple[int,int], size: tuple[float,float], rotation_deg:float=0):
    '''
    Parameters
    ----------
    center : tuple
        Center coordinates in y, x order
    shape : tuple
        Shape in y, x order
    size : tuple, float
        Size of box in height, width order or single value for
        width = height = size
    rotation_deg : float

    '''
    try:
        height, width = size
    except TypeError:
        width = height = size
    y, x = np.indices(shape)
    center_y, center_x = center
    rotation = np.deg2rad(rotation_deg)
    if rotation != 0:
        r = np.hypot(x - center_x, y - center_y)
        phi = np.arctan2(y - center_y, x - center_x)
        y = r * np.sin(phi + rotation) + center_y
        x = r * np.cos(phi + rotation) + center_x
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
        f"combine_paired_cubes({cube_1.shape=}, {cube_2.shape=}, {mask_1.shape=}, {mask_2.shape=}, {fill_value=})"
    )
    if cube_1.shape != cube_2.shape:
        raise ValueError("cube_1 and cube_2 must be the same shape")
    if mask_1.shape != cube_1.shape[1:] or mask_1.shape != mask_2.shape:
        raise ValueError(
            "mask_1 and mask_2 must be the same shape as the last dimensions of cube_1"
        )

    output = fill_value * np.ones_like(cube_1)
    output[:, mask_1] = cube_1[:, mask_1]
    output[:, mask_2] = cube_2[:, mask_2]
    log.debug(f"{output.shape=}")
    return output


@dataclass(kw_only=True)
class Point:
    y : Union[float, int]
    x : Union[float, int]

@dataclass(kw_only=True)
class Pixel(Point):
    y : int
    x : int
    def __post_init__(self):
        self.y = int(self.y)
        self.x = int(self.x)

@dataclass(kw_only=True)
class PixelExtent:
    height : int
    width : int

    def __post_init__(self):
        self.height = int(self.height)
        self.width = int(self.width)

distributed.protocol.register_generic(PixelExtent)

@dataclass
class BBox:
    origin : Pixel
    extent : PixelExtent

    @classmethod
    def from_center(cls, center : Point, extent : PixelExtent):
        cy, cx = center.y, center.x
        origin = Pixel(y=cy - (extent.height - 1) / 2, x=cx - (extent.width - 1)/ 2)
        return cls(origin=origin, extent=extent)

    @property
    def center(self):
        oy, ox = self.origin.y, self.origin.x
        dy, dx = self.extent.height, self.extent.width
        return Point(y=oy + (dy - 1) / 2, x=ox + (dx - 1) / 2)

    @center.setter
    def center(self, value):
        cy, cx = self.center.y, self.center.x
        cy_prime, cx_prime = value.y, value.x
        dy, dx = cy_prime - cy, cx_prime - cx
        oy, ox = self.origin
        if (oy + dy) - int(oy + dy) > 0.01 or (ox + dx) - int(ox + dx) > 0.01:
            warnings.warn(f"Loss of precision rounding origin to integer pixels")
        self.origin = Pixel(y=int(oy + dy), x=int(ox + dx))

    def _slices(self):
        oy, ox = self.origin.y, self.origin.x
        dy, dx = self.extent.height, self.extent.width
        start_y, end_y = oy, oy + dy
        start_x, end_x = ox, ox + dx
        start_y = max(start_y, 0)
        start_x = max(start_x, 0)
        return (start_y, end_y), (start_x, end_x)

    @property
    def min_x(self):
        return self._slices()[1][0]

    @property
    def max_x(self):
        return self._slices()[1][1]

    @property
    def min_y(self):
        return self._slices()[0][0]

    @property
    def max_y(self):
        return self._slices()[0][1]

    @property
    def width(self):
        return self.extent.width

    @property
    def height(self):
        return self.extent.height

    @property
    def shape(self):
        return self.extent.height, self.extent.width

    @property
    def slices(self):
        (start_y, end_y), (start_x, end_x) = self._slices()
        return slice(start_y, end_y), slice(start_x, end_x)

    def _coords_grid_xy(self, slice_y : slice, slice_x : slice):
        xx, yy = np.meshgrid(
            np.arange(slice_x.start, slice_x.stop), # + 0.5,
            np.arange(slice_y.start, slice_y.stop), # + 0.5,
        )
        return xx, yy

    @property
    def coords_grid_xy(self):
        slice_y, slice_x = self.slices
        return self._coords_grid_xy(slice_y, slice_x)

    def get_overlap_slices(self, other):
        (start_y, end_y), (start_x, end_x) = self._slices()
        end_y = min(end_y, other.shape[0])
        end_x = min(end_x, other.shape[1])
        return slice(start_y, end_y), slice(start_x, end_x)

    def get_overlap_grid_xy(self, other):
        slice_y, slice_x = self.get_overlap_slices(other)
        return self._coords_grid_xy(slice_y, slice_x)



distributed.protocol.register_generic(BBox)

@dataclass
class ImageFeatureSpec:
    search_box : BBox
    template: np.ndarray

distributed.protocol.register_generic(ImageFeatureSpec)

def gauss2d(shape, center, sigma):
    """Evaluate Gaussian distribution in 2D on an array of shape
    `shape`, centered at `(center_y, center_x)`, and with a sigma
    of `(sigma_y, sigma_x)`
    """
    mu_y, mu_x = center
    sigma_y, sigma_x = sigma
    yy, xx = np.indices(shape, dtype=float)

    return (
        1
        / (2 * np.pi * sigma_x * sigma_y)
        * np.exp(
            -(
                (xx - mu_x) ** 2 / (2 * sigma_x ** 2)
                + (yy - mu_y) ** 2 / (2 * sigma_y ** 2)
            )
        )
    )


def pad_to_match(arr_a: np.ndarray, arr_b: np.ndarray):
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

def shifts_from_cutout(sci_arr, spec, upsample_factor=100):
    # cut out bbox
    rough_cutout = sci_arr[spec.search_box.slices]
    interp_cutout = interpolate_nonfinite(rough_cutout)
    template = spec.template
    # pad to match shapes
    interp_cutout, template = pad_to_match(interp_cutout, template)
    # xcorr
    shifts, error, phasediff = skimage.registration.phase_cross_correlation(
        reference_image=template,
        moving_image=interp_cutout,
        upsample_factor=upsample_factor,
        normalization=None,
    )
    return shifts

def subpixel_location_from_cutout(sci_arr, spec: ImageFeatureSpec, upsample_factor=100, prefilter_sigma_px: float = 0.0):
    '''Compute the location of the feature given by `spec` to sub-pixel precision
    assuming the spec template is centered at the array center (npix-1)/2

    Parameters
    ----------
    sci_arr : 2D array
    spec : ImageFeatureSpec
    upsample_factor : int = 100
        Supersampling factor for the FFT in the region of the
        correlation peak to find a sub-pixel position
    prefilter_sigma_px : float = 0.0
        For values >0.0, prefilter sci_arr with a Gaussian smoothing
        before applying `shifts_from_cutout` (cross-correlation registration)
    '''
    subframe = sci_arr[spec.search_box.slices]
    subframe = gaussian_smooth(subframe, prefilter_sigma_px)
    subframe, template = pad_to_match(interpolate_nonfinite(subframe), spec.template)
    # xcorr
    shifts, error, phasediff = skimage.registration.phase_cross_correlation(
        reference_image=subframe,
        moving_image=template,
        upsample_factor=upsample_factor,
        normalization=None,
    )
    yc, xc = arr_center(spec.template)
    return Point(y=spec.search_box.origin.y + yc + shifts[0], x=spec.search_box.origin.x + xc + shifts[1])


def aligned_cutout(
    sci_arr: np.ndarray, spec: ImageFeatureSpec, upsample_factor: int = 100,
    prefilter_sigma_px: float = 0.0,
):
    '''Produce an interpolated subarray from sci_arr such that the 
    image feature is aligned with the template in `spec`

    Parameters
    ----------
    sci_arr : np.ndarray
    spec : ImageFeatureSpec
    upsample_factor : int = 100
        Supersampling factor for the FFT in the region of the
        correlation peak to find a sub-pixel position
    prefilter_sigma_px : float = 0.0
        For values >0.0, prefilter sci_arr with a Gaussian smoothing
        before applying `shifts_from_cutout` (cross-correlation registration)
    '''
    sci_arr = interpolate_nonfinite(sci_arr)
    if prefilter_sigma_px > 0:
        correlate_on = gaussian_smooth(sci_arr, prefilter_sigma_px)
    else:
        correlate_on = sci_arr
    shifts = shifts_from_cutout(correlate_on, spec, upsample_factor=upsample_factor)
    subpix_sci_arr = shift2(
        sci_arr,
        shifts[1] - spec.search_box.origin.x,
        shifts[0] - spec.search_box.origin.y,
        output_shape=spec.template.shape,
        anchor_to_center=False  # we're interpolating and cropping at the same time
    )
    assert subpix_sci_arr.shape == spec.template.shape
    return subpix_sci_arr

def histogram_std(hist, bin_centers=None):
    '''Given a histogram of values and optionally coordinates
    for bin centers (if not, assume pixel centers centered at zero)
    approximate the standard deviation'''
    if bin_centers is None:
        bin_centers = pixel_centers(len(hist))
    mean = np.average(bin_centers, weights=hist)
    var = np.average((bin_centers - mean)**2, weights=hist)
    return np.sqrt(var)

def pixel_centers(length):
    '''Returns sequence of length `length` containing pixel
    center indices centered at 0'''
    return np.arange(length) - (length - 1)/ 2

def radial_stds_cube(cube):
    '''Given a data cube, compute the standard deviation
    in X and Y for each frame and combine to get a single
    radial sigma value per frame
    '''
    y_bin_centers = pixel_centers(cube.shape[1])
    x_bin_centers = pixel_centers(cube.shape[2])
    sum_along_ys = cube.sum(axis=1)
    sum_along_xs = cube.sum(axis=2)
    x_stds = np.apply_along_axis(partial(histogram_std, bin_centers=x_bin_centers), 1, sum_along_ys)
    y_stds = np.apply_along_axis(partial(histogram_std, bin_centers=y_bin_centers), 1, sum_along_xs)
    r_stds = np.sqrt(x_stds**2 + y_stds**2)
    return r_stds



def make_grid(
    shape,
    rotation,
    rotation_x_center,
    rotation_y_center,
    scale_x,
    scale_y,
    scale_x_center,
    scale_y_center,
    x_shift,
    y_shift,
):
    """
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
    """
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


def regrid_image(image, x_prime, y_prime, method="cubic", mask=None, fill_value=0.0):
    """Given a 2D image and correspondingly shaped mask,
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
    """
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
        method=method,
    ).reshape(x_prime.shape)
    return new_image


def downsample_first_axis(data, chunk_size, operation: constants.CombineOperation):
    '''Downsample chunks of `chunk_size` along axis 0 with median combination'''
    if operation is constants.CombineOperation.MEAN:
        op_ufunc = np.nanmean
    elif operation is constants.CombineOperation.SUM:
        op_ufunc = np.nansum
    elif operation is constants.CombineOperation.MEDIAN:
        op_ufunc = np.nanmedian
    else:
        raise ValueError("Supported operations: sum, mean, median")
    ndata = data.shape[0]
    nchunks = ndata // chunk_size
    if ndata % chunk_size != 0:
        nchunks += 1
    output = np.zeros((nchunks,) + data.shape[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for chunk_idx in range(nchunks):
            if (chunk_idx + 1) * chunk_size > ndata:
                chunk = data[chunk_idx * chunk_size:]
                output[chunk_idx] = op_ufunc(chunk, axis=0)
            else:
                chunk = data[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                output[chunk_idx] = op_ufunc(chunk, axis=0)
    return output

def encircled_energy_and_profile(
    data : np.ndarray,
    center : tuple[float, float],
    arcsec_per_px : float=None,
    normalize : float =None,
    display : bool=False,
    ee_ax=None,
    profile_ax=None,
    label : str=None,
    saturated_pixel_threshold : Optional[float]=None,
):
    """Compute encircled energies and profiles

    Parameters
    ----------
    data
        2D image data array
    center
        y,x center coordinates for r=0
    arcsec_per_px
        conversion factor, changes returned values from pixels to arcsec
    normalize
        radius in pixels (or arcsec) where EE is fixed to be 1.0
    display
        whether to generate a plot
    ee_ax
        axes on which to plot encircled energy
    profile_ax
        axes on which to plot radial profile
    label
        label to attach to curves in ee_ax and profile_ax
    saturated_pixel_threshold
        annuli containing pixels above this threshold will
        have a profile value of NaN

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
            if saturated_pixel_threshold is not None and np.any(data[ring_mask] >= saturated_pixel_threshold):
                profile_value_at_rho.append(np.nan)
            else:
                profile_value = np.nansum(data[ring_mask]) / profile_npix
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

@numba.njit(numba.float64[:,:](numba.float64, numba.float64), cache=True, inline='always')
def translation_matrix(dx, dy):
    """When multiplied on the right by a [x, y, 1] augmented vector, has the effect
    of translating the point to [x + dx, y + dy]"""
    xform = np.zeros((3, 3))
    xform[0,0] = 1
    xform[0,2] = dx
    xform[1,1] = 1
    xform[1,2] = dy
    xform[2,2] = 1
    # [
    #     [1, 0, dx],
    #     [0, 1, dy],
    #     [0, 0, 1]
    # ]
    return xform

@numba.njit(numba.float64[:,:](numba.float64), cache=True, inline='always')
def rotation_matrix(theta_rad):
    '''When multiplied on the right by a [x, y, 1] augmented vector, has the effect
    of rotating from +X towards +Y (CCW) by `theta_rad`'''
    xform = np.zeros((3, 3))
    xform[0,0] = np.cos(theta_rad)
    xform[0,1] = -np.sin(theta_rad)
    xform[1,0] = np.sin(theta_rad)
    xform[1,1] = np.cos(theta_rad)
    xform[2,2] = 1
    return xform


@numba.njit(numba.float64[:,:](numba.float64, numba.float64), cache=True, inline='always')
def scaling_matrix(sx, sy):
    xform = np.zeros((3, 3))
    xform[0,0] = sx
    xform[1,1] = sy
    xform[2,2] = 1
    return xform


@numba.njit
def make_rotation_about_center(image_shape, rotation_deg):
    '''Construct transformation matrix that maps
    (u, v) final image coordinates to (x, y) source
    image coordinates

    Parameters
    ----------
    image_shape : tuple
        The array shape in NumPy order
    rotation_deg : float
        Rotation in degrees towards the +Y axis (CCW when origin is lower-left)
        as applied to final image (i.e. `transform_mtx` when applied to a unit
        vector in (u,v) space will map to (x, y) coordinates rotated by this
        amount)

    Returns
    -------
    transform_mtx : (3, 3) array
        The augmented matrix expressing the affine transform
    '''
    npix_y, npix_x = image_shape
    ctr_x, ctr_y = (npix_x - 1) / 2, (npix_y - 1) / 2
    theta = np.deg2rad(-rotation_deg)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    if rotation_deg != 0:
        xform = np.zeros((3, 3))
        xform[0,0] = cos_theta
        xform[0,1] = -sin_theta
        xform[0,2] = sin_theta * ctr_y - cos_theta * ctr_x + ctr_x
        xform[1,0] = sin_theta
        xform[1,1] = cos_theta
        xform[1,2] = -ctr_x * sin_theta + -ctr_y * cos_theta + ctr_y
        xform[2,2] = 1
        return xform
    else:
        return np.eye(3)

def rescale_about_center(image, center : Point, scale_factor, output_shape, interpolation_fill_value=np.nan, missing_fill_value=np.nan):
    trans = translation_matrix(center.x, center.y)
    new_center = center_point(output_shape)
    untrans = translation_matrix(-new_center.x, -new_center.y)
    transform_mtx = (
        trans
        @
        scaling_matrix(1/scale_factor, 1/scale_factor)
        @
        untrans
    )
    dest_image = np.zeros(output_shape, dtype=image.dtype)
    matrix_transform_image(image, transform_mtx, dest_image, interpolation_fill_value=interpolation_fill_value, missing_fill_value=missing_fill_value)
    return dest_image

@numba.jit(inline='always', nopython=True, cache=True)
def cpu_cubic1d(t, f_minus1, f_0, f_1, f_2):
    a = 2 * f_0
    b = -1 * f_minus1 + f_1
    c = 2 * f_minus1 - 5 * f_0 + 4 * f_1 - f_2
    d = -1 * f_minus1 + 3 * f_0 - 3 * f_1 + f_2
    return 0.5 * (a + t * b + t ** 2 * c + t ** 3 * d)

@numba.jit(inline='always', nopython=True, cache=True)
def cpu_bicubic(dx, dy, region):
    # Perform 4 1D interpolations by dx along the rows of region
    b_minus1 = cpu_cubic1d(dx, region[0, 0], region[0, 1], region[0, 2], region[0, 3])
    b_0 = cpu_cubic1d(dx, region[1, 0], region[1, 1], region[1, 2], region[1, 3])
    b_1 = cpu_cubic1d(dx, region[2, 0], region[2, 1], region[2, 2], region[2, 3])
    b_2 = cpu_cubic1d(dx, region[3, 0], region[3, 1], region[3, 2], region[3, 3])
    # perform 1 interpolation by dy along the column of b values
    interpolated_value = cpu_cubic1d(dy, b_minus1, b_0, b_1, b_2)
    return interpolated_value

@numba.jit(inline='always', nopython=True, cache=True)
def get_or_fill(arr, y, x, fill_value):
    """Returns (arr[y, x], False) unless that would be out of bounds,
    or not-a-number, in which case returns (`fill_value`, True)"""
    ny, nx = arr.shape
    if y < ny and y >= 0 and x < nx and x >= 0:
        val = arr[y,x]
        if not np.isnan(val):
            return val, False
    return fill_value, True

@numba.jit(nopython=True, cache=True)
def _interpolate_nonfinite(source_image, dest_image):
    for dest_y in range(dest_image.shape[0]):
        for dest_x in range(dest_image.shape[1]):
            if math.isfinite(source_image[dest_y, dest_x]):
                dest_image[dest_y, dest_x] = source_image[dest_y, dest_x]
                continue
            x_int, y_int = dest_x, dest_y
            cutout = np.zeros((4, 4))
            accumulator = 0.0
            n_good_pix = 0
            for i in range(4):
                for j in range(4):
                    src_y, src_x = y_int + (i - 1), x_int + (j - 1)
                    src_pixval, was_fill = get_or_fill(source_image, src_y, src_x, np.nan)
                    if not was_fill:
                        accumulator += src_pixval
                        n_good_pix += 1
                    cutout[i, j] = src_pixval
            avg_pix_val = accumulator / n_good_pix if n_good_pix > 0 else 0.0
            for i in range(4):
                for j in range(4):
                    if not math.isfinite(cutout[i, j]):
                        cutout[i, j] = avg_pix_val
            dest_image[dest_y, dest_x] = cpu_bicubic(0.0, 0.0, cutout)

    return dest_image


def interpolate_nonfinite(source_image, dest_image=None):
    source_image = source_image.astype('=f8')
    if dest_image is None:
        dest_image = np.zeros_like(source_image)
    return _interpolate_nonfinite(source_image, dest_image)

@numba.jit(nopython=True, cache=True)
def matrix_transform_image(source_image, transform_mtx, dest_image, interpolation_fill_value, missing_fill_value):
    '''
    Parameters
    ----------
    source_image : np.ndarray
    transform_mtx : np.ndarray
    dest_image : np.ndarray
    interpolation_fill_value : float
        When a source image location overlaps a border or NaN value,
        the final interpolated value is calculated using this value
        in place of the undefined pixels
    missing_fill_value : float
        When a transformed image location maps to an out-of-bounds
        location in the source image, or to a location where the
        interpolation domain is >1/2 NaN values, this value
        is used for the destination pixel
    '''
    cutout = np.zeros((4, 4))
    for dest_y in range(dest_image.shape[0]):
        for dest_x in range(dest_image.shape[1]):
            x = transform_mtx[0, 0] * dest_x + transform_mtx[0, 1] * dest_y + transform_mtx[0, 2]
            y = transform_mtx[1, 0] * dest_x + transform_mtx[1, 1] * dest_y + transform_mtx[1, 2]
            x_int = int(math.floor(x))
            x_frac = x - x_int
            y_int = int(math.floor(y))
            y_frac = y - y_int

            n_good_pix = 0
            for i in range(4):
                for j in range(4):
                    src_y, src_x = y_int + (i - 1), x_int + (j - 1)
                    val, was_fill = get_or_fill(source_image, src_y, src_x, interpolation_fill_value)
                    if not was_fill:
                        n_good_pix += 1
                    cutout[i, j] = val
            if n_good_pix > 8:  # 1/2 of the interpolation kernel entries
                dest_image[dest_y, dest_x] = cpu_bicubic(x_frac, y_frac, cutout)
            else:
                dest_image[dest_y, dest_x] = missing_fill_value
    return dest_image

@numba.njit(parallel=True, cache=True)
def matrix_transform_cube(data_cube, transform_mtxes, dest_cube, interpolation_fill_value, missing_fill_value):
    for i in numba.prange(data_cube.shape[0]):
        matrix_transform_image(data_cube[i], transform_mtxes[i], dest_cube[i], interpolation_fill_value, missing_fill_value)
    return dest_cube

def rotate(source_image, angle_deg, dest_image=None, interpolation_fill_value=np.nan, missing_fill_value=np.nan):
    source_image = np.asarray(source_image)
    if dest_image is None:
        dest_image = np.zeros_like(source_image)
    transform_mtx = make_rotation_about_center(source_image.shape, angle_deg)
    matrix_transform_image(source_image, transform_mtx, dest_image, interpolation_fill_value, missing_fill_value)
    return dest_image

def trim_radial_profile(image):
    """Compute radial profile from image center, returning radii
    and profile values for all radii up to the first one where
    the profile value is zero or less

    Returns
    -------
    radii
    profile
    """
    radii, _, _, profile = encircled_energy_and_profile(
        image,
        arr_center(image)
    )
    # exclusive upper bound, this is the r where things went negative
    neg_pix = profile < 0
    if np.any(neg_pix):
        max_r = radii[np.min(np.argwhere(neg_pix))]
        return radii[radii < max_r], profile[radii < max_r]
    else:
        return radii, profile

def template_scale_factor_from_image(image, template_radii, template_profile_values, saturated_pixel_threshold : Optional[float]=None):
    template_min_r_px, template_max_r_px = np.min(template_radii), np.max(template_radii)
    radii, _, _, profile_values = encircled_energy_and_profile(
        image,
        arr_center(image),
        saturated_pixel_threshold=saturated_pixel_threshold
    )
    if len(np.argwhere(np.isnan(profile_values))):
        min_r = radii[np.max(np.argwhere(np.isnan(profile_values)))]
    else:
        min_r = 0
    if np.any(profile_values <= 0):
        max_r = radii[np.min(np.argwhere(profile_values <= 0))]
    else:
        max_r = np.max(radii)
    max_r = min(max_r, template_max_r_px)
    min_r = max(min_r, template_min_r_px)
    sat_mask = (
        (radii < max_r) &
        (radii > min_r)
    )
    template_mask = (
        (template_radii < max_r) &
        (template_radii > min_r)
    )

    scale_factor = np.average(profile_values[sat_mask] / template_profile_values[template_mask])
    return scale_factor

def compute_template_scale_factors(
    data_cube : np.ndarray,
    template_array : np.ndarray,
    saturated_pixel_threshold : float,
):
    radii, profile = trim_radial_profile(template_array)

    return np.array([
        template_scale_factor_from_image(x, radii, profile, saturated_pixel_threshold=saturated_pixel_threshold)
        for x in data_cube
    ])

def _make_monotonic_angles_deg(angles_deg):
    angles_deg = angles_deg - np.min(angles_deg)  # shift -180 to 180 into 0 to 360
    angles_deg = np.unwrap(angles_deg, period=360)
    angles_deg -= angles_deg[0]
    if angles_deg[1] - angles_deg[0] < 0:
        angles_deg = -angles_deg
    return angles_deg

@dataclass
class BaseAngularRangeSpec:
    angle_deg_column_name : str
    def to_values_and_delta(self, obs_table):
        derotation_angles = _make_monotonic_angles_deg(obs_table[self.angle_deg_column_name])
        return self.angles_to_values_and_delta(derotation_angles)

@dataclass
class PixelRotationRangeSpec(BaseAngularRangeSpec):
    delta_px : float
    r_px : float
    def angles_to_values_and_delta(self, derotation_angles):
        values = np.deg2rad(derotation_angles) * self.r_px
        return values, self.delta_px
@dataclass
class AngleRangeSpec(BaseAngularRangeSpec):
    delta_deg : float
    def angles_to_values_and_delta(self, derotation_angles):
        return derotation_angles, self.delta_deg
@dataclass
class FrameIndexRangeSpec:
    n_frames : int
    def to_values_and_delta(self, obs_table):
        return np.arange(len(obs_table)), self.n_frames

@dataclass
class WallTimeRangeSpec:
    delta_t_sec : float
    time_secs_col : str

    def to_values_and_delta(self, obs_table):
        return obs_table[self.time_secs_col], self.delta_t_sec


RotationRange = Union[PixelRotationRangeSpec, AngleRangeSpec, FrameIndexRangeSpec, WallTimeRangeSpec]

def combine(cube : np.ndarray, operation: constants.CombineOperation, normalize : Optional[constants.NormalizeToUnit] = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if operation is constants.CombineOperation.MEAN:
            out_image = np.nanmean(cube, axis=0)
        elif operation is constants.CombineOperation.SUM:
            out_image = np.nansum(cube, axis=0)
        elif operation is constants.CombineOperation.MEDIAN:
            out_image = np.nanmedian(cube, axis=0)
        elif operation is constants.CombineOperation.STD:
            out_image = np.nanstd(cube, axis=0)
        else:
            raise ValueError("Supported operations: sum, mean, median")
    if normalize is constants.NormalizeToUnit.PEAK:
        out_image /= np.nanmax(out_image)
    elif normalize is constants.NormalizeToUnit.TOTAL:
        out_image /= np.nansum(out_image)
    return out_image

def combine_ranges(obs_sequences, obs_table, range_spec, operation: constants.CombineOperation = constants.CombineOperation.MEAN):
    """Using derotation angles and a range specified as `range_spec`, combine chunks of
    adjacent frames from `data_cube` and return summed data and averaged derotation angles
    corresponding to the new frames

    Returns
    -------
    out_seqs : list[np.ndarray]
        List of arrays with coadded frames
    outangles : np.ndarray
        Array with averaged derotation angles corresponding to these frames
    out_metadata : Optional[np.ndarray]
        If metadata is a record array, new metadata for the output sequences
    """
    values, delta = range_spec.to_values_and_delta(obs_table)
    target_idx = 0
    chunk_start_idx = 0
    out_seqs = [np.zeros_like(seq) for seq in obs_sequences]
    n_data_cubes = len(obs_sequences)
    n_obs = obs_sequences[0].shape[0]

    if obs_table is not None:
        out_metadata = np.zeros_like(obs_table)
    else:
        out_metadata = None
    def _do_chunk(chunk):
        if obs_table is not None:
            for field in obs_table.dtype.fields:
                field_dtype = obs_table.dtype[field]
                if np.issubdtype(field_dtype, np.floating) or np.issubdtype(field_dtype, np.integer):
                    out_metadata[target_idx][field] = combine(obs_table[chunk][field], operation)
                else:
                    # take first value from chunk since we don't know how to interpolate
                    # unknown types
                    out_metadata[target_idx][field] = obs_table[chunk_start_idx][field]
        for i in range(n_data_cubes):
            if operation is constants.CombineOperation.MEAN:
                out_seqs[i][target_idx] = obs_sequences[i][chunk].mean(axis=0)

    for frame_idx in range(n_obs):
        if values[frame_idx] - values[chunk_start_idx] >= delta:
            chunk = slice(chunk_start_idx, frame_idx)
            _do_chunk(chunk)
            target_idx += 1
            chunk_start_idx = frame_idx

    # handle the last of the observations
    if n_obs - chunk_start_idx > 0:
        _do_chunk(slice(chunk_start_idx, None))

    out_seqs = [np.copy(outcube[:target_idx+1]) for outcube in out_seqs]
    out_metadata = out_metadata[:target_idx+1]
    return out_seqs, out_metadata


@numba.njit(cache=True)
def shift2(image, dx, dy, output_shape=None, interpolation_fill_value=np.nan, missing_fill_value=np.nan, anchor_to_center=True):
    """Shift image by dx, dy with bicubic interpolation
    Direction convention: feature at (0, 0) moves to (dx, dy)
    If ``output_shape`` is larger than ``image.shape``, image will be drawn into the center
    of an array of ``output_shape`` when anchor is 'center', or the lower left when anchor
    is not 'center'
    """
    if output_shape is not None and anchor_to_center:
        # compute center-to-center displacement such that
        # supplying dx == dy == 0.0 will be a no-op (aside
        # from changing shape)
        orig_ctr_x, orig_ctr_y = (image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2
        new_ctr_x, new_ctr_y = (output_shape[1] - 1) / 2, (output_shape[0] - 1) / 2
        base_dx, base_dy = new_ctr_x - orig_ctr_x, new_ctr_y - orig_ctr_y
    else:
        base_dx = base_dy = 0
    xform = translation_matrix(-(dx + base_dx), -(dy + base_dy))
    output = np.zeros_like(image) if output_shape is None else np.zeros(output_shape, dtype=image.dtype)
    matrix_transform_image(image, xform, output, interpolation_fill_value, missing_fill_value)
    return output

@numba.njit(
    parallel=True
)
def _derotate_cube(cube, derotation_angles, output, interpolation_fill_value, missing_fill_value):
    for idx in numba.prange(cube.shape[0]):
        transform_mtx = make_rotation_about_center(cube[idx].shape, derotation_angles[idx])
        matrix_transform_image(cube[idx], transform_mtx, output[idx], interpolation_fill_value, missing_fill_value)
    return output

def derotate_cube(cube, derotation_angles, interpolation_fill_value=0, missing_fill_value=np.nan, output=None):
    """Rotate each plane of `cube` by the corresponding entry
    in `derotation_angles`, with positive angle interpreted as
    deg to rotate E of N when N +Y and E +X (which is CCW
    when 0, 0 at lower left)

    Parameters
    ----------
    cube : array (planes, xpix, ypix)
    derotation_angles : array (planes,)
        Angles to rotate each image by counter-clockwise (when the pixel
        Y coordinate increases towards the top of the image), in degrees.
    fill_value : float

    Returns
    -------
    output : array (xpix, ypix)
    """
    if cube.shape[0] != derotation_angles.shape[0]:
        raise ValueError("Number of cube planes and derotation angles must match")
    if not np.issubdtype(derotation_angles.dtype, np.floating):
        derotation_angles = derotation_angles.astype(np.float32)
    if not np.issubdtype(cube.dtype, np.floating):
        cube = cube.astype(np.float32)

    if output is None:
        output = np.zeros_like(cube)
    else:
        if output.shape != cube.shape:
            raise ValueError(f"Got {output.shape=}, mismatched with {cube.shape=}")
    _derotate_cube(cube, derotation_angles, output, interpolation_fill_value, missing_fill_value)
    return output
