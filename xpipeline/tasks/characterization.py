import itertools
import dataclasses
import logging
import warnings
import numpy as np
import astropy.units as u
import pandas as pd
from typing import List, Optional, Union
from dataclasses import dataclass
from numpy.lib.recfunctions import drop_fields, append_fields
import numba
from numba import njit
import math
from tqdm import tqdm
from scipy.signal import fftconvolve
from scipy.interpolate import griddata, interp1d

try:
    from pyfftw.interfaces import numpy_fft as fft
except ImportError:
    fft = np.fft

from . import improc
from .. import core

log = logging.getLogger()

@dataclass
class Location:
    r_px: float
    pa_deg: float

    def display(self, ax=None):
        return show_r_pa(self.r_px, self.pa_deg, 0, 0)


@dataclass
class CompanionSpec(Location):
    scale: float

    @classmethod
    def from_str(cls, value):
        prefactor = 1
        if len(value.split(",")) == 4:
            inv, scale_str, r_px_str, pa_deg_str = value.split(",")
            if inv == "invert":
                prefactor = -1
        else:
            scale_str, r_px_str, pa_deg_str = value.split(",")

        if scale_str == "?":
            scale_str = "0"
        return cls(
            scale=prefactor * float(scale_str),
            r_px=float(r_px_str),
            pa_deg=float(pa_deg_str),
        )

@dataclass
class Detection(Location):
    snr: float


@dataclass
class RecoveredSignal(CompanionSpec, Detection):
    contrast_estimate_5sigma: float


@dataclass
class TemplateSignal:
    signal : np.ndarray
    scale_factors : Union[np.ndarray,float]

def generate_signal(
    shape: tuple,
    r_px: float,
    pa_deg: float,
    template: np.ndarray,
    derotation_angles: np.ndarray = None,
    template_scale_factors: Union[np.ndarray,float,None] = None,
    outcube: Union[np.ndarray,None] = None,
    scale: float = 1.0
):
    if np.any(~np.isfinite(template)):
        raise ValueError("Non-finite values in template")
    if outcube is None:
        outcube = np.zeros(shape, dtype=template.dtype)
    n_obs, frame_shape = shape[0], shape[1:]

    if template_scale_factors is None:
        template_scale_factors = np.ones(n_obs)
    if np.isscalar(template_scale_factors):
        template_scale_factors = np.repeat(np.array([template_scale_factors]), n_obs)
    if derotation_angles is None:
        derotation_angles = np.zeros(n_obs)
    else:
        assert derotation_angles.shape[0] == shape[0]

    theta = np.deg2rad(90 + pa_deg - derotation_angles)
    for i in range(n_obs):
        dx = r_px * np.cos(theta[i])
        dy = r_px * np.sin(theta[i])
        outcube[i] += scale * template_scale_factors[i] * improc.shift2(template, dx, dy, output_shape=frame_shape, interpolation_fill_value=0.0, missing_fill_value=0.0)
    return outcube

def generate_signals(
    shape: tuple,
    specs: list[CompanionSpec],
    template: np.ndarray,
    derotation_angles: np.ndarray = None,
    template_scale_factors: Union[np.ndarray,float,None] = None,
):
    '''Inject signals for companions specified using
    optional derotation angles to *counter*rotate the coordinates
    before injection, such that rotation CCW (e.g. by `derotate_cube`)
    aligns 0 deg PA with +Y

    Parameters
    ----------
    shape : tuple[int,int,int]
    specs : list[CompanionSpec]
    template : np.ndarray
    derotation_angles : Optional[np.ndarray]
    template_scale_factors : Union[np.ndarray,float,None]
        Scale factor relative to 1.0 being the average brightness
        of the primary over the observation, used to scale the
        template image to reflect particularly sharp or poor
        AO correction

    Returns
    -------
    outcube : np.ndarray
    '''
    outcube = np.zeros(shape, dtype=template.dtype)
    for spec in specs:
        # generate in-place in outcube
        generate_signal(
            outcube.shape,
            spec.r_px,
            spec.pa_deg,
            template,
            derotation_angles,
            template_scale_factors,
            scale=spec.scale,
            outcube=outcube
        )
    return outcube


def inject_signals(
    cube: np.ndarray,
    specs: List[CompanionSpec],
    template: np.ndarray,
    angles: np.ndarray = None,
    template_scale_factors: Union[np.ndarray,float,None] = None,
    saturation_threshold: Optional[float] = None,
):
    '''Generate signals using `generate_signals` (see docstring) and add to
    an input `cube` with possible saturation (implemented as a simple clipping of
    values to some level)
    '''
    signal_only_cube = generate_signals(cube.shape, specs, template, angles, template_scale_factors)
    outcube = cube + signal_only_cube
    if saturation_threshold is not None:
        outcube = np.clip(outcube, a_min=None, a_max=saturation_threshold)
        signal_only_cube = np.clip(signal_only_cube, a_min=None, a_max=saturation_threshold)
    return outcube, signal_only_cube

def specs_to_table(specs, spec_type, value_type=np.float32):
    fields = dataclasses.fields(spec_type)
    dtype = [
        (field.name, value_type)
        for field in fields
    ]
    tbl = np.zeros(len(specs), dtype=dtype)
    for idx, spec in enumerate(specs):
        for fld in fields:
            tbl[idx][fld.name] = getattr(spec, fld.name)
    return tbl


def table_to_specs(table, spec_type):
    specs = []
    fields = dataclasses.fields(spec_type)
    for row in table:
        specs.append(spec_type(**{fld.name: row[fld.name] for fld in fields}))
    return specs

def recover_signals(
    image: np.ndarray,
    specs: List[CompanionSpec],
    aperture_diameter_px: float,
    apertures_to_exclude: int,
) -> List[Detection]:
    signals = []
    for spec in specs:
        _, vals = reduce_apertures(
            image,
            spec.r_px,
            spec.pa_deg,
            aperture_diameter_px,
            np.sum,
            exclude_nearest=apertures_to_exclude,
        )
        snr = calc_snr_mawet(vals[0], vals[1:])
        contrast_estimate_5sigma = 5 * (spec.scale / snr)
        signals.append(RecoveredSignal(
            scale=spec.scale,
            r_px=spec.r_px,
            pa_deg=spec.pa_deg,
            snr=snr,
            contrast_estimate_5sigma=contrast_estimate_5sigma
        ))
    return signals

@njit(cache=True)
def _simple_aperture_locations(r_px, pa_deg, resolution_element_px, xcenter=0, ycenter=0, good_pixel_mask=None):
    circumference = 2 * r_px * np.pi
    aperture_pixel_diameter = resolution_element_px
    n_apertures = int(circumference / aperture_pixel_diameter)
    start_theta = np.deg2rad(pa_deg + 90)
    delta_theta = np.deg2rad(360 / n_apertures)
    idxs = np.arange(n_apertures)
    thetas = start_theta + idxs * delta_theta
    offset_x = r_px * np.cos(thetas)
    offset_y = r_px * np.sin(thetas)
    good_offsets = np.zeros(n_apertures, dtype=np.bool8)
    if good_pixel_mask is not None:
        mask_center = (good_pixel_mask.shape[0] - 1) / 2, (good_pixel_mask.shape[1] - 1) / 2
        for idx in idxs:
            maskpixy, maskpixx = int(offset_y[idx] + ycenter + mask_center[0]), int(offset_x[idx] + xcenter + mask_center[1])
            mask_pixel = good_pixel_mask[maskpixy, maskpixx]
            if not mask_pixel:
                continue
            else:
                good_offsets[idx] = True
    else:
        good_offsets[:] = True
    return np.stack((offset_x[good_offsets] + xcenter, offset_y[good_offsets] + ycenter), axis=-1)


# TODO eliminate some redundancy with simple aperture locator
def generate_probes(iwa_px : float, owa_px : float, n_radii : int, spacing_px : float, scales : list[float], starting_pa_deg : float = 0):
    '''Generator returning CompanionSpec objects for
    radii / PA / contrast scales that cover the region from iwa to owa

    When config.n_radii == 1, only iwa_px matters
    '''
    radii_dpx = (owa_px - iwa_px) / (n_radii - 1) if n_radii > 1 else 0
    for i in range(n_radii):
        r_px = iwa_px + i * radii_dpx
        circumference = np.pi * 2 * r_px
        n_probes = int(circumference // spacing_px)
        angles_ddeg = 360 / n_probes
        for j in range(n_probes):
            pa_deg = (starting_pa_deg + j * angles_ddeg) % 360.0
            for scl in scales:
                yield CompanionSpec(r_px, pa_deg, scl)

def simple_aperture_locations(r_px, pa_deg, resolution_element_px, exclude_nearest=0,
                              exclude_planet=False, xcenter=0, ycenter=0, good_pixel_mask=None):
    """Returns (x_center, y_center) for apertures in a ring of
    radius `r_px` starting at angle `pa_deg` E of N. The first (x, y)
    pair gives the planet location (signal aperture)."""
    locs = _simple_aperture_locations(r_px, pa_deg, resolution_element_px, xcenter, ycenter, good_pixel_mask)
    if exclude_nearest != 0:
        locs = np.concatenate([locs[0][np.newaxis, :], locs[1 + exclude_nearest:-exclude_nearest]])
    if exclude_planet:
        locs = locs[1:]
    return locs

def r_pa_to_x_y(r, pa_deg, xcenter=0, ycenter=0, derotation_angle_deg=0):
    '''Convert (radius, PA (in deg)) to (x, y) cartesian coordinates.
    Derotation angle is subtracted from PA such that when these (x, y) coordinates
    are rotated CCW (when +Y is up and +X is right) by `derotation_angle_deg`
    this point lands at the (x, y) you'd get for the same arguments and a
    `derotation_angle_deg` == 0.0.

    In other words, if you have derotation angles that place north-up-east-left
    and images in the un-derotated / pupil-stabilized frame, this argument
    produces the pixel coordinates in the un-derotated frame given the appropriate
    derotation angle for that frame.
    '''
    return (
       r * np.cos(np.deg2rad(90 + pa_deg - derotation_angle_deg)) + xcenter,
       r * np.sin(np.deg2rad(90 + pa_deg - derotation_angle_deg)) + ycenter
    )

def x_y_to_r_pa(x, y, xcenter, ycenter):
    dx = x - xcenter
    dy = y - ycenter
    pa_deg = np.rad2deg(np.arctan2(dy, dx)) - 90
    r_px = np.sqrt(dx**2 + dy**2)
    if np.any(pa_deg < 0):
        pa_deg = 360 + pa_deg
    return r_px, pa_deg


def show_r_pa(r_px, pa_deg, xcenter, ycenter, ax=None, **kwargs):
    """Overlay an arrow on the current (or provided) axes from
    xcenter, ycenter to r_px, pa_deg. Other arguments passed
    through to ax.arrow.
    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    dx, dy = r_pa_to_x_y(r_px, pa_deg, xcenter, ycenter)
    return ax.arrow(xcenter, ycenter, dx, dy, **kwargs)

def show_simple_aperture_locations(
    image,
    resolution_element_px,
    r_px,
    pa_deg,
    exclude_nearest=0,
    exclude_planet=False,
    ax=None,
    colorbar=True,
):
    """Plot `image` and overplot the circular apertures of diameter
    `resolution_element_px` in a ring at radius `r_px`
    starting at `pa_deg` E of N.
    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    ctr = (image.shape[0] - 1) / 2
    im = ax.imshow(image)
    if colorbar:
        plt.colorbar(im)
    ax.axhline(ctr, color="w", linestyle=":")
    ax.axvline(ctr, color="w", linestyle=":")
    planet_dx, planet_dy = r_pa_to_x_y(r_px, pa_deg, xcenter=0, ycenter=0)
    ax.arrow(ctr, ctr, planet_dx, planet_dy, color="w", lw=2)
    for offset_x, offset_y in simple_aperture_locations(
        r_px,
        pa_deg,
        resolution_element_px,
        exclude_nearest=exclude_nearest,
        exclude_planet=exclude_planet,
    ):
        ax.add_artist(
            plt.Circle(
                (ctr + offset_x, ctr + offset_y),
                radius=resolution_element_px / 2,
                edgecolor="orange",
                facecolor="none",
            )
        )
    return im


@njit(numba.float64(numba.float64, numba.float64[:]), cache=True)
def _calc_snr_mawet(signal, noises):
    noise_total = 0
    num_noises = 0
    for noise in noises:
        if np.isfinite(noise):
            noise_total += noise
            num_noises += 1
    if num_noises == 0:
        return np.nan
    noise_avg = noise_total / num_noises
    numerator = signal - noise_avg
    # second pass thru to compute stddev now we have the noise avg
    stddev_inner_accum = 0
    for noise in noises:
        if np.isfinite(noise):
            meansub = (noise - noise_avg)
            stddev_inner_accum += meansub * meansub
    if stddev_inner_accum == 0:
        return np.nan
    noise_stddev = math.sqrt(stddev_inner_accum / num_noises)
    denominator = noise_stddev * math.sqrt(1 + 1 / num_noises)
    return numerator / denominator


def calc_snr_mawet(signal, noises):
    return _calc_snr_mawet(float(signal), np.asarray(noises, dtype=np.float64))


def reduce_apertures(
    image,
    r_px,
    starting_pa_deg,
    resolution_element_px,
    operation,
    exclude_nearest=0,
    exclude_planet=False,
    good_pixel_mask=None,
):
    """apply `operation` to the pixels within radius `resolution_element_px`/2 of the centers
    of the simple aperture locations for a planet at `r_px` and `starting_pa_deg`, returning
    the locations and the results as a tuple with the first location and result corresponding
    to the planet aperture"""
    center = (image.shape[0] - 1) / 2, (image.shape[1] - 1) / 2
    yy, xx = improc.cartesian_coords(center, image.shape)
    locations = list(
        simple_aperture_locations(
            r_px,
            starting_pa_deg,
            resolution_element_px,
            exclude_nearest=exclude_nearest,
            exclude_planet=exclude_planet,
            good_pixel_mask=good_pixel_mask
        )
    )
    simple_aperture_radius = resolution_element_px / 2
    results = []
    for offset_x, offset_y in locations:

        dist = np.sqrt((xx - offset_x) ** 2 + (yy - offset_y) ** 2)
        mask = dist <= simple_aperture_radius
        result = operation(image[mask] / np.count_nonzero(mask & np.isfinite(image)))
        if not np.isnan(result):
            results.append(result)
    return locations, results


def calculate_snr(image, r_px, pa_deg, resolution_element_px, exclude_nearest, good_pixel_mask=None):
    _, results = reduce_apertures(
        image,
        r_px,
        pa_deg,
        resolution_element_px,
        np.nansum,
        exclude_nearest=exclude_nearest,
        good_pixel_mask=good_pixel_mask
    )
    return calc_snr_mawet(results[0], results[1:])

@njit(cache=True)
def snr_from_convolution(convolved_image, loc_rho, loc_pa_deg, aperture_diameter_px, exclude_nearest, good_pixel_mask=None):
    if good_pixel_mask is None:
        good_pixel_mask = np.ones(convolved_image.shape) == 1
    height, width = convolved_image.shape
    yc, xc = (height - 1) / 2, (width - 1) / 2
    locs = _simple_aperture_locations(
        loc_rho, loc_pa_deg, aperture_diameter_px, xcenter=xc, ycenter=yc
    )
    n_apertures = locs.shape[0]
    signal_x, signal_y = locs[0]
    signal_y_pixel, signal_x_pixel = round(signal_y), round(signal_x)
    if good_pixel_mask[signal_y_pixel, signal_x_pixel] and np.isfinite(convolved_image[signal_y_pixel, signal_x_pixel]):
        signal = convolved_image[signal_y_pixel, signal_x_pixel]
    else:
        return np.nan, np.nan

    n_noises = n_apertures - 1 - 2 * exclude_nearest
    if n_noises < 2:
        # note this is checked for in the wrapper
        # as raising exceptions will leak memory allocated in numba functions
        raise ValueError("Reached radius where only a single noise aperture is available for estimation. Change exclude_nearest or iwa_px.")
    first_noise_offset = 1 + exclude_nearest
    noises = np.zeros(n_noises)
    for i in range(n_noises):
        nidx = first_noise_offset + i
        noise_x, noise_y = locs[nidx]
        noise_y_pixel, noise_x_pixel = round(noise_y),round(noise_x)
        if good_pixel_mask[noise_y_pixel, noise_x_pixel]:
            noises[i] = convolved_image[noise_y_pixel, noise_x_pixel]
        else:
            noises[i] = np.nan
    snr = _calc_snr_mawet(signal, noises)
    return snr, signal



def working_radii_from_aperture_spacing(image_shape, aperture_diameter_px, exclude_nearest, data_min_r_px=None, data_max_r_px=None):
    aperture_r = aperture_diameter_px / 2
    iwa_px = data_min_r_px + aperture_diameter_px / 2 if data_min_r_px is not None else None
    owa_px = data_max_r_px - aperture_diameter_px / 2 if data_max_r_px is not None else None

    # How close in can we really go?
    num_excluded = exclude_nearest * 2 + 1  # some on either side, plus signal aperture itself
    min_apertures = num_excluded + 2
    real_iwa_px = (min_apertures * aperture_diameter_px) / (2 * np.pi) + aperture_r
    if iwa_px is None or iwa_px < real_iwa_px:
        if iwa_px is not None:
            warnings.warn(f'Requested {iwa_px=} < {real_iwa_px}, but at least two noise apertures are needed at the IWA for sensible output. Using {real_iwa_px} instead.')
        min_r = real_iwa_px
    else:
        min_r = iwa_px
    # How far out can we really go?
    real_owa_px = improc.max_radius(improc.arr_center(image_shape), image_shape) - aperture_r
    if owa_px is None or owa_px > real_owa_px:
        if owa_px is not None:
            warnings.warn(f'Requested {owa_px=} > {real_owa_px} but pixel values outside the image are unknown. Using {real_owa_px} instead.')
        max_r = real_owa_px
    else:
        max_r = owa_px
    return min_r, max_r

def tophat_kernel(aperture_diameter_px):
    kernel_npix = math.ceil(aperture_diameter_px) + 2  # one pixel border both sides for no reason
    kernel = np.zeros((kernel_npix, kernel_npix))
    kernel_rho, _ = improc.polar_coords(improc.arr_center(kernel), kernel.shape)
    kernel[kernel_rho <= aperture_diameter_px/2] = 1
    return kernel

@njit(parallel=True, cache=True)
def _calc_snr_image(convolved_image, rho, theta, mask, aperture_diameter_px, exclude_nearest, snr_image_out):
    height, width = convolved_image.shape
    for y in numba.prange(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            loc_rho, loc_theta = rho[y, x], theta[y, x]
            loc_pa_deg = np.rad2deg(loc_theta) - 90

            calculated_snr, _ = snr_from_convolution(convolved_image, loc_rho, loc_pa_deg, aperture_diameter_px, exclude_nearest)
            snr_image_out[y, x] = calculated_snr
    return snr_image_out

def calc_snr_image(image, aperture_diameter_px, data_min_r_px, data_max_r_px, exclude_nearest, convolve=True):
    """Compute simple aperture photometry SNR at each pixel and return an image
    with the SNR map"""
    iwa_px, owa_px = working_radii_from_aperture_spacing(image.shape, aperture_diameter_px, exclude_nearest, data_min_r_px=data_min_r_px, data_max_r_px=data_max_r_px)
    rho, theta = improc.polar_coords(improc.arr_center(image), image.shape)
    mask = (rho >= iwa_px) & (rho <= owa_px)
    kernel = tophat_kernel(aperture_diameter_px)
    image = image.copy()
    image[np.isnan(image)] = 0
    if convolve:
        convolved_image = fftconvolve(image, kernel, mode='same')
    else:
        convolved_image = image
    # return convolved_image
    snr_image = np.zeros_like(convolved_image)
    _calc_snr_image(convolved_image, rho, theta, mask, aperture_diameter_px, exclude_nearest, snr_image)
    return snr_image, (iwa_px, owa_px)


def locate_snr_peaks(image, aperture_diameter_px, data_min_r_px, data_max_r_px, exclude_nearest, snr_threshold):
    """Compute SNR from `aperture_diameter_px` simple apertures"""
    snr_image, (iwa_px, owa_px) = calc_snr_image(image, aperture_diameter_px, data_min_r_px, data_max_r_px, exclude_nearest)
    im_ctr = improc.arr_center(snr_image)
    rho, _ = improc.polar_coords(im_ctr, snr_image.shape)
    mask = (rho >= iwa_px) & (rho <= owa_px) & (snr_image > snr_threshold)
    peaks_mask = local_maxima(snr_image)
    mask &= peaks_mask
    maxima_locs = np.argwhere(mask)

    maxima = []
    for loc in maxima_locs:
        yloc, xloc = loc
        r_px, pa_deg = x_y_to_r_pa(xloc, yloc, im_ctr[1], im_ctr[0])
        snr = snr_image[yloc,xloc]
        maxima.append((
            snr,
            Detection(r_px=r_px, pa_deg=pa_deg, snr=snr)
        ))
    maxima.sort()
    return [x[1] for x in maxima[::-1]], (iwa_px, owa_px), snr_image

@numba.stencil(neighborhood=((-2, 2), (-2, 2)))
def local_maxima(image) -> np.ndarray:
    """Using a 5x5 neighborhood around each pixel, fill a mask array with True where the pixel is a local maximum"""
    this_pixel = image[0,0]
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i,j) == (0,0):
                continue
            if image[i,j] > this_pixel:
                return False
    return True

def sigma_mad(points):
    '''Estimate sigma with the median absolute deviation
    to improve robustness to outliers

    Parameters
    ----------
    points : array-like 1D

    Returns
    -------
    sigma_estimate : float
        Estimated standard deviation from median absolute deviation
    '''
    xp = core.get_array_module(points)
    return 1.48 * xp.median(xp.abs(points - xp.median(points)))

def _scale_trap_noise(tbl, ring_px, exclude_rows_mask):
    emp_snr = tbl['model_coeff'].copy()
    radii = []
    sigmas = []
    for r_px in np.unique(tbl['r_px']):
        ring_mask = np.abs(tbl['r_px'] - r_px) < ring_px
        if exclude_rows_mask is not None:
            ring_values = tbl['model_coeff'][ring_mask & ~exclude_rows_mask]
        else:
            ring_values = tbl['model_coeff'][ring_mask]
        new_sigma = sigma_mad(ring_values)
        radii.append(r_px)
        sigmas.append(new_sigma)
        emp_snr[tbl['r_px'] == r_px] /= new_sigma
    return emp_snr, np.asarray(radii), np.asarray(sigmas)

def points_to_map(xs, ys, zs, mask):
    yy, xx = np.indices(mask.shape, dtype=float)
    post_grid_mask = mask == 1
    points = np.stack((ys, xs), axis=-1)
    newpoints = np.stack((yy.flatten(), xx.flatten()), axis=-1)
    outim = griddata(points, zs, newpoints).reshape(mask.shape)
    outim[~post_grid_mask] = np.nan
    return outim

def detection_map_from_table(tbl, coverage_mask, ring_px=3, exclude_rows_mask=None, **kwargs):
    '''
    Parameters
    ----------
    tbl : recarray-like
        Record array with at least 'model_coeff', 'r_px',
        'x', 'y' columns
    coverage_mask : np.ndarray
    ring_px : int
        Half-width of ring centered on each point's r_px
        within which sigma is estimated from model_coeff
        values
    **kwargs : dict[str,float]
        Equality constraints on other columns of `tbl`
        (e.g. ``k_modes=100`` implies "select rows with
        ``k_modes == 100``)

    Returns
    -------
    detection_map : np.ndarray
        Detection strength in units of sigma
        (estimated at that radius)
    '''
    for kwarg in kwargs:
        mask = tbl[kwarg] == kwargs[kwarg]
        tbl = tbl[mask]
        if exclude_rows_mask is not None:
            exclude_rows_mask = exclude_rows_mask[mask]
    if not len(tbl):
        raise ValueError(f"No points matching {kwargs}")
    yy, xx = np.indices(coverage_mask.shape, dtype=float)
    post_grid_mask = coverage_mask == 1
    points = np.stack((tbl['y'], tbl['x']), axis=-1)
    newpoints = np.stack((yy.flatten(), xx.flatten()), axis=-1)
    emp_snr, radii, sigmas = _scale_trap_noise(tbl, ring_px, exclude_rows_mask)
    outim = griddata(points, emp_snr, newpoints).reshape(coverage_mask.shape)
    outim[~post_grid_mask] = np.nan
    return outim, emp_snr, interp1d(radii, sigmas)

def detection_map_cube_from_table(tbl, coverage_mask, ring_px=3, **kwargs):
    static_kwargs = {}
    varying_kwargs = []
    for kwarg in kwargs:
        try:
            seq = np.unique(kwargs[kwarg])
            varying_kwargs.append((kwarg, seq))
        except TypeError:
            static_kwargs[kwarg] = kwargs[kwarg]

    filter_values = list(itertools.product(*[seq for name, seq in varying_kwargs]))
    filter_names = [name for name, seq in varying_kwargs]
    detection_map_cube = np.zeros((len(filter_values),) + coverage_mask.shape)
    filters = []
    for idx, row in enumerate(filter_values):
        filter_kwargs = {}
        for col_idx, name in enumerate(filter_names):
            filter_kwargs[name] = row[col_idx]
        filters.append(filter_kwargs)
        final_kwargs = filter_kwargs.copy()
        final_kwargs.update(static_kwargs)
        detection_map_cube[idx] = detection_map_from_table(
            tbl,
            coverage_mask,
            ring_px=ring_px,
            **final_kwargs
        )
    return filters, detection_map_cube

def normalize_snr_for_grid(
    tbl,
    group_by_colname : str,
    snr_colname : str,
    injected_scale_colname : str,
    hyperparameter_colnames : list[str],
):
    # retain the original SNR value to look at if we need it
    # it somehow got converted to masked_array, so np.array() changes it back
    tbl = np.array(append_fields(np.ascontiguousarray(tbl), 'unnormalized_snr', tbl[snr_colname]))

    no_injection_mask = tbl[injected_scale_colname] == 0.0
    detections = tbl[no_injection_mask]
    config_params = [group_by_colname] + hyperparameter_colnames
    configurations = np.unique(detections[config_params])
    for config in tqdm(configurations):
        mask = np.ones(len(tbl), dtype=bool)
        for colname in config_params:
            mask &= tbl[colname] == config[colname]
        noise_factor = np.nanstd(tbl[no_injection_mask & mask][snr_colname])
        tbl[snr_colname][mask & ~no_injection_mask] = tbl[mask & ~no_injection_mask][snr_colname] / noise_factor
    return tbl


def summarize_grid(grid_df : pd.DataFrame,
                   r_px_colname : str,
                   pa_deg_colname : str,
                   snr_colname : str,
                   injected_scale_colname : str,
                   hyperparameter_colnames : list[str],
                   min_snr_for_injection: float):
    injection_rows_mask = (grid_df[injected_scale_colname] > 0)
    injections = grid_df[injection_rows_mask & (grid_df[snr_colname] > min_snr_for_injection)].copy()
    detections = grid_df[~injection_rows_mask].copy()
    detections.set_index([r_px_colname, pa_deg_colname])
    if len(injections) == 0:
        raise ValueError(f"No rows in injections table after filtering for nonzero injection and {min_snr_for_injection=}")
    injections['contrast_limit_5sigma'] = injections[injected_scale_colname] / injections[snr_colname] * 5
    grouping_colnames = [r_px_colname, pa_deg_colname]

    best_params : pd.DataFrame = injections.groupby(
        by=[r_px_colname, pa_deg_colname],
    ).apply(
        lambda grp: grp[grp['contrast_limit_5sigma'] == grp['contrast_limit_5sigma'].min()].iloc[:1]
    ).droplevel([r_px_colname, pa_deg_colname])

    detections = best_params.merge(detections, on=grouping_colnames + hyperparameter_colnames, suffixes=('_best', None))
    return best_params, detections

def apparent_mag(absolute_mag, d):
    '''Scale an `absolute_mag` to an apparent magnitude using
    the distance modulus for `d`
    '''
    if not d.unit.is_equivalent(u.pc):
        raise ValueError(f"d must be units of distance, got {d.unit}")
    return 5 * np.log10(d / (10 * u.pc)) + absolute_mag

def absolute_mag(apparent_mag, d):
    '''Scale an `apparent_mag` at distance `d` to `d` = 10 pc
    '''
    if not d.unit.is_equivalent(u.pc):
        raise ValueError(f"d must be units of distance, got {d.unit}")
    return apparent_mag - 5 * np.log10(d / (10 * u.pc))

def contrast_to_deltamag(contrast):
    '''contrast as :math:`10^{-X}` to delta magnitude'''
    return -2.5 * np.log10(contrast)

def deltamag_to_contrast(deltamag):
    '''delta mag as an exponent in :math:`10^{-X}`'''
    return np.power(10, deltamag / -2.5)

def stddev_to_mag_err(value, stddev):
    # f = a log_10 (b * A)
    # sigma_f^2 = (a * sigma_A / (A * ln(10)))^2
    # sigma_f = |a * sigma_A / (A * ln(10))|
    # for magnitudes a = -2.5, b = 1
    return np.abs(-2.5 * stddev / (value * np.log(10)))

def lambda_over_d_to_arcsec(lambda_over_d, wavelength, d):
    unit_lambda_over_d = (wavelength.to(u.m) / d.to(u.m)).si.value * u.radian
    return (lambda_over_d * unit_lambda_over_d).to(u.arcsec)

def arcsec_to_lambda_over_d(arcsec, wavelength, d):
    unit_lambda_over_d = ((wavelength.to(u.m) / d.to(u.m)).si.value * u.radian).to(
        u.arcsec
    )
    lambda_over_d = (arcsec / unit_lambda_over_d).si
    return lambda_over_d
