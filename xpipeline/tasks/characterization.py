import logging
import warnings
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import numba
from numba import njit
import math
from scipy.signal import fftconvolve

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


def inject_signals(
    cube: np.ndarray,
    angles: np.ndarray,
    specs: List[CompanionSpec],
    template: np.ndarray,
    template_scale_factors: Optional[np.ndarray] = None,
    saturation_threshold: Optional[float] = None,
):
    if template_scale_factors is None:
        template_scale_factors = np.ones(cube.shape[0])
    frame_shape = cube.shape[1:]
    outcube = cube.copy()

    for frame_idx in range(cube.shape[0]):
        for spec in specs:
            if spec.scale == 0:
                continue
            theta = np.deg2rad(90 + spec.pa_deg - angles[frame_idx])
            dx, dy = spec.r_px * np.cos(theta), spec.r_px * np.sin(theta)
            addition = spec.scale * template_scale_factors[frame_idx] * improc.ft_shift2(template, dy, dx, output_shape=frame_shape, flux_tol=None)
            result = outcube[frame_idx] + addition  # multiple companions get accumulated in outcube copy
            if saturation_threshold is not None:
                result = np.clip(result, a_min=None, a_max=saturation_threshold)
            outcube[frame_idx] = result
    return outcube


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


def simple_aperture_locations_r_theta(
    r_px, pa_deg, resolution_element_px, exclude_nearest=0, exclude_planet=False
):
    """Aperture centers (x, y) in a ring of radius `r_px` and starting
    at angle `pa_deg` E of N. Unless `exclude_planet` is True,
    the first (x, y) pair gives the planet location (signal aperture).

    Specifying `exclude_nearest` > 0 will skip that many apertures
    from either side of the signal aperture's location"""
    circumference = 2 * r_px * np.pi
    aperture_pixel_diameter = resolution_element_px
    n_apertures = int(circumference / aperture_pixel_diameter)
    start_theta = np.deg2rad(pa_deg + 90)
    delta_theta = np.deg2rad(360 / n_apertures)
    idxs = np.arange(1 + exclude_nearest, n_apertures - exclude_nearest)
    if not exclude_planet:
        idxs = np.concatenate(
            (
                [
                    0,
                ],
                idxs,
            )
        )
    return np.stack((np.repeat(r_px, n_apertures), start_theta + idxs * delta_theta), axis=-1)


@njit(cache=True)
def _simple_aperture_locations(r_px, pa_deg, resolution_element_px, xcenter=0, ycenter=0):
    circumference = 2 * r_px * np.pi
    aperture_pixel_diameter = resolution_element_px
    n_apertures = int(circumference / aperture_pixel_diameter)
    start_theta = np.deg2rad(pa_deg + 90)
    delta_theta = np.deg2rad(360 / n_apertures)
    idxs = np.arange(n_apertures)
    thetas = start_theta + idxs * delta_theta
    offset_x = r_px * np.cos(thetas)
    offset_y = r_px * np.sin(thetas)

    return np.stack((offset_x + xcenter, offset_y + ycenter), axis=-1)


def simple_aperture_locations(r_px, pa_deg, resolution_element_px, exclude_nearest=0,
                              exclude_planet=False, xcenter=0, ycenter=0):
    """Returns (x_center, y_center) for apertures in a ring of
    radius `r_px` starting at angle `pa_deg` E of N. The first (x, y)
    pair gives the planet location (signal aperture)."""
    locs = _simple_aperture_locations(r_px, pa_deg, resolution_element_px, xcenter, ycenter)
    if exclude_nearest != 0:
        locs = np.concatenate([locs[0][np.newaxis, :], locs[1 + exclude_nearest:-exclude_nearest]])
    if exclude_planet:
        locs = locs[1:]
    return locs

def r_pa_to_x_y(r_px, pa_deg, xcenter, ycenter):
   return (
       r_px * np.cos(np.deg2rad(90 + pa_deg)) + xcenter,
       r_px * np.sin(np.deg2rad(90 + pa_deg)) + ycenter
   )

def x_y_to_r_pa(x, y, xcenter, ycenter):
    dx = x - xcenter
    dy = y - ycenter
    pa_deg = np.rad2deg(np.arctan2(dy, dx)) - 90
    r_px = np.sqrt(dx**2 + dy**2)
    if pa_deg < 0:
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
        noise_total += noise
        num_noises += 1
    noise_avg = noise_total / num_noises
    numerator = signal - noise_avg
    stddev_inner_accum = 0
    for i in range(num_noises):
        meansub = (noises[i] - noise_avg)
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
):
    """apply `operation` to the pixels within radius `resolution_element_px`/2 of the centers
    of the simple aperture locations for a planet at `r_px` and `starting_pa_deg`, returning
    the locations and the results as a tuple with the first location and result corresponding
    to the planet aperture"""
    center = (image.shape[0] - 1) / 2, (image.shape[0] - 1) / 2
    yy, xx = improc.cartesian_coords(center, image.shape)
    locations = list(
        simple_aperture_locations(
            r_px,
            starting_pa_deg,
            resolution_element_px,
            exclude_nearest=exclude_nearest,
            exclude_planet=exclude_planet,
        )
    )
    simple_aperture_radius = resolution_element_px / 2
    results = []
    for offset_x, offset_y in locations:
        dist = np.sqrt((xx - offset_x) ** 2 + (yy - offset_y) ** 2)
        mask = dist <= simple_aperture_radius
        results.append(
            operation(image[mask] / np.count_nonzero(mask & np.isfinite(image)))
        )
    return locations, results


def calculate_snr(image, r_px, pa_deg, resolution_element_px, exclude_nearest):
    _, results = reduce_apertures(
        image,
        r_px,
        pa_deg,
        resolution_element_px,
        np.sum,
        exclude_nearest=exclude_nearest,
    )
    return calc_snr_mawet(results[0], results[1:])


@njit(parallel=True, cache=True)
def _calc_snr_image(convolved_image, rho, theta, mask, aperture_diameter_px, exclude_nearest, snr_image_out):
    height, width = convolved_image.shape
    yc, xc = (height - 1) / 2, (width - 1) / 2
    for y in numba.prange(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            loc_rho, loc_theta = rho[y, x], theta[y, x]
            loc_pa_deg = np.rad2deg(loc_theta) - 90
            locs = _simple_aperture_locations(
                loc_rho, loc_pa_deg, aperture_diameter_px, xcenter=xc, ycenter=yc
            )
            n_apertures = locs.shape[0]
            signal_x, signal_y = locs[0]
            signal = convolved_image[round(signal_y), round(signal_x)]
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
                noises[i] = convolved_image[round(noise_y),round(noise_x)]
            calculated_snr = _calc_snr_mawet(signal, noises)
            snr_image_out[y, x] = calculated_snr
    return snr_image_out

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

def calc_snr_image(image, aperture_diameter_px, data_min_r_px, data_max_r_px, exclude_nearest):
    """Compute simple aperture photometry SNR at each pixel and return an image
    with the SNR map"""
    aperture_r = aperture_diameter_px / 2
    iwa_px, owa_px = working_radii_from_aperture_spacing(image.shape, aperture_diameter_px, exclude_nearest, data_min_r_px=data_min_r_px, data_max_r_px=data_max_r_px)
    rho, theta = improc.polar_coords(improc.arr_center(image), image.shape)
    mask = (rho >= iwa_px) & (rho <= owa_px)
    kernel_npix = math.ceil(aperture_diameter_px) + 2  # one pixel border both sides for no reason
    kernel = np.zeros((kernel_npix, kernel_npix))
    kernel_rho, _ = improc.polar_coords(improc.arr_center(kernel), kernel.shape)
    kernel[kernel_rho <= aperture_r] = 1
    image = image.copy()
    image[np.isnan(image)] = 0
    convolved_image = fftconvolve(image, kernel, mode='same')
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
    return [x[1] for x in maxima[::-1]], (iwa_px, owa_px)

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
