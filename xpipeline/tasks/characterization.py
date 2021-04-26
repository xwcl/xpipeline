import numpy as np
import dask
from typing import List
from dataclasses import dataclass

from . import improc
from .. import core
da = core.dask_array

@dataclass
class CompanionSpec:
    scale: float
    r_px: float
    pa_deg: float

    @classmethod
    def from_str(cls, value):
        scale_str, r_px_str, pa_deg_str = value.split(',')
        if scale_str == '?':
            scale_str = '0'
        return cls(scale=float(scale_str), r_px=float(r_px_str), pa_deg=float(pa_deg_str))


@dataclass
class RecoveredSignal(CompanionSpec):
    snr: float

    @classmethod
    def from_spec_snr(cls, spec: CompanionSpec, snr: float):
        return cls(scale=spec.scale, r_px=spec.r_px, pa_deg=spec.pa_deg, snr=snr)


def inject_signals(cube: np.ndarray, angles: np.ndarray, specs: List[CompanionSpec], template: np.ndarray):
    xp = core.get_array_module(cube)
    if xp is da:
        return da.blockwise(
            inject_signals,
            'ijk',
            cube,
            'ijk',
            angles,
            'i',
            specs,
            None,
            template,
            None
        )
    frame_shape = cube.shape[1:]
    outcube = cube.copy()
    for frame_idx in range(cube.shape[0]):
        for spec in specs:
            if spec.scale == 0:
                continue
            theta = np.deg2rad(90 + spec.pa_deg - angles[frame_idx])
            dx, dy = spec.r_px * np.cos(theta), spec.r_px * np.sin(theta)
            addition = spec.scale * improc.shift2(template, dx, dy, output_shape=frame_shape)
            outcube[frame_idx] += addition
    return outcube


def recover_signals(image: np.ndarray, specs: List[CompanionSpec], aperture_diameter_px: float, apertures_to_exclude: int) -> List[RecoveredSignal]:
    signals = []
    for spec in specs:
        _, vals = reduce_apertures(
            image,
            spec.r_px,
            spec.pa_deg,
            aperture_diameter_px,
            np.sum,
            exclude_nearest=apertures_to_exclude
        )
        snr = calc_snr_mawet(vals[0], vals[1:])
        signals.append(RecoveredSignal.from_spec_snr(spec, snr))
    return signals

def simple_aperture_locations_r_theta(r_px, pa_deg, resolution_element_px,
                                      exclude_nearest=0, exclude_planet=False):
    '''Aperture centers (x, y) in a ring of radius `r_px` and starting
    at angle `pa_deg` E of N. Unless `exclude_planet` is True,
    the first (x, y) pair gives the planet location (signal aperture).

    Specifying `exclude_nearest` > 0 will skip that many apertures
    from either side of the signal aperture's location'''
    circumference = 2 * r_px * np.pi
    aperture_pixel_diameter = resolution_element_px
    n_apertures = int(circumference / aperture_pixel_diameter)
    start_theta = np.deg2rad(pa_deg + 90)
    delta_theta = np.deg2rad(360 / n_apertures)
    idxs = np.arange(1 + exclude_nearest, n_apertures - exclude_nearest)
    if not exclude_planet:
        idxs = np.concatenate(([0, ], idxs))
    return np.repeat(r_px, n_apertures), start_theta + idxs * delta_theta


def simple_aperture_locations(r_px, pa_deg, resolution_element_px,
                                  exclude_nearest=0, exclude_planet=False):
    '''Aperture centers (x, y) in a ring of radius `r_px` and starting
    at angle `pa_deg` E of N. Unless `exclude_planet` is True,
    the first (x, y) pair gives the planet location (signal aperture).

    Specifying `exclude_nearest` > 0 will skip that many apertures
    from either side of the signal aperture's location'''
    _, thetas = simple_aperture_locations_r_theta(
        r_px, 
        pa_deg, 
        resolution_element_px, 
        exclude_nearest=exclude_nearest, 
        exclude_planet=exclude_planet
    )
    offset_x = r_px * np.cos(thetas)
    offset_y = r_px * np.sin(thetas)
    return np.stack((offset_x, offset_y), axis=-1)


def calc_snr_mawet(signal, noises):
    '''Calculate signal to noise following the
    two-sample t test as defined in Mawet 2014'''
    return (
        signal - np.average(noises)
    ) / (
        np.std(noises) * np.sqrt(1 + 1 / len(noises))
    )


def cartesian_coords(center, data_shape):
    '''center in x,y order; returns coord arrays xx, yy of data_shape'''
    yy, xx = np.indices(data_shape, dtype=float)
    center_x, center_y = center
    yy -= center_y
    xx -= center_x
    return xx, yy


def reduce_apertures(image, r_px, starting_pa_deg, resolution_element_px, operation,
                     exclude_nearest=0, exclude_planet=False):
    '''apply `operation` to the pixels within radius `resolution_element_px`/2 of the centers
    of the simple aperture locations for a planet at `r_px` and `starting_pa_deg`, returning
    the locations and the results as a tuple with the first location and result corresponding
    to the planet aperture'''
    center = (image.shape[0] - 1) / 2, (image.shape[0] - 1) / 2
    xx, yy = cartesian_coords(center, image.shape)
    locations = list(simple_aperture_locations(
        r_px, starting_pa_deg, resolution_element_px, 
        exclude_nearest=exclude_nearest, exclude_planet=exclude_planet
    ))
    simple_aperture_radius = resolution_element_px / 2
    results = []
    for offset_x, offset_y in locations:
        dist = np.sqrt((xx - offset_x)**2 + (yy - offset_y)**2)
        mask = dist <= simple_aperture_radius
        results.append(operation(image[mask] / np.count_nonzero(mask & np.isfinite(image))))
    return locations, results

def calculate_snr(image, r_px, pa_deg, resolution_element_px, exclude_nearest):
    locations, results = reduce_apertures(
        image,
        r_px,
        pa_deg,
        resolution_element_px,
        np.sum,
        exclude_nearest=exclude_nearest
    )
    return calc_snr_mawet(results[0], results[1:])
