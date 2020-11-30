import io
import copy
import glob
import datetime
import logging
import os
import gc
import os.path
import itertools
import warnings
from functools import partial
from collections import OrderedDict

import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn import decomposition
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import imsave

import subprocess
import shlex
import time
import joblib
import multiprocessing
DEFAULT_N_JOBS = multiprocessing.cpu_count()

import requests

from astropy.modeling.models import Gaussian2D
from astropy.modeling import fitting
from astropy.nddata.utils import Cutout2D
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel
from astropy import units as u
from astropy import wcs
# from photutils import centroid_com, centroid_1dg, centroid_2dg

from skimage.transform import AffineTransform, warp, rotate, rescale
from scipy.ndimage.interpolation import zoom
from scipy.optimize import minimize, fmin
from scipy import interpolate
from skimage.feature import register_translation
from scipy.interpolate import CloughTocher2DInterpolator

import poppy

import doodads as dd

# from doodads import supply_argument, add_colorbar
from functools import wraps

def supply_argument(**override_kwargs):
    '''
    Decorator to supply a keyword argument using a callable if
    it is not provided.
    '''

    def decorator(f):
        @wraps(f)
        def inner(*args, **kwargs):
            for kwarg in override_kwargs:
                if kwarg not in kwargs:
                    kwargs[kwarg] = override_kwargs[kwarg]()
            return f(*args, **kwargs)
        return inner
    return decorator

from mpl_toolkits.axes_grid1 import make_axes_locatable

def add_colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

log = logging.getLogger(__name__)

debug, info, warn, error, critical = (
    log.debug,
    log.info,
    log.warn,
    log.error,
    log.critical,
)

# The land of import-time side effects...
# Comes with the territory, I suppose.
SCRATCH_DIR = os.environ.get('PIPELINE_CACHE', os.path.abspath(os.path.join('.', 'scratch')))
scratch = joblib.Memory(cachedir=SCRATCH_DIR)

# From LCO telescope information page
MAGELLAN_PRIMARY_MIRROR_DIAMETER = 6502.4 * u.mm
MAGELLAN_PRIMARY_STOP_DIAMETER = 6478.4 * u.mm
MAGELLAN_SECONDARY_AREA_FRACTION = 0.074
# computed from the above
MAGELLAN_SECONDARY_DIAMETER = 2 * np.sqrt(((MAGELLAN_PRIMARY_STOP_DIAMETER / 2)**2) * MAGELLAN_SECONDARY_AREA_FRACTION)
# from MagAO-X Pupil Definition doc
MAGELLAN_SPIDERS_OFFSET = 0.34 * u.m

# Morzinski 2015 ApJ Table 18
CLIO2_PIXEL_SCALE = 15.846e-3 * (u.arcsec / u.pixel)
# HAWAII-H1RG (http://www.teledyne-si.com/products/Documents/H1RG%20Brochure%20-%20September%202017.pdf)
CLIO2_PIXEL_PITCH = 18 * u.um
# From Otten+ 2017
VAPP_PSF_ROTATION_DEG = -26
VAPP_PSF_OFFSET_LAMBDA_OVER_D = 35 / 2
VAPP_IWA_LAMD, VAPP_OWA_LAMD = 2, 7
VAPP_LEAKAGE_RATIO = 0.00636
# offset of inner edge of dark hole from center of PSF, 
# chosen such that a median image of Sirius in 3.9 um
# has zero saturated pixels in the dark hole
# (n.b. some bright structure admitted, but no saturated/useless pix)
VAPP_OFFSET_LAMD = 1.44
# radius in lambda/D within which any glint disqualifies a frame
# chosen to circumscribe dark hole region and barely touch first bright feature
# on the dark hole edge
VAPP_GLINT_FREE_RADIUS_LAMD = 10

def make_both_masks(wavelength_um, shape, owa_lamd=VAPP_OWA_LAMD, offset_lamd=VAPP_OFFSET_LAMD,
                    psf_rotation_deg=VAPP_PSF_ROTATION_DEG):
    radius_px = clio2_vapp_lambda_over_d_to_pixel(
        owa_lamd + offset_lamd,
        wavelength_um * u.um
    ).value
    offset_r = clio2_vapp_lambda_over_d_to_pixel(
        offset_lamd,
        wavelength_um * u.um
    ).value

    bot_overall_rotation_radians = 3 * np.pi/2 + np.deg2rad(psf_rotation_deg)
    bot_offset_theta = np.deg2rad(-psf_rotation_deg)

    top_overall_rotation_radians = np.pi/2 + np.deg2rad(psf_rotation_deg)
    top_offset_theta = np.deg2rad(180 - psf_rotation_deg)
    ctr_x, ctr_y = (shape[1] - 1) / 2, (shape[0] - 1) / 2
    bot_mask = dd.mask_arc(
        shape,
        (ctr_x + offset_r * np.cos(bot_offset_theta), ctr_y + offset_r * np.sin(bot_offset_theta)),
        from_radius=0,
        to_radius=radius_px,
        from_radians=0,
        to_radians=np.pi,
        overall_rotation_radians=bot_overall_rotation_radians,
    )
    top_mask = dd.mask_arc(
        shape,
        (ctr_x + offset_r * np.cos(top_offset_theta), ctr_y + offset_r * np.sin(top_offset_theta)),
        from_radius=0,
        to_radius=radius_px,
        from_radians=0,
        to_radians=np.pi,
        overall_rotation_radians=top_overall_rotation_radians,
    )
    return bot_mask, top_mask

def clio2_vapp_lambda_over_d_to_arcsec(lambda_over_d, wavelength, d=MAGELLAN_PRIMARY_MIRROR_DIAMETER):
    unit_lambda_over_d  = (wavelength.to(u.m) / d.to(u.m)).si.value * u.radian
    return (lambda_over_d  * unit_lambda_over_d).to(u.arcsec)
def clio2_vapp_arcsec_to_lambda_over_d(arcsec, wavelength, d=MAGELLAN_PRIMARY_MIRROR_DIAMETER):
    unit_lambda_over_d = ((wavelength.to(u.m) / d.to(u.m)).si.value * u.radian).to(u.arcsec)
    lambda_over_d = (arcsec / unit_lambda_over_d).si
    return lambda_over_d

def clio2_vapp_lambda_over_d_to_pixel(lambda_over_d, wavelength, d=MAGELLAN_PRIMARY_MIRROR_DIAMETER):
    arcsec = clio2_vapp_lambda_over_d_to_arcsec(lambda_over_d, wavelength, d=d)
    return (arcsec / CLIO2_PIXEL_SCALE).to(u.pixel)
def clio2_vapp_pixel_to_lambda_over_d(px, wavelength, d=MAGELLAN_PRIMARY_MIRROR_DIAMETER):
    arcsec = (px * CLIO2_PIXEL_SCALE).to(u.arcsec)
    return clio2_vapp_arcsec_to_lambda_over_d(arcsec, wavelength, d=d)

CLIO2_DETECTOR_SHAPE = (512, 1024)

PSF_FINDING_SMOOTH_KERNEL_WIDTH_PX = 9
LEAK_BOX_SIZE = 75
PSF_BOX_SIZE = 227
PSF_DEAD_CENTER = (PSF_BOX_SIZE - 1) / 2, (PSF_BOX_SIZE - 1) / 2
GHOST_SEARCH_BOX_SIZE = 350, 150
GHOST_BOX_SIZE = 110
INITIAL_GUESS_LEAK_LEFT_CENTER = 250, 287
INITIAL_GUESS_LEAK_RIGHT_CENTER = 630, 311
GLINT_BOX_SIZE = 90
INITIAL_GUESS_GLINT_CENTER = 130, 95
INITIAL_GUESS_GHOST_CENTER = 800, 350
GLIMMER_BOX_SIZE = 40
INITIAL_GUESS_GLIMMER_CENTER = 185, 355
PSF_OFFSET = -60, 130
INITIAL_GUESS_TOP_LEFT_CENTER = INITIAL_GUESS_LEAK_LEFT_CENTER[0] + PSF_OFFSET[0], INITIAL_GUESS_LEAK_LEFT_CENTER[1] + PSF_OFFSET[1]
INITIAL_GUESS_BOTTOM_LEFT_CENTER = INITIAL_GUESS_LEAK_LEFT_CENTER[0] - PSF_OFFSET[0], INITIAL_GUESS_LEAK_LEFT_CENTER[1] - PSF_OFFSET[1]
INITIAL_GUESS_TOP_RIGHT_CENTER = INITIAL_GUESS_LEAK_RIGHT_CENTER[0] + PSF_OFFSET[0], INITIAL_GUESS_LEAK_RIGHT_CENTER[1] + PSF_OFFSET[1]
INITIAL_GUESS_BOTTOM_RIGHT_CENTER = INITIAL_GUESS_LEAK_RIGHT_CENTER[0] - PSF_OFFSET[0], INITIAL_GUESS_LEAK_RIGHT_CENTER[1] - PSF_OFFSET[1]

DQ_BAD_PIXEL         = 0b00000001
DQ_SATURATED         = 0b00000010
DQ_NOT_OVERLAPPING   = 0b00000100
DQ_BAD_INTERPOLATION = 0b00001000


# Min weight of a good pixel that's been regridded
MIN_PIXEL_WEIGHT = 0.8

# Width of convolution kernel to simulate blurring in saturated exposures
DETECTOR_SATURATION_BLUR_PX = 1.5

# @scratch.cache
def collect_data(data_pattern, sky_pattern):
    data_files = glob.glob(data_pattern)
    sky_files = glob.glob(sky_pattern)
    return data_files, sky_files

OBS_UNKNOWN = -1
OBS_SCIENCE = 1
OBS_SKY = 2

def rebin(a, bin_factor):
    '''Combine a `bin_factor` by `bin_factor` square of pixels into a single pixel by summing'''
    intermediate_shape = a.shape[0] // bin_factor, bin_factor, a.shape[1] // bin_factor, bin_factor
    return a.reshape(intermediate_shape).sum(-1).sum(1)

def yoink(observation_date_key, file_path, extra_data=None):
    result = OrderedDict()
    result['filename'] = file_path
    if extra_data is not None:
        for key in extra_data:
            result[key] = extra_data[key]

    with open(file_path, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdulist = fits.open(f)
            hdulist.verify('fix')
        for hdu in hdulist:
            for key in hdu.header:
                if key in ('EXTEND', 'COMMENT', 'BITPIX', 'EXTEND', 'EXTNAME', 'SIMPLE', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'XTENSION'):
                    continue
                result[key] = hdu.header.get(key)
        # result['data_min'] = np.min(hdulist[0].data)
        # result['data_max'] = np.max(hdulist[0].data)
        # result['data_mean'] = np.average(hdulist[0].data)
        # result['data_percentile_5'] = np.percentile(hdulist[0].data, 5)
        # result['data_percentile_95'] = np.percentile(hdulist[0].data, 95)
    if observation_date_key in result:
        result[observation_date_key] = np.datetime64(result[observation_date_key])
    else:
        result[observation_date_key] = None
    return result



# @scratch.cache
def construct_observations_table(data_files, sky_files, observation_date_key='DATE-OBS', n_jobs=DEFAULT_N_JOBS):
    # yoink_science = partial(yoink, observation_date_key, extra_data={'type': OBS_SCIENCE})
    # yoink_sky = partial(yoink, observation_date_key, extra_data={'type': OBS_SKY})
    metadata_generator = itertools.chain(
        (joblib.delayed(yoink)(observation_date_key, fn, extra_data={'type': OBS_SCIENCE})
         for fn in data_files),
        (joblib.delayed(yoink)(observation_date_key, fn, extra_data={'type': OBS_SKY})
         for fn in sky_files),
    )
    results = joblib.Parallel(n_jobs=n_jobs, backend='multiprocessing')(metadata_generator)
    all_columns = ['filename', 'type', observation_date_key]
    for r in results:
        for key in r:
            if key not in all_columns:
                all_columns.append(key)

    # construct table
    df = pd.DataFrame(
        index=np.arange(0, len(data_files) + len(sky_files)),
        columns=all_columns
    )
    for idx, frame_data in enumerate(results):
        df.loc[idx] = [frame_data.get(col) for col in all_columns]
    df.sort_values(observation_date_key, inplace=True)
    return df.reset_index(drop=True)  # idx <=> date

def extend_observations_table(obs_table, func, n_jobs=DEFAULT_N_JOBS, overwrite=False):
    new_table = obs_table.copy()
    if n_jobs == 1:
        results = map(lambda idx_row_tuple: func(idx_row_tuple[1]), obs_table.iterrows())
    else:
        results = joblib.Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            joblib.delayed(func)(row) for _, row in obs_table.iterrows()
        )
    added_columns = set()
    for idx, new_data in zip(obs_table.index, results):
        for col in new_data:
            if col not in new_table.columns:
                new_table[col] = np.nan
                added_columns.add(col)
            else:
                if col not in added_columns:
                    if not overwrite:
                        # column already existed
                        raise RuntimeError("Column {} exists in the provided table already. "
                                           "Pass overwrite=True to replace all of its values.")
                    else:
                        new_table[col] = np.nan
            new_table.loc[idx, col] = new_data.get(col)

#     all_columns = []
#     for new_data in results:
#         for key in new_data:
#             if key not in all_columns:
#                 all_columns.append(key)
#     for col in all_columns:
#         new_table[col] = np.nan
#     for idx, new_data in zip(obs_table.index, results):
#         for k in all_columns:
#             new_table.loc[idx, k] = new_data.get(k)
    return new_table

def centroid_around(data, position, size, display=False, ax=None):
    if display and ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    try:
        width, height = size
    except TypeError:
        width = height = size
    cutout = Cutout2D(data, position, (height, width))
    # cutout_centroid = centroid_com(cutout.data)
    y, x = np.unravel_index(np.argmax(cutout.data), cutout.data.shape)
    final_height, final_width = cutout.data.shape
    if y in (0, final_height - 1) or x in (0, final_width - 1):
        railed = True  # the glint wasn't found
        # (or was on the edge of the cutout, so the centroid is unreliable)
    else:
        railed = False
    centroid = cutout.to_original_position((x, y))
    if display:
        ax.imshow(cutout.data)
        ax.axhline(y=y)
        ax.axvline(x=x)
    return centroid, railed

def locate_psf_centers_for_row(table_row,
                       badpix,
                       sky_reference,
                       smooth_px=PSF_FINDING_SMOOTH_KERNEL_WIDTH_PX,
                       display=False):
    out = {}
    if table_row.type == OBS_SKY:
        return {}
    else:
        with open(table_row['filename'], 'rb') as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hdulist = fits.open(f)
                hdulist.verify('fix')
                data = hdulist[0].data
        return locate_rough_psf_centers(
            data,
            badpix,
            sky_reference,
            smooth_px=smooth_px,
            display=display,
        )

PSF_LOCATIONS_WHAT_WHERE = [
    ('top_left', INITIAL_GUESS_TOP_LEFT_CENTER, PSF_BOX_SIZE),
    ('bottom_left', INITIAL_GUESS_BOTTOM_LEFT_CENTER, PSF_BOX_SIZE),
    ('leak_left',  INITIAL_GUESS_LEAK_LEFT_CENTER, LEAK_BOX_SIZE),
    ('top_right', INITIAL_GUESS_TOP_RIGHT_CENTER, PSF_BOX_SIZE),
    ('bottom_right', INITIAL_GUESS_BOTTOM_RIGHT_CENTER, PSF_BOX_SIZE),
    ('leak_right',  INITIAL_GUESS_LEAK_RIGHT_CENTER, LEAK_BOX_SIZE),
    ('ghost', INITIAL_GUESS_GHOST_CENTER, GHOST_SEARCH_BOX_SIZE),
    ('glint', INITIAL_GUESS_GLINT_CENTER, GLINT_BOX_SIZE),
    ('glimmer', INITIAL_GUESS_GLIMMER_CENTER, GLIMMER_BOX_SIZE),
]

def locate_rough_psf_centers(
        data,
        badpix=None,
        sky_reference=None,
        smooth_px=PSF_FINDING_SMOOTH_KERNEL_WIDTH_PX,
        display=False,
        ax=None,
    ):
    out = {}
    raw_data = data

    if badpix is not None:
        data = mask_bad_pixels(data, badpix, fill_value=0.0)
    if sky_reference is not None:
        data -= sky_reference / np.average(sky_reference) * np.nanmean(data)
    # thresholded = data.copy()
    # thresholded[data < np.nanmedian(data)] = 0
    smoothed = smooth(data, smooth_px)
    # high_pass_data = data - smoothed
    if display:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=len(PSF_LOCATIONS_WHAT_WHERE) + 1, ncols=1, figsize=(8, 4 * len(PSF_LOCATIONS_WHAT_WHERE)))
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(smoothed, vmin=np.nanpercentile(smoothed, 5), vmax=np.nanpercentile(smoothed, 95))
    for idx, (name, position, size) in enumerate(PSF_LOCATIONS_WHAT_WHERE, start=1):
        (x, y), railed = centroid_around(
            smoothed,
            position,
            size,
            display=display,
            ax=axes[idx] if display else None
        )
        out[name+'_psf_x'], out[name+'_psf_y'] = x, y
        out[name+'_raw_peak'] = raw_data[y, x]
        if not railed and out[name+'_raw_peak'] > 0:
            out[name + '_psf_found'] = True
        else:
            out[name + '_psf_found'] = False
        if display:
            axes[idx].set_title('{} {}'.format(name, position))
            ax.scatter(
                x,
                y,
                marker='o' if out[name + '_psf_found'] else '*', s=100, label=name,
                ec='w',
            )
    if display:
        ax.legend(loc=(1.1, 0.1))
    # if the top and bottom PSFs aren't separated by the appropriate amount,
    # the detection is probably bogus:
    # def _check_sep(top_x, top_y, bottom_x, bottom_y):
    #     if (top_x - bottom_x) - ()
    # for side in ('left', 'right'):
    #     if out[f'{side}_top_psf_found'] and out[f'{side}_bottom_psf_found']:
            
    for side in ('left', 'right'):
        # Note that the leakage term is not always visible above the noise
        found = all(out['{}_{}_psf_found'.format(name, side)] for name in ('top', 'bottom'))
        if found:
            if 'top_psf_x' in out:
                raise RuntimeError("Unable to decide if this is a left or right nodded image (found both)")
            out['side_found'] = side
            for name in ('top', 'leak', 'bottom'):
                original_name = name + '_' + side
                for key in ('{}_psf_x', '{}_psf_y', '{}_raw_peak', '{}_psf_found'):
                    out[key.format(name)] = out[key.format(original_name)]
            # guess locations of any PSFs we didn't find
            # top
            if not out['top_psf_found']:
                if out['leak_psf_found']:
                    out['top_psf_x'], out['top_psf_y'] = out['leak_psf_x'] + PSF_OFFSET[0], out['leak_psf_y'] + PSF_OFFSET[1]
                elif out['bottom_psf_found']:
                    out['top_psf_x'], out['top_psf_y'] = out['leak_psf_x'] + 2 * PSF_OFFSET[0], out['leak_psf_y'] + 2 * PSF_OFFSET[1]
                else:
                    if side == 'left':
                        out['top_psf_x'], out['top_psf_y'] = INITIAL_GUESS_TOP_LEFT_CENTER
                    elif side == 'right':
                        out['top_psf_x'], out['top_psf_y'] = INITIAL_GUESS_TOP_RIGHT_CENTER
                    else:
                        raise RuntimeError("Unknown side")
            # leak
            if not out['leak_psf_found']:
                if out['top_psf_found']:
                    out['leak_psf_x'], out['leak_psf_y'] = out['top_psf_x'] - PSF_OFFSET[0], out['top_psf_y'] - PSF_OFFSET[1]
                elif out['bottom_psf_found']:
                    out['leak_psf_x'], out['leak_psf_y'] = out['bottom_psf_x'] + PSF_OFFSET[0], out['leak_psf_y'] + PSF_OFFSET[1]
                else:
                    if side == 'left':
                        out['leak_psf_x'], out['leak_psf_y'] = INITIAL_GUESS_LEAK_LEFT_CENTER
                    elif side == 'right':
                        out['leak_psf_x'], out['leak_psf_y'] = INITIAL_GUESS_LEAK_RIGHT_CENTER
                    else:
                        raise RuntimeError("Unknown side")
            # bottom
            if not out['bottom_psf_found']:
                if out['leak_psf_found']:
                    out['bottom_psf_x'], out['bottom_psf_y'] = out['leak_psf_x'] - PSF_OFFSET[0], out['leak_psf_y'] - PSF_OFFSET[1]
                elif out['top_psf_found']:
                    out['bottom_psf_x'], out['bottom_psf_y'] = out['leak_psf_x'] - 2 * PSF_OFFSET[0], out['leak_psf_y'] - 2 * PSF_OFFSET[1]
                else:
                    if side == 'left':
                        out['bottom_psf_x'], out['bottom_psf_y'] = INITIAL_GUESS_BOTTOM_LEFT_CENTER
                    elif side == 'right':
                        out['bottom_psf_x'], out['bottom_psf_y'] = INITIAL_GUESS_BOTTOM_RIGHT_CENTER
                    else:
                        raise RuntimeError("Unknown side")

    if not 'side_found' in out:
        out['side_found'] = 'neither'
    return out

def check_clio_glint_centers(hdr, wavelength_um):
    '''Ensure Clio's many stray light sources aren't in the way'''
    exclusion_radius_px = clio2_vapp_lambda_over_d_to_pixel(
        VAPP_GLINT_FREE_RADIUS_LAMD, wavelength_um * u.um
    ).value
    
    # If we can't find the glint, it's probably hidden in the
    # bright part of the PSF. So, we ensure it's found ('GLINTOK')
    # and that its location is far from the dark hole
    dist_to_glint = np.sqrt((hdr['BOTXC'] - hdr['GLINTXC'])**2 + (hdr['BOTYC'] - hdr['GLINTYC'])**2)
    glint_found_outside = hdr['GLINTOK'] and dist_to_glint > exclusion_radius_px
    # if glimmer is found, ensure it's outside the dark hole. If not found, don't worry.
    dist_to_glimmer = np.sqrt((hdr['TOPXC'] - hdr['GLIMXC'])**2 + (hdr['TOPYC'] - hdr['GLIMYC'])**2)
    glimmer_not_contaminating = (not hdr['GLIMOK']) or (dist_to_glimmer > exclusion_radius_px)
    frame_ok = glint_found_outside and glimmer_not_contaminating
    return frame_ok

def find_center_shift(data, model):
    if np.any(~np.isfinite(data)):
        yy, xx = np.indices(data.shape)
        data, _ = regrid_image(data, xx, yy)
    else:
        data = data.copy()
    # nans confuse register_translation:
    data[~np.isfinite(data)] = np.nanmin(data)

    (shift_y, shift_x), _, _ = register_translation(model, data, upsample_factor=100)
    rough_x_center, rough_y_center = data.shape[1] / 2, data.shape[0] / 2
    new_x_center, new_y_center = rough_x_center - shift_x, rough_y_center - shift_y
    info("Refined centroid from {} to {} (by shifting {})".format(
        (rough_x_center, rough_y_center),
        (new_x_center, new_y_center),
        (-shift_x, -shift_y)
    ))
    return shift_x, shift_y

def make_fake_saturated_percentile(unsaturated_psf, percentile, smooth_px=1.5, fill_value=1):
    psf = unsaturated_psf.copy()
    psf /= np.nanpercentile(psf, percentile)
    psf = smooth(psf, smooth_px)
    psf[psf > 1] = fill_value
    return psf

def cutout_and_interpolate(data, center, npix, mask=None):
    '''Cutout `npix` by `npix` of `data` centered at `center`
    which need not be an integer pixel. 
    '''
    # pixel has value y from x - 0.5 to x + 0.5
    # | 0 | 1 | 2 |   total: 3
    #       ^ 
    # center is 1
    new_center = (npix - 1) / 2
    
    # | 0 | 1 | 2 |
    indices = np.arange(npix, dtype=float)
    
    # now we want these to be offsets from center
    # | -1 | 0 | 1 |
    indices -= new_center
    
    # now we make the coordinates in the *original* indices
    # for x and y
    #
    # suppose center[0] = 1.2
    # | 0.2 | 1.2 | 2.2 |
    # so the original center is present at index 1
    xs = center[0] + indices
    ys = center[1] + indices
    xx, yy = np.meshgrid(xs, ys)
    # the resulting image will be interpolated from `data`
    # at the coords in xx and yy
    print('before regrid')
    new_image, _ = regrid_image(data, xx, yy, mask=mask, method='linear')
    print('done')
    return new_image

def make_centered_grid(center, npix):
    '''Cutout `npix` by `npix` of `data` centered at `center`
    which need not be an integer pixel. 
    '''
    # pixel has value y from x - 0.5 to x + 0.5
    # | 0 | 1 | 2 |   total: 3
    #       ^ 
    # center is 1
    new_center = (npix - 1) / 2
    
    # | 0 | 1 | 2 |
    indices = np.arange(npix, dtype=float)
    
    # now we want these to be offsets from center
    # | -1 | 0 | 1 |
    indices -= new_center
    
    # now we make the coordinates in the *original* indices
    # for x and y
    #
    # suppose center[0] = 1.2
    # | 0.2 | 1.2 | 2.2 |
    # so the original center is present at index 1
    xs = center[0] + indices
    ys = center[1] + indices
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy

def refine_psf_centers(data, models, rough_centers, display=False):
    '''Refine rough centroids using FT cross-correlation

    Parameters
    ----------
    data : array-like
    models : dict
        Dictionary with keys 'top', 'leak', 'bottom' and values
        of PSF model arrays of PSF_BOX_SIZE and LEAK_BOX_SIZE respectively
    rough_centers : dict
        keys in the convention of locate_rough_psf_centers output
    display : bool
        Plot some plots
    '''
    centers = rough_centers.copy()
    yy, xx = np.indices(data.shape)
    isfinite_mask = np.isfinite(data)
    points = np.stack((xx[isfinite_mask].flat, yy[isfinite_mask].flat), axis=-1)
    values = data[isfinite_mask].flatten()
    interpolator = CloughTocher2DInterpolator(
        points,
        values,
        fill_value=0
    )
    def cutout_at(center, npix):
        cutout_xx, cutout_yy = make_centered_grid(center, npix)
        coords = np.stack((cutout_xx.flat, cutout_yy.flat), axis=-1)
        return interpolator(coords).reshape(npix, npix)
    # if dq is not None:
    #     only_saturated = (dq ^ DQ_SATURATED) == 0
    #     data = data.copy()
    #     data[only_saturated] = MORZINSKI_CORRECTED_MAX
    #     dq = dq.copy()
    #     dq = dq & ~DQ_SATURATED

    # interpolated_image, _ = regrid_image(data, xx, yy)
    # ('leak', LEAK_BOX_SIZE),
    for location, size in (('top', PSF_BOX_SIZE), ('bottom', PSF_BOX_SIZE)):
        print(location)
        box_center = (size - 1) / 2
        xc, yc = rough_centers['{}_psf_x'.format(location)], rough_centers['{}_psf_y'.format(location)]
        # cutout = Cutout2D(
        #     interpolated_image,
        #     (xc, yc),
        #     (size, size),
        #     mode='partial'
        # )
        # nans from partial overlap confuse register_translation:
        # cutout.data[~np.isfinite(cutout.data)] = np.nanmin(cutout.data)
        cutout_data = cutout_at((xc, yc), size)

        # cutout_data[~np.isfinite(cutout_data)] = np.nanmin(cutout_data)
        normalized_cutout = cutout_data / np.max(cutout_data)
        if location == 'top':
            normalized_cutout = np.fliplr(np.flipud(normalized_cutout))
            normalized_model = models['bottom'] / np.max(models['bottom'])
        else:
            normalized_model = models[location] / np.max(models[location])
        (shift_y, shift_x), _, _ = dd.find_shift(normalized_model, normalized_cutout)
        if display:
            import matplotlib.pyplot as plt
            fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,6))
            
            ax1.imshow(normalized_cutout, vmin=0, vmax=1)
            ax1.set_xlim(box_center - 30, box_center + 30)
            ax1.set_ylim(box_center - 30, box_center + 30)
            ax2.imshow(normalized_model, vmin=0, vmax=1)
            ax2.set_xlim(box_center - 30, box_center + 30)
            ax2.set_ylim(box_center - 30, box_center + 30)
            ax3.imshow(normalized_model - normalized_cutout, vmin=-1, vmax=1, cmap='RdBu_r')
            ax3.set_xlim(box_center - 30, box_center + 30)
            ax3.set_ylim(box_center - 30, box_center + 30)
        if location == 'top':
            # since we have flipped top and xcorred with bottom,
            # shift_x and shift_y are negative in the un-flipped
            # frame
            new_xc, new_yc = xc + shift_x, yc + shift_y
        else:
            new_xc, new_yc = xc - shift_x, yc - shift_y
        # TODO remove
        # print('double check')
        # cutout_data = cutout_at((new_xc, new_yc), size)
        # normalized_cutout = cutout_data / np.max(cutout_data)
        # if location == 'top':
        #     normalized_cutout = np.fliplr(np.flipud(normalized_cutout))
        #     normalized_model = models['bottom'] / np.max(models['bottom'])
        # else:
        #     normalized_model = models[location] / np.max(models[location])
        # (shift_y, shift_x), _, _ = register_translation(normalized_model, normalized_cutout, upsample_factor=100)
        # info(f"Second shift: {shift_x}, {shift_y}")
        # assert shift_y < 0.05, f'shift_y = {shift_y}'
        # assert shift_x < 0.05, f'shift_x = {shift_x}'
        # if display:
        #     import matplotlib.pyplot as plt
        #     fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,6))
            
        #     ax1.imshow(normalized_cutout, vmin=0, vmax=1)
        #     ax1.set_xlim(box_center - 30, box_center + 30)
        #     ax1.set_ylim(box_center - 30, box_center + 30)
        #     ax2.imshow(normalized_model, vmin=0, vmax=1)
        #     ax2.set_xlim(box_center - 30, box_center + 30)
        #     ax2.set_ylim(box_center - 30, box_center + 30)
        #     ax3.imshow(normalized_model - normalized_cutout, vmin=-1, vmax=1, cmap='RdBu_r')
        #     ax3.set_xlim(box_center - 30, box_center + 30)
        #     ax3.set_ylim(box_center - 30, box_center + 30)
        
        centers['{}_psf_x'.format(location)], centers['{}_psf_y'.format(location)] = new_xc, new_yc
        centers[location+'_raw_peak'] = data[int(new_yc), int(new_xc)]
        info("Refined {} from {} to {} (by shifting {})".format(
            location,
            (xc, yc),
            (new_xc, new_yc),
            (-shift_x, -shift_y)
        ))
    # refine leak
    top_xc, top_yc = centers['top_psf_x'], centers['top_psf_y']
    bot_xc, bot_yc = centers['bottom_psf_x'], centers['bottom_psf_y']
    centers['leak_psf_x'], centers['leak_psf_y'] = (top_xc + bot_xc) / 2, (top_yc + bot_yc) / 2
    return centers

def rotate_about_center(image, rotation):
    xc, yc = image.shape[1] / 2, image.shape[0] / 2
    xx, yy = make_grid(
        image.shape,
        rotation=rotation,
        rotation_x_center=xc, rotation_y_center=yc,
        scale_x=1, scale_y=1,
        scale_x_center=xc, scale_y_center=yc,
        x_shift=0, y_shift=0,
    )
    new_image, _ = regrid_image(image, xx, yy)
    return new_image

def make_corrected_frame(table_row, badpix, outdir):
    out = {}
    with open(table_row['filename'], 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdulist = fits.open(f)
            hdulist.verify('fix')
        data = hdulist[0].data
        corrected_data = mask_bad_pixels(correct_linearity(data), badpix)
        new_hdulist = fits.HDUList(hdulist.copy())
        new_hdulist[0].data = corrected_data
        filename = os.path.basename(table_row['filename'])
        outpath = os.path.join(outdir, filename)
        new_hdulist.writeto(outpath, output_verify='silentfix', overwrite=True)
        out['instrument_corrected_filename'] = outpath
    return out

# @scratch.cache
def make_instrument_corrected_frames(obs_table, badpix):
    outdir = '../scratch/instrument_corrected/'
    os.makedirs(outdir, exist_ok=True)
    func = partial(make_corrected_frame, badpix=badpix, outdir=outdir)
    return extend_observations_table(
        obs_table,
        func
    )

# @scratch.cache
def centroid_science_frames(obs_table, badpix, sky_reference):
    return extend_observations_table(
        obs_table,
        partial(locate_psf_centers_for_row, badpix=badpix, sky_reference=sky_reference),
    )

# Morzinski+ 2015 ApJ -- Appendix B
MORZINSKI_COEFFICIENTS = (112.575, 1.00273, -1.40776e-6, 4.59015e-11)
MORZINSKI_DOMAIN = MORZINSKI_LINEARITY_MIN, MORZINSKI_LINEARITY_MAX = [2.7e4, 4.5e4]
MORZINSKI_RANGE = MORZINSKI_CORRECTED_MIN, MORZINSKI_CORRECTED_MAX = [
    sum([coeff * x**idx for idx, coeff in enumerate(MORZINSKI_COEFFICIENTS)])
    for x in MORZINSKI_DOMAIN
]

def correct_linearity(data, coeffs=MORZINSKI_COEFFICIENTS, correctable_domain=MORZINSKI_DOMAIN,
                      verbose=False):
    corrected_domain = [
        sum([coeff * x**idx for idx, coeff in enumerate(MORZINSKI_COEFFICIENTS)])
        for x in MORZINSKI_DOMAIN
    ]
    if verbose:
        debug('Mapping correctable domain {} to corrected domain {}'.format(correctable_domain, corrected_domain))
    linearity_correction = np.polynomial.polynomial.Polynomial(
        coef=coeffs,
        domain=correctable_domain,
        window=corrected_domain
    )
    result = data.copy().astype(float)
    nonlinear_pixels = data >= correctable_domain[0]
    debug("Found {} pixels outside the correctable domain of {}".format(np.count_nonzero(nonlinear_pixels), correctable_domain))
    in_range = nonlinear_pixels & (data <= correctable_domain[1])
    result[in_range] = linearity_correction(data[in_range])
    dq = np.zeros(result.shape, dtype=int)
    saturated = data > correctable_domain[1]
    dq[saturated] = DQ_SATURATED
    if verbose:
        debug("Corrected {}/{} pixels".format(np.count_nonzero(in_range), np.count_nonzero(nonlinear_pixels)))
        debug("Found {} saturated".format(np.count_nonzero(saturated)))
    return result, dq

# @scratch.cache
def retrieve_binary(url, verify=True):
    resp = requests.get(url, verify=verify)
    return io.BytesIO(resp.content)

def fits_from_url(url, verify=True):
    data = retrieve_binary(url, verify=verify)
    return fits.open(data)

def interpolate_bad_pixels(exposure, badpix, fill_value=np.nan):
    """Accepts exposure array and bad pixel map (where
    bad pixels have != 0 values), averages NSEW neighbor
    pixels (omitting up to 3 of these if they too are bad),
    and flagging with `fill_value` those it can't interpolate"""
    interp_exposure = exposure.copy().astype(float)
    ys, xs = np.where(badpix != 0)
    _could_fix = 0
    for y, x in zip(ys, xs):
        neighbors = (
            (y - 1, x),
            (y + 1, x),
            (y, x - 1),
            (y, x + 1)
        )
        neighbor_values = []
        n_pix = 0
        for neighbor_y, neighbor_x in neighbors:
            if (
                (neighbor_y >= exposure.shape[0] or neighbor_y < 0) or
                (neighbor_x >= exposure.shape[1] or neighbor_x < 0) or
                badpix[neighbor_y, neighbor_x] != 0 or
                not np.isfinite(exposure[neighbor_y, neighbor_x])
            ):
                continue
            else:
                neighbor_values.append(exposure[neighbor_y, neighbor_x])
                n_pix += 1
        if n_pix > 0:
            interp_exposure[y,x] = np.sum(neighbor_values) / n_pix
            _could_fix += 1
        else:
            interp_exposure[y, x] = fill_value
    debug("Able to guess {}/{} values".format(_could_fix, np.count_nonzero(badpix)))
    return interp_exposure

def mask_bad_pixels(exposure, badpix, fill_value=np.nan):
    masked_exposure = exposure.astype(float).copy()
    masked_exposure[badpix != 0] = fill_value
    debug('Replaced {} bad pixels with {}'.format(np.count_nonzero(badpix), fill_value))
    return masked_exposure

# @scratch.cache
def get_sky_cube(sky_files, bad_pixel_mask, fill_value=np.nan):
    with open(sky_files[0], 'rb') as f:
        hdul = fits.open(f)
        shape = hdul[0].data.shape
        if len(shape) == 3:
            planes, *shape = shape
        elif len(shape) == 2:
            planes = 1
        else:
            raise ValueError()
    dtype = float
    cube = np.zeros((len(sky_files) * planes,) + tuple(shape), dtype=dtype)

    def _do_linearity_badpix(data):
        corrected_data, _ = correct_linearity(data.astype(float))
        corrected_data = mask_bad_pixels(corrected_data, bad_pixel_mask, fill_value=fill_value)
        return corrected_data
    for idx, filename in enumerate(sky_files):
        with open(filename, 'rb') as f:
            hdul = fits.open(f)
            if planes == 1:
                cube[idx] = _do_linearity_badpix(hdul[0].data)
            else:
                for plane in range(planes):
                    cube[idx * planes + plane] = _do_linearity_badpix(hdul[0].data[plane])
    return cube

# @scratch.cache
def nanmedian_cube(cube):
    return np.nanmedian(cube, axis=0)

# @scratch.cache
def nanstd_cube(cube):
    return np.nanstd(cube, axis=0)

# @scratch.cache
def factorize(data_cube, n_components, strategy='PCA'):
    #all_real_cube = data_cube.copy()
    # since our resulting sky model won't need to reproduce bad pixels, this should be reasonable:
    data_cube[np.isnan(data_cube)] = 0.0
    zz, yy, xx = data_cube.shape
    all_real_cube = data_cube.reshape((zz, yy * xx))
    ts = datetime.datetime.now()
    debug("Beginning {} for {} components".format(strategy, n_components))
    if strategy == 'NMF':
        fitter = decomposition.NMF(n_components=n_components)
    elif strategy == 'PCA':
        fitter = decomposition.PCA(n_components=n_components)
    elif strategy == 'FastICA':
        fitter = decomposition.FastICA(n_components=n_components)
    elif strategy == 'IncrementalPCA':
        fitter = decomposition.IncrementalPCA(
            n_components=n_components,
            # doc says batch_size should be 5x n_features,
            # but that seems wrong (and causes a memoryerror)
            batch_size=5 * n_components,
        )
    else:
        raise ValueError("Select strategy from among NMF, PCA, and ICA")
    fitter.fit(all_real_cube)
    debug("Finished in {}".format(datetime.datetime.now() - ts))
    return fitter


# def mask_box(x, y, center, size):
#     try:
#         width, height = size
#     except TypeError:
#         width = height = size
#     center_x, center_y = center
#     return (
#         (y >= center_y - height / 2) &
#         (y <= center_y + height / 2) &
#         (x >= center_x - width / 2) &
#         (x <= center_x + width / 2)
#     )

# def mask_circle(shape, center, radius):
#     rho, _ = polar_coords(center, shape)
#     return rho <= radius

# def make_science_regions_mask(shape,
#                               badpix,
#                               top_center,
#                               bottom_center,
#                               leak_center,
#                               ghost_center=INITIAL_GUESS_GHOST_CENTER,
#                               glint_center=INITIAL_GUESS_GLINT_CENTER,
#                               leak_box_size=LEAK_BOX_SIZE,
#                               ghost_box_size=GHOST_BOX_SIZE,
#                               glint_box_size=GLINT_BOX_SIZE,
#                               psf_box_size=PSF_BOX_SIZE):

#     y, x = np.indices(shape)
#     mask = (
#         mask_circle(shape, leak_center, leak_box_size / 2) |
#         # mask_box(x, y, leak_center, leak_box_size) |
#         mask_circle(shape, top_center, psf_box_size / 2) |
#         mask_circle(shape, bottom_center, psf_box_size / 2)
#         # mask_box(x, y, top_center, psf_box_size) |
#         # mask_box(x, y, bottom_center, psf_box_size)
#     )
#     if ghost_center is not None:
#         mask |= mask_circle(shape, ghost_center, ghost_box_size / 2)
#     if glint_center is not None:
#         mask |= mask_box(glint_center[0], glint_center[1], shape, glint_box_size / 2)

#     if badpix is not None:
#         mask |= (badpix != 0).astype(mask.dtype)
#     return mask
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
        (y >= center_y - height / 2) &
        (y <= center_y + height / 2) &
        (x >= center_x - width / 2) &
        (x <= center_x + width / 2)
    )

def mask_circle(shape, center, radius):
    rho, _ = dd.polar_coords(center, shape)
    return rho <= radius

PSF_MASK_RADIUS = 226
GHOST_MASK_RADIUS = 66
SPIKE_MASK_WIDTH = 900
SPIKE_MASK_HEIGHT = 82
BLOB1_MASK_RADIUS = 214
BLOB2_MASK_RADIUS = 52
BLOB3_MASK_RADIUS = 70
CLIO2_DETECTOR_SHAPE = (512, 1024)
NEG_IMG1_DX = 518
NEG_IMG2_DX = 625
NEG_IMG_WIDTH = 71

def psf_centers_from_header(hdr):
    psf_centers = {
        'top_psf_x': hdr['TOPXC'],
        'top_psf_y': hdr['TOPYC'],
        'bottom_psf_x': hdr['BOTXC'],
        'bottom_psf_y': hdr['BOTYC'],
        'leak_psf_x': hdr['LEAKXC'],
        'leak_psf_y': hdr['LEAKYC'],
        'ghost_psf_x': hdr['GHOSTXC'],
        'ghost_psf_y': hdr['GHOSTYC'],
        'ghost_psf_found': hdr['GHOSTOK'],
        'glint_psf_x': hdr['GLINTXC'],
        'glint_psf_y': hdr['GLINTYC'],
        'glint_psf_found': hdr['GLINTOK'],
    }
    return psf_centers

def make_science_regions_mask(psf_centers=None):
    ''''''
    if psf_centers is None:
        psf_centers = {
            'top_psf_x': INITIAL_GUESS_TOP_LEFT_CENTER[0],
            'top_psf_y': INITIAL_GUESS_TOP_LEFT_CENTER[1],
            'bottom_psf_x': INITIAL_GUESS_BOTTOM_LEFT_CENTER[0],
            'bottom_psf_y': INITIAL_GUESS_BOTTOM_LEFT_CENTER[1],
            'leak_psf_x': INITIAL_GUESS_LEAK_LEFT_CENTER[0],
            'leak_psf_y': INITIAL_GUESS_LEAK_LEFT_CENTER[1],
            'ghost_psf_x': INITIAL_GUESS_GHOST_CENTER[0],
            'ghost_psf_y': INITIAL_GUESS_GHOST_CENTER[1],
            'ghost_psf_found': True,
            'glint_psf_x': INITIAL_GUESS_GLINT_CENTER[0],
            'glint_psf_y': INITIAL_GUESS_GLINT_CENTER[1],
            'glint_psf_found': True,
        }
    y, x = np.indices(CLIO2_DETECTOR_SHAPE)
    topxc, topyc = psf_centers['top_psf_x'], psf_centers['top_psf_y']
    top_to_blob1_dx, top_to_blob1_dy = 470, -51
    blob1_center = topxc + top_to_blob1_dx, topyc + top_to_blob1_dy
    top_to_blob2_dx, top_to_blob2_dy = 468.4, -269.4
    blob2_center = topxc + top_to_blob2_dx, topyc + top_to_blob2_dy
    top_to_blob3_dx, top_to_blob3_dy = 595, -137
    blob3_center = topxc + top_to_blob3_dx, topyc + top_to_blob3_dy
    second_ghost_dx, second_ghost_dy = -63, 179
    mask = (
        mask_circle(CLIO2_DETECTOR_SHAPE, (topxc, topyc), PSF_MASK_RADIUS) |
        mask_circle(CLIO2_DETECTOR_SHAPE, (psf_centers['bottom_psf_x'], psf_centers['bottom_psf_y']), PSF_MASK_RADIUS) |
        mask_circle(CLIO2_DETECTOR_SHAPE, blob1_center, BLOB1_MASK_RADIUS) |
        mask_circle(CLIO2_DETECTOR_SHAPE, blob2_center, BLOB2_MASK_RADIUS) |
        mask_circle(CLIO2_DETECTOR_SHAPE, blob3_center, BLOB3_MASK_RADIUS) |
        mask_box(CLIO2_DETECTOR_SHAPE, (topxc, topyc), (SPIKE_MASK_WIDTH, SPIKE_MASK_HEIGHT), -18) |
        mask_box(CLIO2_DETECTOR_SHAPE, (psf_centers['bottom_psf_x'], psf_centers['bottom_psf_y']), (SPIKE_MASK_WIDTH, SPIKE_MASK_HEIGHT), -27) |
        mask_box(CLIO2_DETECTOR_SHAPE, (topxc + NEG_IMG1_DX, 512/2), (NEG_IMG_WIDTH, 512)) |
        mask_box(CLIO2_DETECTOR_SHAPE, (topxc + NEG_IMG2_DX, 512/2), (NEG_IMG_WIDTH, 512))
    )
    if psf_centers['ghost_psf_found']:
        xc, yc = psf_centers['ghost_psf_x'], psf_centers['ghost_psf_y']
        mask |= mask_circle(CLIO2_DETECTOR_SHAPE, (xc, yc), GHOST_MASK_RADIUS)
        mask |= mask_circle(CLIO2_DETECTOR_SHAPE, (xc + second_ghost_dx, yc + second_ghost_dy), GHOST_MASK_RADIUS)
    if psf_centers['glint_psf_found']:
        mask |= mask_box(CLIO2_DETECTOR_SHAPE, (psf_centers['glint_psf_x'], psf_centers['glint_psf_y']), GLINT_BOX_SIZE)

    return mask

def smooth(data, kernel_stddev_px):
    return convolve_fft(
        data,
        Gaussian2DKernel(kernel_stddev_px),
        boundary='wrap'
    )

CLIO2_NORTH_DEG = -1.797

def calculate_derotation_angle(rotoff_val):
    '''This is the counterclockwise angle to rotate the data
    in order to align +Y with +Dec (North Up)
    '''
    # Morz et al 2015 p 18:
    # > Derotation Angle: DEROT_Clio = ROTOFF - 180 + NORTH_Clio
    # > A positive angle is counterclockwise to get north up and east left.
    # DEROT = ROTOFF - 180 + NORTH
    # = ROTOFF - 180 + -1.797
    return rotoff_val - 180 + CLIO2_NORTH_DEG

from scipy.ndimage import rotate as scipy_rotate

def derotate(data, derotation_angle):
    '''Derotate `data` by `derotation_angle`, defined such that positive derotation
    is CCW.
    
    n.b. `scipy.ndimage.rotate` rotates CW when 0,0 at bottom left 
    '''
    return scipy_rotate(data, -derotation_angle, reshape=False)


def construct_centered_wcs(npix, ref_ra, ref_dec, rot_deg, deg_per_px):
    '''
    Arguments
    ---------
    npix : int
        number of pixels for a square cutout
    ref_ra : float
        right ascension in degrees
    ref_dec : float
        declination in degrees
    rot_deg : float
       angle between +Y pixel and +Dec sky axes

    Note: FITS images are 1-indexed, with (x,y) = (1,1)
    placed at the lower left when displayed. To place North up,
    the data should be rotated clockwise by `rot_deg`.
    '''
    # +X should be -RA when N is up and E is left
    # +Y should be +Dec when N is up and E is left
    scale_m = np.matrix([
        [-deg_per_px, 0],
        [0, deg_per_px]
    ])
    theta_rad = -1 * np.deg2rad(rot_deg)
    rotation_m = np.matrix([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    the_wcs = wcs.WCS(naxis=2)
    the_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    the_wcs.wcs.crpix = [npix / 2 + 1, npix / 2 + 1]  # FITS is 1-indexed
    the_wcs.wcs.crval = [ref_ra, ref_dec]
    the_wcs.wcs.cd = rotation_m @ scale_m
    return the_wcs

def fill_masked(data, mask, fill='nan'):
    assert data.shape == (512, 1024)
    output = data.copy().astype('=f8')
    if not np.isscalar(fill):
        try:
            fill_value = fill[mask]
        except:
            raise ValueError("Fill should be the same shape as data, or one of: 'nan', 'median', 'zero'.")
    elif fill == 'nan':
        fill_value = np.nan
    elif fill == 'median':
        fill_value = np.median(data[~mask])
    elif fill == 'zero':
        fill_value = 0.0
    else:
        raise ValueError("Fill should be the same shape as data, or one of: 'nan', 'median', 'zero'.")
    output[mask] = fill_value
    return output

def reconstruct_masked(original_image, fitter, mask, fill='median', how='project', display=False):
    '''
    mask pixels with True are excluded (either by replacing
    with `fill` or ignored in least-squares fit). fit uses only "~mask" pixels
    '''
    if how == 'project':
        clean_image = fill_masked(original_image, mask, fill=fill)
        reconstruction = fitter.inverse_transform(fitter.transform(
            clean_image.flatten()[np.newaxis,:]
        ))[0].reshape(clean_image.shape)
    elif how == 'fit':
        mask_1d = mask.flatten()
        x, residuals, rank, s = np.linalg.lstsq(
            fitter.components_[:,~mask_1d].T,
            original_image.flatten()[~mask_1d] - fitter.mean_[~mask_1d],
            rcond=None
        )
        reconstruction = (np.dot(fitter.components_.T, x) + fitter.mean_).reshape(original_image.shape)
        if display:
            import matplotlib.pyplot as plt
            plt.subplot(141)
            plt.imshow(original_image)
            plt.subplot(142)
            masked_image = original_image.copy()
            masked_image[mask] = np.nan
            plt.imshow(masked_image)
            plt.subplot(143)
            plt.imshow(reconstruction)
            plt.subplot(144)
            masked_reconstruction = reconstruction.copy()
            masked_reconstruction[mask] = np.nan
            plt.imshow(masked_image - masked_reconstruction)
    return reconstruction

def _subplots_agg(nrows=1, ncols=1, **kwargs):
    fig = Figure(**kwargs)
    canvas = FigureCanvasAgg(fig)
    axes = [fig.add_subplot(nrows, ncols, i) for i in range(1, nrows * ncols + 1)]
    return fig, axes

def reconstruction_rms(row, fitter, sky_reference, badpix, plots_dir=None,
                       percentile_min=5, percentile_max=95):
    with open(row['instrument_corrected_filename'], 'rb') as f:
        hdul = fits.open(f, memmap=False)
        original_image = hdul[0].data
    mask = make_science_regions_mask(
        badpix.shape,
        badpix,
        leak_center=(row['leak_psf_x'], row['leak_psf_y']),
        top_center=(row['top_psf_x'], row['top_psf_y']),
        bottom_center=(row['bottom_psf_x'], row['bottom_psf_y']),
        ghost_center=(row['ghost_psf_x'], row['ghost_psf_y']),
    )
    clean_image = fill_masked(original_image, mask, fill=sky_reference)
    reconstruction = reconstruct_masked(original_image, fitter, mask, fill=sky_reference)
    diff = original_image - reconstruction
    overall_rms = np.sqrt(np.nanmean(diff**2))
    masked_rms = np.sqrt(np.nanmean(diff[mask]**2))
    unmasked_rms = np.sqrt(np.nanmean(diff[~mask]**2))

    # Use the row index for the plot name so we can put them in a single movie
    base_plot_name = 'frame{:06}.png'.format(row[0])  # Is index always the first col?
    # base_plot_name = os.path.basename(row['instrument_corrected_filename']).replace('.fits', '.png')
    background_subtracted_png = os.path.join(plots_dir, base_plot_name)
    figure_path = os.path.join(plots_dir, 'compare_' + base_plot_name)
    if True: # not os.path.exists(background_subtracted_png) or not os.path.exists(figure_path):
        # comparison plot
#         fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, figsize=(14, 4))
        fig, [ax1, ax2, ax3, ax4] = _subplots_agg(1, 4, figsize=(14, 4))
        vmin, vmax = (
            np.nanpercentile(original_image, percentile_min),
            np.nanpercentile(original_image, percentile_max)
        )
        ax1.imshow(original_image, vmin=vmin, vmax=vmax)
        ax1.imshow(mask, cmap='viridis', alpha=0.3)
        ax2.imshow(clean_image, vmin=vmin, vmax=vmax)
        ax3.imshow(reconstruction, vmin=vmin, vmax=vmax)
        diff_vmax = np.nanpercentile(diff, percentile_max)
        diff_vmin = np.nanpercentile(diff, percentile_min)
        diff_im = ax4.imshow(diff, vmin=diff_vmin, vmax=diff_vmax, cmap='RdBu_r')
        add_colorbar(diff_im)
        ax4.set_title("RMS overall: {:3.0f} counts\n"
                      "in masked region: {:3.0f} counts\n"
                      "elsewhere: {:3.0f} counts".format(overall_rms, masked_rms, unmasked_rms),
                      loc='left')
        fig.savefig(figure_path)
        # visualization of subtracted frame
        imsave(background_subtracted_png, diff, vmin=diff_vmin, vmax=diff_vmax)
    #plt.close('all')
    return {
        'background_overall_rms': overall_rms,
        'background_masked_rms': masked_rms,
        'background_unmasked_rms': unmasked_rms,
        'background_figure': figure_path,
        'background_subtracted_png': background_subtracted_png,
    }


def register_simulated_to_data(simulated, data, intensity_scale_factor, tx, ty, scale, rot, blur_fwhm_px, display=False):
    sim_y_ctr, sim_x_ctr = np.unravel_index(np.nanargmax(simulated), simulated.shape)
    if rot == 0:
        rotated_simulated = simulated
    else:
        rotated_simulated = rotate(simulated, rot)
    # print('scale', scale)
    if scale == 1:
        scaled_rotated_simulated = rotated_simulated
    else:
        # print('scaling ', rotated_simulated.shape, 'by', scale)
        scaled_rotated_simulated = clipped_zoom(rotated_simulated, scale)
    # print('scaled_rotated_simulated.shape', scaled_rotated_simulated.shape)

    if tx == 0 and ty == 0:
        translated_scaled_rotated_simulated = scaled_rotated_simulated
    else:
        translated_scaled_rotated_simulated = warp(scaled_rotated_simulated, AffineTransform(translation=(-tx, -ty)))

    if blur_fwhm_px is not None:
        blurred_translated_scaled_rotated_simulated = smooth(translated_scaled_rotated_simulated, FWHM_TO_STDDEV * blur_fwhm_px)
    else:
        blurred_translated_scaled_rotated_simulated = translated_scaled_rotated_simulated
    # need to crop to central region before subtracting
    data_height, data_width = data.shape
    simulated_height, simulated_width = simulated.shape
    # print('shape of blurred_translated_scaled_rotated_simulated', blurred_translated_scaled_rotated_simulated.shape)

    yfrom, yto = sim_y_ctr - data_height // 2, sim_y_ctr + data_height // 2
    xfrom, xto = sim_x_ctr - data_width // 2, sim_x_ctr + data_width // 2
    # print(yfrom, yto, xfrom, xto)

    cropped = blurred_translated_scaled_rotated_simulated[yfrom:yto,xfrom:xto]
    # print('cropped shape', cropped.shape)
    cropped *= (intensity_scale_factor * np.nansum(data)) / np.nansum(cropped)

    if display:
        vmin, vmax = 0.1, np.nanpercentile(data, 99.9)
        ticks = [0, 85, 170]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 3))
        plt.subplot(131)
#         print(data.shape)
#         print(cropped.shape)
#         masked_data = data.copy()
#         mask = mask_center(masked_data.shape, (masked_data.shape[0] // 2 + 10, masked_data.shape[1] // 2 + 10), 25)
#         masked_data[mask] = np.nan
        plt.imshow(data, vmin=vmin, vmax=vmax, cmap='viridis', norm=LogNorm())
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.grid()
        plt.colorbar()
        plt.subplot(132)
#         masked_cropped = cropped.copy()
#         masked_cropped[mask] = np.nan
        plt.imshow(cropped, vmin=vmin, vmax=vmax, cmap='viridis', norm=LogNorm())
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.grid()
        plt.colorbar()
        plt.subplot(133)
        resid = data - cropped
        plt.imshow(resid, cmap='RdBu_r', vmin=-np.nanmax(np.abs(resid)), vmax=np.nanmax(np.abs(resid)))
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.grid()
        plt.colorbar()
    return cropped

def compute_psf_registration(simulated, data, intensity_scale_factor=1, tx=0, ty=0, scale=1.0, rot=0, blur_fwhm_px=0.1, display=False):
    # Apply initial warp parameters to get a PSF to cross-correlate
    # print('scale at compute_psf_registration stage 1', scale)
    sim_stage1 = register_simulated_to_data(
        simulated, data,
        intensity_scale_factor=intensity_scale_factor,
        tx=tx,
        ty=ty,
        scale=scale,
        rot=rot,
        blur_fwhm_px=blur_fwhm_px,
        display=display
    )
    # cross-correlate to find translation
    (updated_tx, updated_ty), _, __ = register_translation(data, sim_stage1, upsample_factor=100)
    # sim_stage2 = register_simulated_to_data(
    #     simulated, data,
    #     intensity_scale_factor=intensity_scale_factor,
    #     tx=updated_tx,
    #     ty=updated_ty,
    #     scale=scale,
    #     rot=rot,
    #     blur_fwhm_px=blur_fwhm_px,
    #     display=display
    # )
    def minimizer(x):
        intensity_scale_factor, tx, ty, scale, rot, blur_fwhm_px = x
        aligned_sim = register_simulated_to_data(
            simulated, data, intensity_scale_factor,
            updated_tx, updated_ty, scale, rot, blur_fwhm_px, display=display
        )
        result_as_kwargs = {
            'intensity_scale_factor': intensity_scale_factor,
            'tx': tx,
            'ty': ty,
            'scale': scale,
            'rot': rot,
            'blur_fwhm_px': blur_fwhm_px,
        }
        print(result_as_kwargs)
        rms = np.sqrt(np.nanmean(data**2 * (data - aligned_sim)**2))
        return rms
    result = minimize(
        minimizer,
        x0=(intensity_scale_factor, updated_tx, updated_ty, scale, rot, blur_fwhm_px),
        #bounds=[(0.5, 1.5), (-10, 10), (-10, 10), (0.1, 2.0), (0, 359), (2, 10)]
    )
    intensity_scale_factor, tx, ty, scale, rot, blur_fwhm_px = result.x
    result_as_kwargs = {
        'intensity_scale_factor': intensity_scale_factor,
        'tx': tx,
        'ty': ty,
        'scale': scale,
        'rot': rot,
        'blur_fwhm_px': blur_fwhm_px,
    }
    return result_as_kwargs

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

def regrid_image(image, x_prime, y_prime, method='cubic', mask=None):
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
    '''
    if mask is not None:
        mask = mask.copy()
        mask &= np.isfinite(image)
    else:
        mask = np.isfinite(image)
    weights = np.ones_like(image)
    weights[~mask] = 0
    yy, xx = np.indices(image.shape)
    xx_sub = xx[mask]
    yy_sub = yy[mask]
    zz = image[mask]
    new_image = interpolate.griddata(
        np.stack((xx_sub.flat, yy_sub.flat), axis=-1),
        zz.flatten(),
        (x_prime, y_prime),
        method=method).reshape(x_prime.shape)
    new_mask = interpolate.griddata(
        np.stack((xx.flat, yy.flat), axis=-1),
        weights.flatten(),
        (x_prime, y_prime),
        method=method).reshape(x_prime.shape)
    return new_image, new_mask

def is_iterable(obj):
    try:
        iterator = iter(obj)
    except TypeError:
        return False
    else:
        return True

import queue
import threading

def multi_invoke_threaded(command, *args, max_processes=1, poll_wait_interval=0.05):
    '''
    Invoke a command multiple times with varying arguments and
    wait for all jobs to complete

    Parameters
    ----------
    command : string
        Prefix from which to construct command
    *args : string or other iterable
        Non-string iterables have their values used to generate
        command line strings, string values are kept static.
        Non-string iterables must all have the same length.
    max_processes : int
        Number of processes to run concurrently. Defaults to 1
        (serial execution).

    Examples
    --------

    >>> multi_invoke('echo', '-n', ['cats', 'dogs'], 'are great')
    # spawns "echo -n cats are great" and "echo -n dogs are great"
    '''
    n_arg_variants = set(map(len, (seq for seq in args if is_iterable(seq) and not isinstance(seq, str))))
    assert len(n_arg_variants) == 1, 'Sequences of arguments must be the same length'
    n_arg_variants = list(n_arg_variants)[0]
    commands = queue.Queue()
    for i in range(n_arg_variants):
        cmd = command + ' '
        for item in args:
            if isinstance(item, str) or not is_iterable(item):
                arg = item
            else:
                arg = item[i]
            cmd += '{} '.format(arg)
        commands.put(cmd)
    n_processes = max_processes if n_arg_variants > max_processes else n_arg_variants
    STOP_SENTINEL = None
    def worker():
        while True:
            item = commands.get()
            if item is STOP_SENTINEL:
                break
            subprocess.check_call(shlex.split(item))
    threads = []
    for i in range(n_processes):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    # Stop all threads
    for i in range(n_processes):
        commands.put(STOP_SENTINEL)
    for t in threads:
        t.join()

def multi_invoke(command, *args, max_processes=1, poll_wait_interval=0.05, poll_sleep_interval=0.1):
    '''
    Invoke a command multiple times with varying arguments and
    wait for all jobs to complete

    Parameters
    ----------
    command : string
        Prefix from which to construct command
    *args : string or other iterable
        Non-string iterables have their values used to generate
        command line strings, string values are kept static.
        Non-string iterables must all have the same length.
    max_processes : int
        Number of processes to run concurrently. Defaults to 1
        (serial execution).
    poll_wait_interval : float
        Number of seconds to wait on a process each iteration
        of the polling loop.
    poll_sleep_interval : float
        Number of seconds to sleep each loop

    Examples
    --------

    >>> multi_invoke('echo', '-n', ['cats', 'dogs'], 'are great')
    # spawns "echo -n cats are great" and "echo -n dogs are great"
    '''
    n_args = set(map(len, (seq for seq in args if is_iterable(seq) and not isinstance(seq, str))))
    assert len(n_args) == 1, 'Sequences of arguments must be the same length'
    n_args = list(n_args)[0]
    commands = []
    running = set()
    for i in range(n_args):
        cmd = command + ' '
        for item in args:
            if isinstance(item, str) or not is_iterable(item):
                arg = item
            else:
                arg = item[i]
            cmd += '{} '.format(arg)
        commands.append(cmd)
    n_processes = max_processes if len(commands) > max_processes else len(commands)

    def _start(cmd):
        info("Starting: {}".format(cmd))
        running.add(subprocess.Popen(shlex.split(cmd)))

    try:
        # Make initial set of subprocesses
        for i in range(n_processes):
            cmd = commands.pop()
            _start(cmd)
        # Consume commands queue
        while commands:
            finished_processes = set()
            for p in running:
                try:
                    p.wait(poll_wait_interval)
                    finished_processes.add(p)
                except subprocess.TimeoutExpired:
                    pass
            running.difference_update(finished_processes)
            for proc in finished_processes:
                if proc.returncode != 0:
                    raise RuntimeError("Process exited with {} ({})".format(proc.returncode, ' '.join(proc.args)))
                info("Finished: {}".format(' '.join(proc.args)))
                if commands:
                    # n.b. commands can be exhausted
                    cmd = commands.pop()
                    _start(cmd)
            time.sleep(poll_sleep_interval)
        # Ensure all processes complete by waiting for them
        while running:
            proc = running.pop()
            info("Waiting on {}".format(proc.args))
            proc.wait()
    finally:
        # If a subprocess fails to exit cleanly (or another exception happens)
        # kill the rest of them
        for proc in running:
            info("Terminating {}".format(proc.args))
            proc.terminate()


def scale_models_to_cutouts(cutouts, model_cutouts, min_r_px=40, max_r_px=100, display=True):
    res = {'TOP': None, 'BOT': None}
    import matplotlib.pyplot as plt
    fig, (ee_ax, prof_ax) = plt.subplots(nrows=2, sharex=True, figsize=(10,10))
    for loc in ('TOP', 'BOT'):
        cutout_data = cutouts[loc+'_SCI'].data
        model_data = model_cutouts[loc+'_SCI'].data
        ctr = (cutout_data.shape[0] - 1) / 2
        _, _, radii, prof_to_match = dd.encircled_energy_and_profile(cutout_data, (ctr, ctr))
        mask = (radii >= min_r_px) & (radii <= max_r_px)
    
        def minimize_me(x, model_image):
            amplitude_scale_factor = x
            _, _, _, this_prof = dd.encircled_energy_and_profile(
                amplitude_scale_factor * model_image, (ctr, ctr)
            )
            squared_errors = (this_prof[mask] - prof_to_match[mask])**2
            return np.sqrt(np.average(squared_errors))
        scale_factor = fmin(
            minimize_me,
            x0=(np.nanmax(cutout_data) / np.nanmax(model_data),),
            args=(model_data,)
        )
        res[loc] = scale_factor
        model_cutouts[loc+'_SCI'].data *= scale_factor
        dd.encircled_energy_and_profile(
            model_cutouts[loc+'_SCI'].data, (ctr, ctr),
            arcsec_per_px=CLIO2_PIXEL_SCALE.value,
            display=True,
            ee_ax=ee_ax,
            profile_ax=prof_ax,
            label=loc + ' model',
            normalize=(CLIO2_PIXEL_SCALE * min_r_px * u.pixel).to(u.arcsec).value
        )
        dd.encircled_energy_and_profile(
            cutout_data, (ctr, ctr),
            arcsec_per_px=CLIO2_PIXEL_SCALE.value,
            display=True,
            ee_ax=ee_ax,
            profile_ax=prof_ax,
            label=loc + ' ref',
            normalize=(CLIO2_PIXEL_SCALE * min_r_px * u.pixel).to(u.arcsec).value
        )
    cropped_top, cropped_bot = crop_paired_frames(model_cutouts['TOP_SCI'].data, model_cutouts['BOT_SCI'].data)
    # now we've cropped NaNs off, pad to PSF_BOX_SIZE
    dpix = PSF_BOX_SIZE - cropped_top.shape[0]
    if dpix != 0:
        # Are we undersized?
        assert dpix % 2 == 0, f"Can't pad to full PSF box size, non-even difference {dpix}"
        pad_pix = int(dpix // 2)
        model_cutouts['TOP_SCI'].data = np.zeros((PSF_BOX_SIZE, PSF_BOX_SIZE))
        model_cutouts['TOP_SCI'].data[pad_pix:-pad_pix,pad_pix:-pad_pix] = cropped_top
        model_cutouts['BOT_SCI'].data = np.zeros((PSF_BOX_SIZE, PSF_BOX_SIZE))
        model_cutouts['BOT_SCI'].data[pad_pix:-pad_pix,pad_pix:-pad_pix] = cropped_bot
    else:
        # Maybe nothing needed to be cropped
        model_cutouts['TOP_SCI'].data = cropped_top
        model_cutouts['BOT_SCI'].data = cropped_bot
    return model_cutouts


def simulate_magellan_vapp_psf(vapp_optic, wavelength, rotation_deg, size):
    pixelscale_arcsec_per_px = CLIO2_PIXEL_SCALE
    support_width = 3.81 * u.cm # 1.5 inches
    support_angles = np.deg2rad(np.arange(45, 360, 90))  # meters
    offset = MAGELLAN_SPIDERS_OFFSET.to(u.m).value
    pupil_mask = poppy.CompoundAnalyticOptic([
        poppy.CircularAperture(radius=MAGELLAN_PRIMARY_STOP_DIAMETER / 2),
        poppy.AsymmetricSecondaryObscuration(
            support_angle=support_angles,
            support_width=support_width,
            support_offset_x=-offset * np.cos(support_angles),
            support_offset_y=offset * np.sin(support_angles),
            secondary_radius=MAGELLAN_SECONDARY_DIAMETER / 2
        )
    ])
    osys = poppy.OpticalSystem(oversample=1)
    osys.add_pupil(pupil_mask)
    osys.add_pupil(vapp_optic)
    osys.add_rotation(rotation_deg)
    osys.add_detector(pixelscale_arcsec_per_px, fov_pixels=size)
    psf = osys.calc_psf(wavelength=wavelength, normalize='last')
    return psf


class VAPPDataset:
    '''Data container for aligned, paired PSFs and
    their derotation angles
    '''
    @classmethod
    def from_file(cls, filename):
        hdul = fits.open(filename)
        return cls(hdul)
    @staticmethod
    def _validate_shapes(top_shape, bot_shape):
        if not top_shape == bot_shape:
            raise ValueError(f'Shape mismatch between top and bot: {top_shape} != {bot_shape}')
        if top_shape[-1] != top_shape[-2]:
            raise ValueError(f'Non-square images: {top_shape[-1]} != {top_shape[-2]} in {top_shape}')
        return True
    @staticmethod
    def _construct_extensions(top_cube, bot_cube, angles):
        top_ext = fits.ImageHDU(top_cube)
        top_ext.header['EXTNAME'] = 'TOP_SCI'
        bot_ext = fits.ImageHDU(bot_cube)
        bot_ext.header['EXTNAME'] = 'BOT_SCI'
        angles_ext = fits.ImageHDU(angles)
        angles_ext.header['EXTNAME'] = 'ANGLES'
        angles_ext.header.add_comment('Derotation angles defined as the counterclockwise (when pixel 1,1 is at lower left) rotation to align +Y and +Dec')
        return top_ext, bot_ext, angles_ext
    @classmethod
    def from_source_files(cls, source_files):
        nframes = len(source_files)
        if not nframes:
            raise RuntimeError("No frames to combine!")
        primary_hdu = fits.PrimaryHDU()
        hdul = fits.HDUList([primary_hdu])
        with open(source_files[0], 'rb') as f:
            first_frame = fits.open(f)
            first_frame_top = first_frame['TOP_SCI'].data
            first_frame_bot = first_frame['BOT_SCI'].data
            VAPPDataset._validate_shapes(first_frame_top.shape, first_frame_bot.shape)
            top_cube = np.zeros((nframes,) + first_frame_top.shape)
            bot_cube = np.zeros((nframes,) + first_frame_top.shape)
            angles = np.zeros(nframes)
            # TODO restore after finding out what happened to hr3188
            # for key in ('DATE-OBS', 'WAVEL'):
                # primary_hdu.header[key] = first_frame[0].header[key]
            # primary_hdu.header['WAVEL'] = 3.9
        
        for idx, fn in enumerate(sorted(source_files)):
            with fits.open(fn, memmap=False) as frame_hdul:
                top_cube[idx] = frame_hdul['TOP_SCI'].data
                bot_cube[idx] = frame_hdul['BOT_SCI'].data
                angles[idx] = calculate_derotation_angle(frame_hdul[0].header['ROTOFF'])
            del frame_hdul
        top_cube, bot_cube = crop_paired_cubes(top_cube, bot_cube)
        assert np.count_nonzero(bot_cube[-1] == 0) == 0
        top_ext, bot_ext, angles_ext = cls._construct_extensions(top_cube, bot_cube, angles)
        hdul.extend([
            top_ext,
            bot_ext,
            angles_ext
        ])
        return cls(hdul)
    def __init__(self, hdul):
        '''Initialize vAPP dataset given astropy.io.fits.HDUList
        with `TOP_SCI`, `BOT_SCI`, and `ANGLES` extensions.
        '''
        self.hdul = hdul
        self.header = hdul[0].header
        self.top_cube = hdul['TOP_SCI'].data.astype('=f8')
        self.bot_cube = hdul['BOT_SCI'].data.astype('=f8')
        self.angles = hdul['ANGLES'].data.astype('=f8')
        VAPPDataset._validate_shapes(self.top_cube.shape, self.bot_cube.shape)
        self.bot_mask, self.top_mask = make_both_masks(self.header['WAVEL'], self.top_cube.shape[1:])
    def make_downsampled_dataset(self, chunk_size):
        new_hdul = fits.HDUList([self.hdul[0]])
        new_top_cube = dd.downsample_first_axis(self.top_cube, chunk_size)
        new_bot_cube = dd.downsample_first_axis(self.bot_cube, chunk_size)
        new_angles = dd.downsample_first_axis(self.angles, chunk_size)
        new_hdul.extend(self._construct_extensions(new_top_cube, new_bot_cube, new_angles))
        return VAPPDataset(new_hdul)
    def writeto(self, *args, **kwargs):
        self.hdul.writeto(*args, **kwargs)
    @property
    def npix(self):
        return self.top_cube.shape[1]
    @property
    def frame_shape(self):
        return self.top_cube.shape[1:]
    @property
    def nframes(self):
        return self.top_cube.shape[0]
    @property
    def center(self):
        return (self.npix - 1) / 2, (self.npix - 1) / 2
    @property
    def top_cube_flipped(self):
        return np.flip(self.top_cube, axis=(-2, -1))
    @property
    def bot_cube_flipped(self):
        return np.flip(self.bot_cube, axis=(-2, -1))
    @property
    def top_dark_half_mask(self):
        rho, theta = dd.polar_coords(self.center, self.frame_shape)
        return (
            (theta < np.deg2rad(-90 - VAPP_PSF_ROTATION_DEG)) |
            (
                (theta > np.deg2rad(90 - VAPP_PSF_ROTATION_DEG)) & (theta > 0)
            )
        )
    @property
    def bot_dark_half_mask(self):
        return ~self.top_half_mask
    top_light_half_mask = bot_dark_half_mask
    bot_light_half_mask = top_dark_half_mask

def count_nans(arr):
    return np.count_nonzero(np.isnan(arr))

def determine_frames_crop_amount(frame_a, frame_b):
    the_slice = slice(0,frame_a.shape[1])
    crop_px = 0
    while (
        count_nans(frame_a[the_slice,the_slice]) != 0 or 
        count_nans(frame_b[the_slice,the_slice]) != 0
    ):
        crop_px += 1
        the_slice = slice(crop_px,-crop_px)
    return crop_px

def determine_cubes_crop_amount(top_cube, bot_cube):
    # note *not* nansum, since we want nans to propagate
    top_combined = np.sum(top_cube, axis=0)
    bot_combined = np.sum(bot_cube, axis=0)
    return determine_frames_crop_amount(top_combined, bot_combined)

def crop_paired_frames(top_frame, bot_frame):
    crop_px = determine_frames_crop_amount(top_frame, bot_frame)
    if crop_px > 0:
        return top_frame[crop_px:-crop_px,crop_px:-crop_px], bot_frame[crop_px:-crop_px,crop_px:-crop_px]
    return top_frame, bot_frame

def crop_paired_cubes(top_cube, bot_cube):
    crop_px = determine_cubes_crop_amount(top_cube, bot_cube)
    if crop_px > 0:
        cropped_top_cube = top_cube[:,crop_px:-crop_px,crop_px:-crop_px]
        cropped_bot_cube = bot_cube[:,crop_px:-crop_px,crop_px:-crop_px]
        return cropped_top_cube, cropped_bot_cube
    return top_cube, bot_cube

if __name__ == "__main__":
    print("herp a derp")
