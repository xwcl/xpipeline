import dask
import dask.bag as db
import numpy as np
from astropy.io import fits
import astropy.units as u

from ..tasks import iofits, improc
from .. import constants
from . import magellan

# Morzinski 2015 ApJ Table 18
CLIO2_PIXEL_SCALE = 15.846e-3 * (u.arcsec / u.pixel)
# HAWAII-H1RG (http://www.teledyne-si.com/products/Documents/H1RG%20Brochure%20-%20September%202017.pdf)
CLIO2_PIXEL_PITCH = 18 * u.um
# From Otten+ 2017
VAPP_PSF_ROTATION_DEG = 26
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

def lambda_over_d_to_arcsec(lambda_over_d, wavelength, d=magellan.PRIMARY_MIRROR_DIAMETER):
    unit_lambda_over_d  = (wavelength.to(u.m) / d.to(u.m)).si.value * u.radian
    return (lambda_over_d  * unit_lambda_over_d).to(u.arcsec)
def arcsec_to_lambda_over_d(arcsec, wavelength, d=magellan.PRIMARY_MIRROR_DIAMETER):
    unit_lambda_over_d = ((wavelength.to(u.m) / d.to(u.m)).si.value * u.radian).to(u.arcsec)
    lambda_over_d = (arcsec / unit_lambda_over_d).si
    return lambda_over_d

def lambda_over_d_to_pixel(lambda_over_d, wavelength, d=magellan.PRIMARY_MIRROR_DIAMETER):
    arcsec = lambda_over_d_to_arcsec(lambda_over_d, wavelength, d=d)
    return (arcsec / CLIO2_PIXEL_SCALE).to(u.pixel)
def pixel_to_lambda_over_d(px, wavelength, d=magellan.PRIMARY_MIRROR_DIAMETER):
    arcsec = (px * CLIO2_PIXEL_SCALE).to(u.arcsec)
    return arcsec_to_lambda_over_d(arcsec, wavelength, d=d)

@dask.delayed
def split_frames_cube(hdul):
    new_hdul = hdul.copy()
    # I base everything on DATE. so in the extracted-from-cubes files,
    # the first file is given DATE-OBS = DATE from the cube file 
    # itself, and subsequent frames are given DATE-OBS = DATE + n*EXPTIME
    # - jrmales

    #TODO
    # return new_hdul

@dask.delayed
def correct_linearity(hdu):
    return hdu

# Morzinski+ 2015 ApJ -- Appendix B
MORZINSKI_COEFFICIENTS = (112.575, 1.00273, -1.40776e-6, 4.59015e-11)
MORZINSKI_DOMAIN = MORZINSKI_LINEARITY_MIN, MORZINSKI_LINEARITY_MAX = [2.7e4, 4.5e4]
MORZINSKI_RANGE = MORZINSKI_CORRECTED_MIN, MORZINSKI_CORRECTED_MAX = [
    sum([coeff * x**idx for idx, coeff in enumerate(MORZINSKI_COEFFICIENTS)])
    for x in MORZINSKI_DOMAIN
]


# CLIO2_DETECTOR_SHAPE = (512, 1024)

# PSF_FINDING_SMOOTH_KERNEL_STDDEV_PX = 9
# LEAK_BOX_SIZE = 75
# PSF_BOX_SIZE = 227
# PSF_DEAD_CENTER = (PSF_BOX_SIZE - 1) / 2, (PSF_BOX_SIZE - 1) / 2
# GHOST_SEARCH_BOX_SIZE = 350, 150
# GHOST_BOX_SIZE = 110
# INITIAL_GUESS_LEAK_LEFT_CENTER = 250, 287
# INITIAL_GUESS_LEAK_RIGHT_CENTER = 630, 311
# GLINT_BOX_SIZE = 90
# INITIAL_GUESS_GLINT_CENTER = 130, 95
# INITIAL_GUESS_GHOST_CENTER = 800, 350
# GLIMMER_BOX_SIZE = 40
# INITIAL_GUESS_GLIMMER_CENTER = 185, 355
# PSF_OFFSET = -60, 130
# INITIAL_GUESS_TOP_LEFT_CENTER = INITIAL_GUESS_LEAK_LEFT_CENTER[0] + PSF_OFFSET[0], INITIAL_GUESS_LEAK_LEFT_CENTER[1] + PSF_OFFSET[1]
# INITIAL_GUESS_BOTTOM_LEFT_CENTER = INITIAL_GUESS_LEAK_LEFT_CENTER[0] - PSF_OFFSET[0], INITIAL_GUESS_LEAK_LEFT_CENTER[1] - PSF_OFFSET[1]
# INITIAL_GUESS_TOP_RIGHT_CENTER = INITIAL_GUESS_LEAK_RIGHT_CENTER[0] + PSF_OFFSET[0], INITIAL_GUESS_LEAK_RIGHT_CENTER[1] + PSF_OFFSET[1]
# INITIAL_GUESS_BOTTOM_RIGHT_CENTER = INITIAL_GUESS_LEAK_RIGHT_CENTER[0] - PSF_OFFSET[0], INITIAL_GUESS_LEAK_RIGHT_CENTER[1] - PSF_OFFSET[1]

# PSF_LOCATIONS_WHAT_WHERE = [
#     ('top_left', INITIAL_GUESS_TOP_LEFT_CENTER, PSF_BOX_SIZE),
#     ('bottom_left', INITIAL_GUESS_BOTTOM_LEFT_CENTER, PSF_BOX_SIZE),
#     ('leak_left',  INITIAL_GUESS_LEAK_LEFT_CENTER, LEAK_BOX_SIZE),
#     ('top_right', INITIAL_GUESS_TOP_RIGHT_CENTER, PSF_BOX_SIZE),
#     ('bottom_right', INITIAL_GUESS_BOTTOM_RIGHT_CENTER, PSF_BOX_SIZE),
#     ('leak_right',  INITIAL_GUESS_LEAK_RIGHT_CENTER, LEAK_BOX_SIZE),
#     ('ghost', INITIAL_GUESS_GHOST_CENTER, GHOST_SEARCH_BOX_SIZE),
#     ('glint', INITIAL_GUESS_GLINT_CENTER, GLINT_BOX_SIZE),
#     ('glimmer', INITIAL_GUESS_GLIMMER_CENTER, GLIMMER_BOX_SIZE),
# ]

# # indices in y, x order
# LEAK_LEFT_Y, LEAK_LEFT_X = 287, 250
# PSF_LOCATIONS_VAPP_LEFT = {
#     'left_leak': (LEAK_LEFT_Y, LEAK_LEFT_X),
#     'left_top': (LEAK_LEFT_Y + PSF_OFFSET_Y, LEAK_LEFT_X),
# }

# def locate_rough_psf_centers(data, guesses, mean_sky=None, smooth_stddev_px=PSF_FINDING_SMOOTH_KERNEL_STDDEV_PX):
#     meansub_data = data - mean_sky if mean_sky is not None else data
#     smooth_data = improc.gaussian_smooth(meansub_data, smooth_stddev_px)
#     centers = {}
#     for key, (loc, size) in guesses.items():
#         (y, x), found = rough_peak_in_box(smooth_data, loc, size)
#         centers[key] = (y, x) if found else None
#     return centers

# # Measured from typical frames in 3.9 um
# # right nod (beam == 1)
# _vapp_ur_x, _vapp_ur_y = 775, 407
# _vapp_lr_x, _vapp_lr_y = 901, 148
# _vapp_cr_x, _vapp_cr_y = (_vapp_ur_x + _vapp_lr_x) / 2, (_vapp_ur_y + _vapp_lr_y) / 2
# # left nod (beam == 0)
# _vapp_ul_x, _vapp_ul_y = 775, 407
# _vapp_ll_x, _vapp_ll_y = 901, 148
# _vapp_cl_x, _vapp_cl_y = (_vapp_ul_x + _vapp_ll_x) / 2, (_vapp_ul_y + _vapp_ll_y) / 2


# def _generate_psf_center_guesses(header):
#     nod = header['BEAM'] == 0
#     if header['FILT2'] == 'vAPP A1 ':
#         # vAPP with paired antisymmetric dark holes
#         if nod:
#             initial_x, initial_y = INITIAL_GUESS_LEAK_RIGHT_CENTER
#         else:
#             initial_x, initial_y = INITIAL_GUESS_LEAK_LEFT_CENTER
#     else:
#         # normal point source imaging

# def rough_centers(hdul, mean_sky, smooth_stddev_px=PSF_FINDING_SMOOTH_KERNEL_STDDEV_PX):
#     # generate initial guesses based on observing mode and nod position
#     if vapp:
#         top, leak, and bottom
    

# def locate_rough_psf_centers(
#         data,
#         badpix=None,
#         sky_reference=None,
#         smooth_px=PSF_FINDING_SMOOTH_KERNEL_WIDTH_PX,
#         display=False,
#         ax=None,
#     ):
#     out = {}
#     raw_data = data

#     if badpix is not None:
#         data = mask_bad_pixels(data, badpix, fill_value=0.0)
#     if sky_reference is not None:
#         data -= sky_reference / np.average(sky_reference) * np.nanmean(data)
#     # thresholded = data.copy()
#     # thresholded[data < np.nanmedian(data)] = 0
#     smoothed = smooth(data, smooth_px)
#     # high_pass_data = data - smoothed
#     if display:
#         import matplotlib.pyplot as plt
#         fig, axes = plt.subplots(nrows=len(PSF_LOCATIONS_WHAT_WHERE) + 1, ncols=1, figsize=(8, 4 * len(PSF_LOCATIONS_WHAT_WHERE)))
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(8, 4))
#         ax.imshow(smoothed, vmin=np.nanpercentile(smoothed, 5), vmax=np.nanpercentile(smoothed, 95))
#     for idx, (name, position, size) in enumerate(PSF_LOCATIONS_WHAT_WHERE, start=1):
#         (x, y), railed = centroid_around(
#             smoothed,
#             position,
#             size,
#             display=display,
#             ax=axes[idx] if display else None
#         )
#         out[name+'_psf_x'], out[name+'_psf_y'] = x, y
#         out[name+'_raw_peak'] = raw_data[y, x]
#         if not railed and out[name+'_raw_peak'] > 0:
#             out[name + '_psf_found'] = True
#         else:
#             out[name + '_psf_found'] = False
#         if display:
#             axes[idx].set_title('{} {}'.format(name, position))
#             ax.scatter(
#                 x,
#                 y,
#                 marker='o' if out[name + '_psf_found'] else '*', s=100, label=name,
#                 ec='w',
#             )
#     if display:
#         ax.legend(loc=(1.1, 0.1))
#     # if the top and bottom PSFs aren't separated by the appropriate amount,
#     # the detection is probably bogus:
#     # def _check_sep(top_x, top_y, bottom_x, bottom_y):
#     #     if (top_x - bottom_x) - ()
#     # for side in ('left', 'right'):
#     #     if out[f'{side}_top_psf_found'] and out[f'{side}_bottom_psf_found']:
            
#     for side in ('left', 'right'):
#         # Note that the leakage term is not always visible above the noise
#         found = all(out['{}_{}_psf_found'.format(name, side)] for name in ('top', 'bottom'))
#         if found:
#             if 'top_psf_x' in out:
#                 raise RuntimeError("Unable to decide if this is a left or right nodded image (found both)")
#             out['side_found'] = side
#             for name in ('top', 'leak', 'bottom'):
#                 original_name = name + '_' + side
#                 for key in ('{}_psf_x', '{}_psf_y', '{}_raw_peak', '{}_psf_found'):
#                     out[key.format(name)] = out[key.format(original_name)]
#             # guess locations of any PSFs we didn't find
#             # top
#             if not out['top_psf_found']:
#                 if out['leak_psf_found']:
#                     out['top_psf_x'], out['top_psf_y'] = out['leak_psf_x'] + PSF_OFFSET[0], out['leak_psf_y'] + PSF_OFFSET[1]
#                 elif out['bottom_psf_found']:
#                     out['top_psf_x'], out['top_psf_y'] = out['leak_psf_x'] + 2 * PSF_OFFSET[0], out['leak_psf_y'] + 2 * PSF_OFFSET[1]
#                 else:
#                     if side == 'left':
#                         out['top_psf_x'], out['top_psf_y'] = INITIAL_GUESS_TOP_LEFT_CENTER
#                     elif side == 'right':
#                         out['top_psf_x'], out['top_psf_y'] = INITIAL_GUESS_TOP_RIGHT_CENTER
#                     else:
#                         raise RuntimeError("Unknown side")
#             # leak
#             if not out['leak_psf_found']:
#                 if out['top_psf_found']:
#                     out['leak_psf_x'], out['leak_psf_y'] = out['top_psf_x'] - PSF_OFFSET[0], out['top_psf_y'] - PSF_OFFSET[1]
#                 elif out['bottom_psf_found']:
#                     out['leak_psf_x'], out['leak_psf_y'] = out['bottom_psf_x'] + PSF_OFFSET[0], out['leak_psf_y'] + PSF_OFFSET[1]
#                 else:
#                     if side == 'left':
#                         out['leak_psf_x'], out['leak_psf_y'] = INITIAL_GUESS_LEAK_LEFT_CENTER
#                     elif side == 'right':
#                         out['leak_psf_x'], out['leak_psf_y'] = INITIAL_GUESS_LEAK_RIGHT_CENTER
#                     else:
#                         raise RuntimeError("Unknown side")
#             # bottom
#             if not out['bottom_psf_found']:
#                 if out['leak_psf_found']:
#                     out['bottom_psf_x'], out['bottom_psf_y'] = out['leak_psf_x'] - PSF_OFFSET[0], out['leak_psf_y'] - PSF_OFFSET[1]
#                 elif out['top_psf_found']:
#                     out['bottom_psf_x'], out['bottom_psf_y'] = out['leak_psf_x'] - 2 * PSF_OFFSET[0], out['leak_psf_y'] - 2 * PSF_OFFSET[1]
#                 else:
#                     if side == 'left':
#                         out['bottom_psf_x'], out['bottom_psf_y'] = INITIAL_GUESS_BOTTOM_LEFT_CENTER
#                     elif side == 'right':
#                         out['bottom_psf_x'], out['bottom_psf_y'] = INITIAL_GUESS_BOTTOM_RIGHT_CENTER
#                     else:
#                         raise RuntimeError("Unknown side")

#     if not 'side_found' in out:
#         out['side_found'] = 'neither'
#     return out

# def check_clio_glint_centers(hdr, wavelength_um):
#     '''Ensure Clio's many stray light sources aren't in the way'''
#     exclusion_radius_px = clio2_vapp_lambda_over_d_to_pixel(
#         VAPP_GLINT_FREE_RADIUS_LAMD, wavelength_um * u.um
#     ).value
    
#     # If we can't find the glint, it's probably hidden in the
#     # bright part of the PSF. So, we ensure it's found ('GLINTOK')
#     # and that its location is far from the dark hole
#     dist_to_glint = np.sqrt((hdr['BOTXC'] - hdr['GLINTXC'])**2 + (hdr['BOTYC'] - hdr['GLINTYC'])**2)
#     glint_found_outside = hdr['GLINTOK'] and dist_to_glint > exclusion_radius_px
#     # if glimmer is found, ensure it's outside the dark hole. If not found, don't worry.
#     dist_to_glimmer = np.sqrt((hdr['TOPXC'] - hdr['GLIMXC'])**2 + (hdr['TOPYC'] - hdr['GLIMYC'])**2)
#     glimmer_not_contaminating = (not hdr['GLIMOK']) or (dist_to_glimmer > exclusion_radius_px)
#     frame_ok = glint_found_outside and glimmer_not_contaminating
#     return frame_ok
