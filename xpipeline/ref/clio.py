import dask
import dask.bag as db
import numpy as np
from astropy.io import fits
import astropy.units as u

from ..tasks import iofits, improc, vapp
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

def make_vapp_dark_hole_masks(shape, wavelength):
    return vapp.make_dark_hole_masks(shape,
        owa_px=lambda_over_d_to_pixel(VAPP_OWA_LAMD, wavelength).value,
        offset_px=lambda_over_d_to_pixel(VAPP_OFFSET_LAMD, wavelength).value,
        psf_rotation_deg=VAPP_PSF_ROTATION_DEG
    )
