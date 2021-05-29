import datetime
from functools import partial
import dask
import dask.bag as db
import numpy as np
from astropy.io import fits
import astropy.units as u
from dataclasses import dataclass

import dateutil

from ...tasks import iofits, improc, vapp, detector
from ... import constants
from .. import magellan
from ... import utils

import logging

log = logging.getLogger(__name__)

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


def lambda_over_d_to_arcsec(
    lambda_over_d, wavelength, d=magellan.PRIMARY_MIRROR_DIAMETER
):
    unit_lambda_over_d = (wavelength.to(u.m) / d.to(u.m)).si.value * u.radian
    return (lambda_over_d * unit_lambda_over_d).to(u.arcsec)


def arcsec_to_lambda_over_d(arcsec, wavelength, d=magellan.PRIMARY_MIRROR_DIAMETER):
    unit_lambda_over_d = ((wavelength.to(u.m) / d.to(u.m)).si.value * u.radian).to(
        u.arcsec
    )
    lambda_over_d = (arcsec / unit_lambda_over_d).si
    return lambda_over_d


def lambda_over_d_to_pixel(
    lambda_over_d, wavelength, d=magellan.PRIMARY_MIRROR_DIAMETER
):
    arcsec = lambda_over_d_to_arcsec(lambda_over_d, wavelength, d=d)
    return (arcsec / CLIO2_PIXEL_SCALE).to(u.pixel)


def pixel_to_lambda_over_d(px, wavelength, d=magellan.PRIMARY_MIRROR_DIAMETER):
    arcsec = (px * CLIO2_PIXEL_SCALE).to(u.arcsec)
    return arcsec_to_lambda_over_d(arcsec, wavelength, d=d)


def _interpolation_endpoints(all_headers, varying_numeric_kw):
    endpoints = []
    for hdr_1, hdr_2 in zip(all_headers, all_headers[1:]):
        these_endpoints = {}
        for kw in varying_numeric_kw:
            these_endpoints[kw] = hdr_1[kw], hdr_2[kw]
        endpoints.append(these_endpoints)
    # special case last as linear extrapolation
    prev = endpoints[-1]
    last = all_headers[-1]
    last_endpoints = {}
    for kw in varying_numeric_kw:
        kwstart, kwend = prev[kw]
        last_endpoints[kw] = last[kw], last[kw] + (kwend - kwstart)
    endpoints.append(last_endpoints)
    return endpoints


def serial_split_frames_cube(all_hduls, filenames, ext=0):
    # I base everything on DATE. so in the extracted-from-cubes files,
    # the first file is given DATE-OBS = DATE from the cube file
    # itself, and subsequent frames are given DATE-OBS = DATE + n*EXPTIME
    # - jrmales
    # as I recall from when this question came up on the MagAO team, it's
    # the values from the start of the exposure, even though the header is
    # written at the end of the exposure (here "exposure" would mean the
    # whole cube, all the images).
    # - katiem

    all_headers = [hdul[ext].header for hdul in all_hduls]
    (
        non_varying_kw,
        varying_kw,
        varying_dtypes,
    ) = iofits.separate_varying_header_keywords(all_headers)
    varying_numeric_dtypes = list(
        filter(lambda x: np.issubdtype(x[1], np.number), varying_dtypes)
    )
    varying_numeric_kw = set([x[0] for x in varying_numeric_dtypes])
    all_other_kw = non_varying_kw | (varying_kw ^ varying_numeric_kw)
    endpoints = _interpolation_endpoints(all_headers, varying_numeric_kw)
    outputs = []
    for hdu_idx, (hdul, filename) in enumerate(zip(all_hduls, filenames)):
        filename_base = utils.basename(filename).rsplit(".", 1)[0]
        if len(hdul[ext].data.shape) != 3:
            raise ValueError(f"Cannot split {hdul[ext].data.shape} data into frames")
        planes = hdul[ext].data.shape[0]
        # interpolation points correspond to each sample of this cube plus first sample of next
        interps = np.zeros(planes + 1, dtype=varying_numeric_dtypes)
        for dtype in varying_numeric_dtypes:
            kw = dtype[0]
            kind = dtype[1]
            start, end = endpoints[hdu_idx][kw]
            interps[kw] = np.linspace(start, end, num=planes + 1, dtype=kind)
        for i in range(hdul[ext].data.shape[0]):
            outfile = f"{filename_base}_{i:03}.fits"
            start_time = dateutil.parser.parse(hdul[ext].header["DATE"])
            new_hdul = iofits.DaskHDUList(
                [iofits.DaskHDU(hdul[ext].data[i], hdul[ext].header.copy())]
            )
            # example INT card:
            # INT     =                 5000 / Integration time per frame in msec
            current_time = start_time + datetime.timedelta(
                milliseconds=i * hdul[ext].header["INT"]
            )
            new_hdul[0].header["DATE-OBS"] = current_time.isoformat()
            for kw in varying_numeric_kw:
                # construct new card with same comment (if any) and new value
                card = hdul[ext].header.cards[kw]
                new_hdul[0].header[kw] = (interps[kw][i], card.comment)
            new_hdul[0].header["ORIGFILE"] = filename
            new_hdul[0].header["INTRPLTD"] = i != 0, "Are varying numeric header values interpolated?"
            outputs.append(new_hdul)
    return outputs


# Morzinski+ 2015 ApJ -- Appendix B
MORZINSKI_COEFFICIENTS = (112.575, 1.00273, -1.40776e-6, 4.59015e-11)
MORZINSKI_DOMAIN = MORZINSKI_LINEARITY_MIN, MORZINSKI_LINEARITY_MAX = [2.7e4, 4.5e4]
MORZINSKI_RANGE = MORZINSKI_CORRECTED_MIN, MORZINSKI_CORRECTED_MAX = [
    sum([coeff * x ** idx for idx, coeff in enumerate(MORZINSKI_COEFFICIENTS)])
    for x in MORZINSKI_DOMAIN
]


def make_vapp_dark_hole_masks(shape, wavelength):
    return vapp.make_dark_hole_masks(
        shape,
        owa_px=lambda_over_d_to_pixel(VAPP_OWA_LAMD, wavelength).value,
        offset_px=lambda_over_d_to_pixel(VAPP_OFFSET_LAMD, wavelength).value,
        psf_rotation_deg=VAPP_PSF_ROTATION_DEG,
    )


correct_linearity = partial(
    detector.correct_linearity,
    coeffs=MORZINSKI_COEFFICIENTS,
    correctable_domain=MORZINSKI_DOMAIN,
)

# def correct_linearity(data, coeffs=MORZINSKI_COEFFICIENTS, correctable_domain=MORZINSKI_DOMAIN,
#                       verbose=False):
#     corrected_domain = [
#         sum([coeff * x**idx for idx, coeff in enumerate(coeffs)])
#         for x in correctable_domain
#     ]
#     if verbose:
#         log.debug('Mapping correctable domain {} to corrected domain {}'.format(correctable_domain, corrected_domain))
#     linearity_correction = np.polynomial.polynomial.Polynomial(
#         coef=coeffs,
#         domain=correctable_domain,
#         window=corrected_domain
#     )
#     result = data.copy().astype(float)
#     nonlinear_pixels = data >= correctable_domain[0]
#     log.debug("Found {} pixels outside the correctable domain of {}".format(np.count_nonzero(nonlinear_pixels), correctable_domain))
#     in_range = nonlinear_pixels & (data <= correctable_domain[1])
#     result[in_range] = linearity_correction(data[in_range])
#     dq = np.zeros(result.shape, dtype=int)
#     saturated = data > correctable_domain[1]
#     dq[saturated] = constants.DQ_SATURATED
#     if verbose:
#         log.debug("Corrected {}/{} pixels".format(np.count_nonzero(in_range), np.count_nonzero(nonlinear_pixels)))
#         log.debug("Found {} saturated".format(np.count_nonzero(saturated)))
#     return result, dq


@dataclass
class Subarray:
    cutout_origin: tuple[int, int]
    detector_origin: tuple[int, int]
    extent: tuple[int, int]


SUBARRAYS = {
    (512, 1024): [
        Subarray(detector_origin=(0, 0), cutout_origin=(0, 0), extent=(512, 1024))
    ],
    (300, 1024): [
        Subarray(detector_origin=(212, 0), cutout_origin=(0, 0), extent=(300, 1024))
    ],
    (200, 400): [
        Subarray(detector_origin=(312, 0), cutout_origin=(0, 0), extent=(200, 200)),
        Subarray(detector_origin=(312, 512), cutout_origin=(0, 200), extent=(200, 200)),
    ],
    (100, 50): [
        Subarray(detector_origin=(462, 0), cutout_origin=(0, 0), extent=(200, 200)),
        Subarray(detector_origin=(462, 512), cutout_origin=(0, 200), extent=(200, 200)),
    ],
}


def badpix_for_shape(badpix_arr, shape):
    """Given a final readout `shape`, cut the appropriate
    chunks out of `badpix_arr` for that subarray mode
    """
    if shape not in SUBARRAYS:
        raise ValueError(f"Unknown subarray mode: {shape}")
    axis = 1
    chunks = []
    for sub in SUBARRAYS[shape]:
        det_y, det_x = sub.detector_origin
        dy, dx = sub.extent
        chunks.append(badpix_arr[det_y : det_y + dy, det_x : det_x + dx])
    return np.concatenate(chunks, axis=1)
