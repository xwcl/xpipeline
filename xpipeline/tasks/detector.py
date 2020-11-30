import logging
import dask
import numpy as np

from .. import constants as const

log = logging.getLogger(__name__)

@dask.delayed
def set_dq_flag(image_hdul, mask, flag, ext=0, dq_ext='DQ'):
    mask = mask.astype(np.uint8)
    dq = image_hdul[dq_ext].data
    image_hdul[dq_ext].data = dq | (mask * flag)
    return image_hdul

@dask.delayed
def get_masked_data(image_hdul, permitted_flags=0, ext=0, dq_ext='DQ', fill=np.nan):
    data = image_hdul[ext].data
    dq = image_hdul[dq_ext].data
    dq = dq & ~permitted_flags
    mask = dq == 0
    data[mask] = fill
    return data

def correct_linearity(hdul, coeffs, correctable_domain, ext=0):
    corrected_domain = [
        sum([coeff * x**idx for idx, coeff in enumerate(coeffs)])
        for x in correctable_domain
    ]
    # log.debug('Mapping correctable domain {} to corrected domain {}'.format(correctable_domain, corrected_domain))
    linearity_correction = np.polynomial.polynomial.Polynomial(
        coef=coeffs,
        domain=correctable_domain,
        window=corrected_domain
    )
    result = data.copy().astype(float)
    nonlinear_pixels = data >= correctable_domain[0]
    log.debug("Found {} pixels outside the correctable domain of {}".format(np.count_nonzero(nonlinear_pixels), correctable_domain))
    in_range = nonlinear_pixels & (data <= correctable_domain[1])
    result[in_range] = linearity_correction(data[in_range])
    dq = np.zeros(result.shape, dtype=int)
    saturated = data > correctable_domain[1]
    dq[saturated] = dq[saturated] | const.DQ_SATURATED
    if verbose:
        debug("Corrected {}/{} pixels".format(np.count_nonzero(in_range), np.count_nonzero(nonlinear_pixels)))
        debug("Found {} saturated".format(np.count_nonzero(saturated)))
    return result, dq
