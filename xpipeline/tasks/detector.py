import logging
import numpy as np
import dask
from .. import constants as const

log = logging.getLogger(__name__)

def correct_linearity(hdul, coeffs, correctable_domain, ext=0, dq_ext='DQ'):
    log.info(f'Correcting linearity')
    data = hdul[ext].data
    corrected_domain = [
        sum([coeff * x**idx for idx, coeff in enumerate(coeffs)])
        for x in correctable_domain
    ]
    log.debug('Mapping correctable domain {} to corrected domain {}'.format(correctable_domain, corrected_domain))
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
    dq = hdul[dq_ext].data
    saturated = data > correctable_domain[1]
    dq[saturated] = dq[saturated] | const.DQ_SATURATED
    log.debug("Corrected {}/{} pixels".format(np.count_nonzero(in_range), np.count_nonzero(nonlinear_pixels)))
    log.debug("Found {} saturated".format(np.count_nonzero(saturated)))
    
    out_hdul = hdul.copy()
    out_hdul[ext].data = data
    out_hdul[dq_ext].data = dq
    return out_hdul
