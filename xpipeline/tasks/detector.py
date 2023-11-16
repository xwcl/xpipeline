import logging
from multiprocessing import Value
import numpy as np
import dask
from .. import constants as const

log = logging.getLogger(__name__)


def correct_linearity(hdul, coeffs, correctable_domain, ext=0, dq_ext="DQ"):
    log.info(f"Correcting linearity")
    if "XLINRTY" in hdul[ext].header:
        raise ValueError(f"Already linearity corrected the data in this HDUList")
    if "XSATURAT" in hdul[ext].header:
        raise ValueError(f"Already flagged saturated data in this HDUList")
    data = hdul[ext].data
    corrected_domain = [
        sum([coeff * x ** idx for idx, coeff in enumerate(coeffs)])
        for x in correctable_domain
    ]
    log.debug(
        "Mapping correctable domain {} to corrected domain {}".format(
            correctable_domain, corrected_domain
        )
    )
    linearity_correction = np.polynomial.polynomial.Polynomial(
        coef=coeffs, domain=correctable_domain, window=corrected_domain
    )
    result = data.copy().astype(float)
    nonlinear_pixels = data >= correctable_domain[0]
    log.debug(
        "Found {} pixels outside the correctable domain of {}".format(
            np.count_nonzero(nonlinear_pixels), correctable_domain
        )
    )
    in_range = nonlinear_pixels & (data <= correctable_domain[1])
    result[in_range] = linearity_correction(data[in_range])
    dq = hdul[dq_ext].data
    saturated = data > correctable_domain[1]
    dq[saturated] = dq[saturated] | const.DQ_SATURATED
    log.debug(
        "Corrected {}/{} pixels".format(
            np.count_nonzero(in_range), np.count_nonzero(nonlinear_pixels)
        )
    )
    log.debug("Found {} saturated".format(np.count_nonzero(saturated)))

    out_hdul = hdul.copy()
    out_hdul[ext].data = result
    out_hdul[ext].header["XLINARTY"] = True, "Has linearity correction been performed?"
    out_hdul[ext].header["XSATURAT"] = True, "Has linearity correction been performed?"
    out_hdul[dq_ext].data = dq
    return out_hdul


def flag_saturation(hdul, saturation_level, ext=0, dq_ext="DQ"):
    log.info("Flagging saturated pixels")
    if "XSATURAT" in hdul[ext].header:
        raise ValueError(f"Already flagged saturated data in this HDUList")
    data = hdul[ext].data
    dq = hdul[dq_ext].data
    saturated = data > saturation_level
    dq[saturated] = dq[saturated] | const.DQ_SATURATED
    log.debug(
        "Flagged {}/{} pixels".format(
            np.count_nonzero(saturated), np.product(data.shape)
        )
    )
    log.debug("Found {} saturated".format(np.count_nonzero(saturated)))

    out_hdul = hdul.copy()
    out_hdul[ext].header["XSATURAT"] = True, "Has linearity correction been performed?"
    out_hdul[dq_ext].data = dq
    return out_hdul


def subtract_bias(hdul, bias_array, ext=0):
    log.info(f"Subtracting bias level")
    if "XBIAS" in hdul[ext].header:
        raise ValueError(f"Already subtracted bias level from the data in this HDUList")
    result = hdul[ext].data - bias_array

    out_hdul = hdul.copy()
    out_hdul[ext].data = result
    out_hdul[ext].header["XBIAS"] = True, "Has bias subtraction been performed?"
    return out_hdul

def subtract_median(hdul, each_column=False, each_row=False, ext=0):
    if each_column and each_row:
        raise ValueError("The order in which the row- and column-wise median are subtracted is undefined; call this function twice instead")
    how = " by row" if each_row else (" by column" if each_column else "")
    log.info("Subtracting median counts level from pixels" + how)
    data = hdul[ext].data
    out_hdul = hdul.copy()
    if each_column:
        median_by_column = np.median(data, axis=0)
        result = data - median_by_column[np.newaxis, :]
    elif each_row:
        median_by_row = np.median(data, axis=1)
        result = data - median_by_row[:, np.newaxis]
    else:
        result = data - np.median(data)
    out_hdul[ext].data = result
    out_hdul[ext].header.add_history("Subtracted median counts level from pixels" + how)
    return out_hdul



def empirical_badpix(std_img, std_percentile, max_img, max_percentile):
    """Using a cube of 2D images, produce a 2D image of probable
    bad pixels by identifying pixels whose standard deviation through
    the cube is > than the `std_percentile` value of all standard
    deviations and whose max value is > the `max_percentile` value
    of all the pixel maxima.
    """
    return (std_img > np.percentile(std_img, std_percentile)) | (
        max_img > np.percentile(max_img, max_percentile)
    )
