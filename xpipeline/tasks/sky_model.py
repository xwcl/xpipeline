import logging
from dataclasses import dataclass
from typing import List, Union

import dask
import dask.array as da
import distributed.protocol
import numpy as np
import scipy.ndimage

from . import iofits, improc
from ..utils import unwrap

log = logging.getLogger(__name__)


@dataclass
class SkyModel:
    components: np.ndarray
    mean_sky: np.ndarray
    stddev_sky: np.ndarray
    min_err: float
    max_err: float
    avg_err: float

    @classmethod
    def from_hdulist(cls, hdulist):
        cls(
            components=hdulist["COMPONENTS"].data,
            mean_sky=hdulist["MEAN_SKY"].data,
            stddev_sky=hdulist["STDDEV_SKY"].data,
            min_err=hdulist["COMPONENTS"].header["MIN_ERR"],
            max_err=hdulist["COMPONENTS"].header["MAX_ERR"],
            avg_err=hdulist["COMPONENTS"].header["AVG_ERR"],
        )

    def to_dask_hdulist(self):
        components, mean_sky, stddev_sky, min_err, max_err, avg_err = dask.compute(
            self.components,
            self.mean_sky,
            self.stddev_sky,
            self.min_err,
            self.max_err,
            self.avg_err,
        )
        components_hdu = iofits.DaskHDU(components)
        components_hdu.header["EXTNAME"] = "COMPONENTS"
        components_hdu.header["MIN_ERR"] = min_err
        components_hdu.header["MAX_ERR"] = max_err
        components_hdu.header["AVG_ERR"] = avg_err
        mean_sky_hdu = iofits.DaskHDU(mean_sky)
        mean_sky_hdu.header["EXTNAME"] = "MEAN_SKY"
        stddev_sky_hdu = iofits.DaskHDU(stddev_sky)
        stddev_sky_hdu.header["EXTNAME"] = "STDDEV_SKY"
        return iofits.DaskHDUList([components_hdu, mean_sky_hdu, stddev_sky_hdu])

    def to_fits_hdulist(self):
        return self.to_dask_hdulist().to_fits()


distributed.protocol.register_generic(SkyModel)


def compute_components(sky_cube, n_components):
    log.debug(f"{sky_cube=}")
    sky_cube[da.isnan(sky_cube)] = 0.0
    mean_sky_image = da.mean(sky_cube, axis=0)
    stddev_sky_image = da.std(sky_cube, axis=0)
    planes, rows, cols = sky_cube.shape
    all_real_mtx = (
        (sky_cube - mean_sky_image).reshape((planes, rows * cols)).T
    )  # now cube is rows*cols x planes
    log.info(f"requesting SVD of {all_real_mtx.shape} matrix")
    mtx_u, _, _ = da.linalg.svd_compressed(
        all_real_mtx, k=n_components
    )  # mtx_u is rows*cols x n_components

    if n_components > mtx_u.shape[1]:
        raise ValueError(
            f"Couldn't compute {n_components} components from {sky_cube.shape} cube"
        )
    components_cube = mtx_u.T.reshape((n_components, rows, cols))

    return components_cube, mean_sky_image, stddev_sky_image


def reconstruct_masked(original_image, components_cube, model_mean, bad_bg_mask):
    """Reconstruct the masked regions of `original_image` with images from
    `components_cube` (which should be orthogonal eigenimages)
    mask pixels with True are excluded (either by replacing
    with `fill` or ignored in least-squares fit). fit uses only "~mask" pixels
    """
    planes, ny, nx = components_cube.shape

    mean_imvec = model_mean.reshape(ny * nx)
    meansub_imvec = original_image.reshape(ny * nx) - mean_imvec
    component_imvecs = components_cube.reshape((planes, ny * nx))
    mask_1d = (~bad_bg_mask).reshape(ny * nx)

    log.info(f"{component_imvecs.shape=}")
    log.info(f"{meansub_imvec.shape=}")
    a = component_imvecs[:, mask_1d]
    b = meansub_imvec[mask_1d]
    assert a.shape[1] == b.shape[0]
    x, residuals, rank, s = np.linalg.lstsq(
        a.T,
        b,
        # rcond=None
    )
    reconstruction = (np.dot(component_imvecs.T, x) + mean_imvec).reshape(
        original_image.shape
    )
    return reconstruction


def generate_background_mask(
    std_bg_arr: np.ndarray,
    mean_bg_arr: np.ndarray,
    badpix_arr: np.ndarray,
    iterations: int,
    n_sigma: float = None,
    science_arr: np.ndarray = None,
    exclude: Union[List[improc.BBox], None] = None,
):
    """Detects pixels that are not reliable for background level sensing
    using heuristics based on background pixel statistics, bad pixel mask,
    and science frame intensity (if given)

    Returns
    -------
    bad_bg_pix_mask : 2D boolean array
        True pixels are bad for background estimation
    """
    percentile_threshold = np.percentile(std_bg_arr, 90)
    median_bg_std = np.median(std_bg_arr)
    bad_bg_pix_mask = badpix_arr != 0
    bad_bg_pix_mask = (badpix_arr != 0) | bad_bg_pix_mask
    bad_bg_pix_mask = (std_bg_arr > percentile_threshold) | bad_bg_pix_mask
    if science_arr is not None:
        if n_sigma is None:
            raise ValueError("When science_arr is supplied, n_sigma must be too")
        bad_bg_pix_mask = (
            science_arr > n_sigma * median_bg_std + mean_bg_arr
        ) | bad_bg_pix_mask
    bad_bg_pix_mask = scipy.ndimage.binary_dilation(
        bad_bg_pix_mask, iterations=iterations
    )
    if exclude is not None:
        for bbox in exclude:
            yslice, xslice = bbox.slices
            bad_bg_pix_mask[yslice, xslice] = True
    assert np.count_nonzero(~bad_bg_pix_mask) >= 0.05 * std_bg_arr.size
    return bad_bg_pix_mask


def cross_validate(
    sky_cube_test, components_cube, std_bg_arr, mean_bg_arr, badpix_arr, iterations
):
    errs = []
    for idx, sky_img in enumerate(sky_cube_test):
        bad_bg_pix = generate_background_mask(
            std_bg_arr, mean_bg_arr, badpix_arr, iterations
        )
        bg_estimate = reconstruct_masked(
            sky_img, components_cube, mean_bg_arr, bad_bg_pix
        )
        recons_vals_std = np.std(bg_estimate - sky_img)
        errs.append(recons_vals_std)
    min_err, max_err, avg_err = np.min(errs), np.max(errs), np.average(errs)
    log.info(f"STD: {min_err=}, {max_err=}, {avg_err=}")
    return min_err, max_err, avg_err


def background_subtract(
    hdul: iofits.DaskHDUList,
    sky_model: SkyModel,
    badpix_arr: np.ndarray,
    iterations: int,
    n_sigma: float,
    exclude: Union[List[improc.BBox], None],
    ext=0,
):
    log.info(f"Subtracting sky background from {hdul[ext].data}")
    sci_arr = hdul[ext].data
    bad_bg_pix = generate_background_mask(
        sky_model.stddev_sky,
        sky_model.mean_sky,
        badpix_arr,
        iterations,
        n_sigma=n_sigma,
        science_arr=sci_arr,
        exclude=exclude,
    )
    bg_estimate = reconstruct_masked(
        sci_arr, sky_model.components, sky_model.mean_sky, bad_bg_pix
    )
    sci_final = sci_arr - bg_estimate
    log.debug(
        f"Mean in background measurement pixels: {np.average(sci_final[~bad_bg_pix])}"
    )

    n_bad_bg_pix = np.count_nonzero(bad_bg_pix)
    n_pix_total = sci_arr.size
    recons_vals_std = np.std(bg_estimate[~bad_bg_pix] - sci_arr[~bad_bg_pix])

    # TODO: iterate mask shape until convergence?

    msg = unwrap(
        f"""
        Reconstructed sky background using {sky_model.components.shape[0]} image
        basis. Background mask used threshold  {n_sigma} * std(background),
        repeated dilation for {iterations} iterations, leaving 
        {n_pix_total - n_bad_bg_pix} / {n_pix_total}
        ({100 * (1 - n_bad_bg_pix/n_pix_total):2.1f}%)
        for estimating the background. RMS error in background pixels
        was {recons_vals_std}.
    """
    )
    out_hdul = hdul.updated_copy(
        sci_final,
        {
            "BGRMS": (recons_vals_std, "RMS error in background sensing pixels"),
            "XBGSUB": (True, "Background subtraction complete?"),
        },
        history=msg,
        ext=ext,
    )
    log.info(msg)
    return out_hdul
