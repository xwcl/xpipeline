import logging
import dask
import numpy as np

from .. import constants as const

log = logging.getLogger(__name__)


def set_dq_flag(image_hdul, mask, flag, ext=0, dq_ext="DQ"):
    mask = mask.astype(np.uint8)
    dq = image_hdul[dq_ext].data
    image_hdul[dq_ext].data = dq | (mask * flag)
    return image_hdul


def get_masked_data(image_hdul, permitted_flags=0, ext=0, dq_ext="DQ", fill=np.nan, excluded_pixels_mask=None):
    data = image_hdul[ext].data
    if dq_ext in image_hdul:
        dq = image_hdul[dq_ext].data
        dq = dq & ~permitted_flags
        mask = dq == 0
        data[~mask] = fill
    if excluded_pixels_mask is not None and np.any(excluded_pixels_mask):
        data[excluded_pixels_mask] = fill
    return data
