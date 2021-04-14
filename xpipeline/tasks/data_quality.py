import logging
import dask
import numpy as np

from .. import constants as const

log = logging.getLogger(__name__)

def set_dq_flag(image_hdul, mask, flag, ext=0, dq_ext='DQ'):
    mask = mask.astype(np.uint8)
    dq = image_hdul[dq_ext].data
    image_hdul[dq_ext].data = dq | (mask * flag)
    return image_hdul

def get_masked_data(image_hdul, permitted_flags=0, ext=0, dq_ext='DQ', fill=np.nan):
    data = image_hdul[ext].data
    dq = image_hdul[dq_ext].data
    dq = dq & ~permitted_flags
    mask = dq == 0
    data[mask] = fill
    return data
