import argparse
import glob
from multiprocessing import Pipe
import os.path
import logging
from pprint import pformat
import numpy as np
from astropy.io import fits
import dask
import dask.array as da
import pandas as pd

from .. import constants as const
from ..core import LazyPipelineCollection
from ..tasks import (
    obs_table, iofits, sky_model, detector, data_quality, learning,
    improc,
    starlight_subtraction,
    characterization,
)
from ..ref import clio

log = logging.getLogger(__name__)

## To preserve sanity, pipelines mustn't do anything you can't do to a dask.delayed
## and mustn't compute or branch on intermediate results.
## They should return delayeds (for composition).

def compute_sky_model(
    inputs_collection,
    badpix_arr,
    test_fraction,
    random_state,
    n_components,
    mask_dilate_iters,
):
    log.debug('Assembling pipeline...')
    sky_cube = (
        inputs_collection.map(iofits.ensure_dq)
        .map(data_quality.set_dq_flag, badpix_arr, const.DQ_BAD_PIXEL)
        .map(clio.correct_linearity)
        .collect(iofits.hdulists_to_dask_cube)
    )
    sky_cube_train, sky_cube_test = dask.delayed(learning.train_test_split, nout=2)(
        sky_cube, test_fraction, random_state=random_state
    )
    components, mean_sky, stddev_sky = dask.delayed(sky_model.compute_components, nout=3)(
        sky_cube_train, n_components
    )
    min_err, max_err, avg_err = dask.delayed(sky_model.cross_validate, nout=3)(
        sky_cube_test, components, stddev_sky, mean_sky, badpix_arr, mask_dilate_iters
    )
    log.debug('done assembling')
    return components, mean_sky, stddev_sky, min_err, max_err, avg_err


def klip_adi(
    sci_arr,
    rot_arr,
    region_mask: np.ndarray,
    rotation_keyword: str,
    rotation_scale: float,
    rotation_offset: float,
    # rotation_exclusion_frames: int,
    exclude_nearest_n_frames: int,
    k_klip_value: int,
    plane_shape: tuple=None,
):
    '''Perform KLIP starlight subtraction on 2D ADI data

    Parameters
    ----------
    sorted_inputs_collection : LazyPipelineCollection
        inputs already sorted by date such that
        adjacent frames are adjacent in time (single epoch)
    region_mask : ndarray
        mask shaped like one input image that is True where
        pixels should be kept and False elsewhere
    '''
    log.debug('Assembling pipeline...')

    derotation_angles = rotation_scale * rot_arr + rotation_offset
    mtx_x, x_indices, y_indices = improc.unwrap_cube(sci_arr, region_mask)
    log.debug(f'{mtx_x.shape=}')

    subtracted_mtx = starlight_subtraction.klip_cube(
        mtx_x,
        k_klip_value,
        exclude_nearest_n_frames
    )
    outcube = improc.wrap_matrix(subtracted_mtx, sci_arr.shape, x_indices, y_indices)
    log.debug(f'{outcube.shape=}')
    out_image = dask.delayed(improc.quick_derotate)(outcube, derotation_angles)
    log.debug('done assembling')
    return out_image

def evaluate_starlight_subtraction(
    inputs_collection,
    date_keyword,
    rotation_keyword,
    rotation_scale,
    rotation_offset,
    rotation_exclusion_deg,
    rotation_exclusion_frames,
    k_klip_values,
    signal_injection_array,
    target_signal_spec,
    other_signal_specs,
    matched_filter,
):
    '''Take an input LazyPipelineCollection and first inject signals at
    one or more locations, then apply `klip_adi`, then
    attempt recovery of the injected signal(s).
    Produces (delayed) sequence of `RecoveredSignal` instances
    giving the SNR at each `k_klip_values` value.

    See `subtract_starlight` for other input parameters.
    '''
    pass
