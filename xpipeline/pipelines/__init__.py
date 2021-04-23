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
from typing import List

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
    sci_arr: da.core.Array,
    rot_arr: da.core.Array,
    region_mask: np.ndarray,
    angle_scale: float,
    angle_offset: float,
    exclude_nearest_n_frames: int,
    k_klip_value: int
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

    derotation_angles = angle_scale * rot_arr + angle_offset
    mtx_x, subset_idxs = improc.unwrap_cube(sci_arr, region_mask)
    log.debug(f'{mtx_x.shape=}')

    subtracted_mtx = starlight_subtraction.klip_cube(
        mtx_x,
        k_klip_value,
        exclude_nearest_n_frames
    )
    outcube = improc.wrap_matrix(subtracted_mtx, sci_arr.shape, subset_idxs)
    log.debug(f'{outcube.shape=}')
    out_image = dask.delayed(improc.quick_derotate)(outcube, derotation_angles)
    log.debug('done assembling')
    return out_image

def evaluate_starlight_subtraction(
    sci_arr: da.core.Array,
    rot_arr: da.core.Array,
    region_mask: np.ndarray,
    specs: List[characterization.CompanionSpec],
    template_psf: np.ndarray,
    angle_scale: float,
    angle_offset: float,
    exclude_nearest_n_frames: int,
    k_klip_value: int,
    aperture_diameter_px: float,
    apertures_to_exclude: int
):
    '''Take an input LazyPipelineCollection and first inject signals at
    one or more locations, then apply `klip_adi`, then
    attempt recovery of the injected signal(s).
    Produces (delayed) sequence of `RecoveredSignal` instances
    giving the SNR at each `k_klip_values` value.

    See `subtract_starlight` for other input parameters.
    '''
    injected_sci_arr = characterization.inject_signals(
        sci_arr,
        rot_arr,
        specs,
        template_psf
    )
    out_image = klip_adi(
        injected_sci_arr,
        rot_arr,
        region_mask,
        angle_scale,
        angle_offset,
        exclude_nearest_n_frames,
        k_klip_value,
    )
    recovered_signals = dask.delayed(characterization.recover_signals)(
        out_image,
        specs,
        aperture_diameter_px,
        apertures_to_exclude
    )
    return recovered_signals
