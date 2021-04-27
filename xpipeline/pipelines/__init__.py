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


def adi(
    cube: da.core.Array,
    derotation_angles: da.core.Array,
    operation='sum'
):
    derotated_cube = improc.derotate_cube(cube, derotation_angles)
    if operation == 'average':
        out_image = da.nanmean(derotated_cube, axis=0)
    elif operation == 'sum':
        out_image = da.nansum(derotated_cube, axis=0)
    else:
        raise ValueError("Supported operations: average, sum")
    return out_image

def klip(
    sci_arr: da.core.Array,
    estimation_mask: np.ndarray,
    combination_mask: np.ndarray,
    exclude_nearest_n_frames: int,
    k_klip_value: int
):
    log.debug('Assembling pipeline...')
    mtx_x, subset_idxs = improc.unwrap_cube(sci_arr, estimation_mask)
    log.debug(f'{mtx_x.shape=}')

    subtracted_mtx = starlight_subtraction.klip_mtx(
        mtx_x,
        k_klip_value,
        exclude_nearest_n_frames
    )
    outcube = improc.wrap_matrix(subtracted_mtx, sci_arr.shape, subset_idxs)
    # TODO apply combination_mask
    log.debug(f'{outcube.shape=}')
    log.debug('done assembling')
    return outcube

def evaluate_starlight_subtraction(
    sci_arr: da.core.Array,
    derotation_angles: da.core.Array,
    estimation_mask: np.ndarray,
    combination_mask: np.ndarray,
    specs: List[characterization.CompanionSpec],
    template_psf: np.ndarray,
    exclude_nearest_n_frames: int,
    k_klip_value: int,
    aperture_diameter_px: float,
    apertures_to_exclude: int
):
    injected_sci_arr = characterization.inject_signals(
        sci_arr,
        derotation_angles,
        specs,
        template_psf
    )
    outcube = klip(
        injected_sci_arr,
        estimation_mask,
        combination_mask,
        exclude_nearest_n_frames,
        k_klip_value,
    )
    out_image = adi(
        outcube,
        derotation_angles
    )

    recovered_signals = dask.delayed(characterization.recover_signals)(
        out_image,
        specs,
        aperture_diameter_px,
        apertures_to_exclude
    )
    return recovered_signals
