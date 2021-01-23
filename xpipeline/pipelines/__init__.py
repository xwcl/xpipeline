import argparse
import glob
import os.path
import logging
from pprint import pformat

from astropy.io import fits
import dask
import dask.array as da
import pandas as pd

from .. import constants as const
from ..core import PipelineCollection
from ..tasks import obs_table, iofits, sky_model, detector, data_quality, learning
from ..instruments import clio

log = logging.getLogger(__name__)

## To preserve sanity, pipelines mustn't do anything you can't do to a dask.delayed
## and mustn't compute or branch on intermediate results.
## They should return delayeds.

def compute_sky_model(
    inputs_collection,
    badpix_arr,
    test_fraction,
    random_state,
    n_components,
    mask_dilate_iters,
):
    sky_cube = (
        inputs_collection.map(iofits.ensure_dq)
        .map(data_quality.set_dq_flag, badpix_arr, const.DQ_BAD_PIXEL)
        .map(clio.correct_linearity)
        .collect(iofits.hdulists_to_dask_cube)
    )
    sky_cube_train, sky_cube_test = learning.train_test_split(
        sky_cube, test_fraction, random_state=random_state
    )
    components, mean_sky, stddev_sky = sky_model.compute_components(
        sky_cube_train, n_components
    )
    min_err, max_err, avg_err = sky_model.cross_validate(
        sky_cube_test, components, stddev_sky, mean_sky, badpix_arr, mask_dilate_iters
    )
    return components, mean_sky, stddev_sky
