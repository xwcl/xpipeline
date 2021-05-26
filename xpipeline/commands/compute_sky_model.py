from astropy.io import fits
import numpy as np
import argparse
import dask
import os.path
import logging

# from . import constants as const
from ..utils import unwrap
from .. import utils
from .. import pipelines  # , irods
from ..core import LazyPipelineCollection
from ..tasks import iofits  # obs_table, iofits, sky_model, detector, data_quality

# from .ref import clio

from .base import MultiInputCommand

log = logging.getLogger(__name__)


class ComputeSkyModel(MultiInputCommand):
    name = "compute_sky_model"
    help = "Compute sky model eigenimages"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--sky-n-components",
            type=int,
            default=6,
            help="Number of PCA components to calculate, default 6",
        )
        parser.add_argument(
            "--mask-dilate-iters",
            type=int,
            default=4,
            help=unwrap(
                """
                Number of times to grow mask regions before selecting
                cross-validation pixels (default: 4)
            """
            ),
        )
        parser.add_argument(
            "--test-fraction",
            type=float,
            default=0.25,
            help="Fraction of inputs to reserve for cross-validation",
        )
        return super(ComputeSkyModel, ComputeSkyModel).add_arguments(parser)

    def main(self):
        destination = self.destination

        # inputs
        all_files = self.all_files
        n_components = self.args.sky_n_components
        mask_dilate_iters = self.args.mask_dilate_iters
        test_fraction = self.args.test_fraction
        random_state = self.args.random_state

        # outputs
        model_fn = os.path.join(destination, "sky_model.fits")
        if self.check_for_outputs([model_fn]):
            return

        # execute
        inputs_coll = LazyPipelineCollection(all_files).map(iofits.load_fits_from_path)

        one_input_hdul = dask.compute(inputs_coll.items[0])[0]
        plane_shape = one_input_hdul[0].data.shape

        d_sky_model = pipelines.compute_sky_model(
            inputs_coll,
            plane_shape,
            test_fraction,
            random_state,
            n_components,
            mask_dilate_iters,
        )
        the_sky_model = dask.compute(d_sky_model)[0]
        hdul = the_sky_model.to_hdulist()
        hdul.writeto(model_fn, overwrite=True)
        log.info(f"Sky model written to {model_fn}")
