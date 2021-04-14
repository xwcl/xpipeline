from astropy.io import fits
import numpy as np
import argparse
import dask
import os.path
import logging
# from . import constants as const
from ..utils import unwrap
from .. import utils
from .. import pipelines #, irods
from ..core import LazyPipelineCollection
from ..tasks import iofits # obs_table, iofits, sky_model, detector, data_quality
# from .ref import clio

from .base import BaseCommand

log = logging.getLogger(__name__)


class ComputeSkyModel(BaseCommand):
    name = "compute_sky_model"
    help = "Compute sky model eigenimages"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "badpix", help="Path to FITS image of bad pixel map (1 is bad, 0 is good)"
        )
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
            help=unwrap('''
                Number of times to grow mask regions before selecting
                cross-validation pixels (default: 4)
            '''),
        )
        # parser.add_argument(
        #     "--mask-n-sigma",
        #     type=float,
        #     default=2,
        #     help=unwrap('''
        #         Pixels excluded if mean science image (after mean background subtraction)
        #         has value p[y,x] > N * sigma[y,x] (from the sky standard deviation image)",
        #     ''')
        # )
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
        badpix_path = self.args.badpix
        mask_dilate_iters = self.args.mask_dilate_iters
        test_fraction = self.args.test_fraction
        random_state = self.args.random_state

        # outputs
        components_fn = os.path.join(destination, "sky_model_components.fits")
        mean_fn = os.path.join(destination, "sky_model_mean.fits")
        stddev_fn = os.path.join(destination, "sky_model_std.fits")
        output_files = [components_fn, mean_fn, stddev_fn]
        if self.check_for_outputs(output_files):
            return

        # execute
        badpix_arr = dask.delayed(iofits.load_fits_from_path)(badpix_path)[0].data.persist()
        inputs_coll = LazyPipelineCollection(all_files).map(iofits.load_fits_from_path)
        # coll = LazyPipelineCollection(all_files)
        # sky_cube = (
        #     coll.map(iofits.load_fits)
        #     .map(iofits.ensure_dq)
        #     .map(data_quality.set_dq_flag, badpix_arr, const.DQ_BAD_PIXEL)
        #     .map(clio.correct_linearity)
        #     .collect(iofits.hdulists_to_dask_cube)
        # ).persist()
        # sky_cube_train, sky_cube_test = learning.train_test_split(
        #     sky_cube, test_fraction, random_state=random_state
        # )
        # components, mean_sky, stddev_sky = sky_model.compute_components(
        #     sky_cube_train, n_components
        # ).persist()
        # min_err, max_err, avg_err = sky_model.cross_validate(
        #     sky_cube_test, components, stddev_sky, mean_sky, badpix_arr, mask_dilate_iters
        # ).compute()
        # log.info(f"Cross-validation reserved {100 * test_fraction:2.1f} of inputs")
        # log.info(f"STD: {min_err=}, {max_err=}, {avg_err=}")
        results = pipelines.compute_sky_model(
            inputs_coll,
            badpix_arr,
            test_fraction,
            random_state,
            n_components,
            mask_dilate_iters,
        )
        dask.persist(results)
        (
            components, mean_sky, stddev_sky,
            min_err, max_err, avg_err
        ) = results
        # save
        fits.PrimaryHDU(
            np.asarray(components.compute())
        ).writeto(components_fn, overwrite=True)
        log.info(f"{n_components} components written to {components_fn}")
        fits.PrimaryHDU(np.asarray(mean_sky.compute())).writeto(mean_fn, overwrite=True)
        log.info(f"Mean sky written to {mean_fn}")
        fits.PrimaryHDU(np.asarray(stddev_sky.compute())).writeto(stddev_fn, overwrite=True)
        log.info(f"Stddev sky written to {stddev_fn}")
