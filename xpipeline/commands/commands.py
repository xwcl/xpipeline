import argparse
import glob
import os.path
import logging

# import warnings
# warnings.simplefilter('error')
from pprint import pformat

from astropy.io import fits
import dask
import dask.array as da
from dask.distributed import Client
import pandas as pd
import numpy as np

from . import constants as const
from .utils import unwrap
from . import utils
from . import pipelines, irods
from .core import LazyPipelineCollection
from .tasks import obs_table, iofits, sky_model, detector, data_quality
from .ref import clio

log = logging.getLogger(__name__)


def _generate_output_filenames(all_files, destination):
    return [os.path.join(destination, os.path.basename(x)) for x in all_files]


def ingest():
    parser = argparse.ArgumentParser()
    # add extra arguments
    parser.add_argument("--obs-date-key", default="DATE-OBS")
    parser.add_argument(
        "--name-prefix",
        default="frame",
        help='Prefix for output filenames (default: "frame" -> "frame_00000.fits")',
    )
    # parse and generate list of `all_files`
    args = _final_args_and_parse(parser)

    destination = args.destination
    all_files = args.all_files
    observation_date_key = args.obs_date_key
    log.info("Processing: %s", pformat(all_files))
    d_names_to_hdulists = {fn: iofits.load_fits(fn) for fn in all_files}
    table_path = os.path.join(destination, "obs.csv")
    if os.path.exists(table_path):
        log.info(
            "Found existing 'obs.csv' table at %s, remove to regenerate", table_path
        )
        table = pd.read_csv(table_path)
    else:
        (table,) = dask.compute(
            obs_table.construct_observations_table(
                d_names_to_hdulists, observation_date_key=observation_date_key
            )
        )
        output_files = [
            os.path.join(destination, f"frame_{i:06}.fits")
            for i in range(len(all_files))
        ]
        table["output_name"] = output_files
        table.to_csv(table_path)

    def normalize_one(row):
        if os.path.exists(row["output_name"]):
            log.info(f'Found existing {row["output_name"]} for {row["original_name"]}')
            return row["output_name"]
        orig_fn = row["original_name"]
        fn = row["output_name"]
        log.debug(d_names_to_hdulists[orig_fn])
        log.debug(type(d_names_to_hdulists[orig_fn]))
        return dask.compute(iofits.write_fits_to_disk(d_names_to_hdulists[orig_fn], fn))

    result = dask.compute(*table.apply(normalize_one, axis="columns"))
    log.info(result)


def local_to_irods():
    parser = argparse.ArgumentParser()
    # _ = Client()
    # dask.config.set(scheduler='single-threaded')
    # parse and generate list of `all_files`
    args = _final_args_and_parse(parser)
    destination = args.destination
    irods.ensure_collection(destination)
    all_files = args.all_files
    output_files = _generate_output_filenames(args.all_files, args.destination)
    # compute
    inputs_coll = LazyPipelineCollection(all_files)
    destination_paths = (
        inputs_coll.map(iofits.load_fits)
        .zip_map(iofits.write_fits_to_irods, output_files, overwrite=True)
        .end()
    )
    destination_paths = dask.compute(destination_paths)
    log.info(f"Uploaded files to {destination} using iRODS")
    log.info(destination_paths)


def compute_sky_model():
    parser = argparse.ArgumentParser()
    # add extra arguments
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
        default=1,
        help="Number of times to grow mask regions to improve background estimates",
    )
    parser.add_argument(
        "--mask-n-sigma",
        type=float,
        default=2,
        help=unwrap(
            """
            Pixels excluded if mean science image (after mean background subtraction)
            has value p[y,x] > N * sigma[y,x] (from the sky standard deviation image)",
        """
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Fraction of inputs to reserve for cross-validation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed state for reproducibility (default: 0)",
    )
    # parse and generate list of `all_files`
    args = _final_args_and_parse(parser)
    destination = args.destination
    os.makedirs(destination, exist_ok=True)
    all_files = args.all_files
    n_components = args.sky_n_components
    badpix_path = args.badpix
    mask_dilate_iters = args.mask_dilate_iters
    test_fraction = args.test_fraction
    random_state = args.random_state
    # outputs
    components_fn = os.path.join(destination, f"sky_model_components.fits")
    mean_fn = os.path.join(destination, f"sky_model_mean.fits")
    stddev_fn = os.path.join(destination, f"sky_model_std.fits")
    # output_files = [components_fn, mean_fn, stddev_fn]
    # if all(map(os.path.exists, output_files)):
    #     log.info(f'All outputs exist: {output_files}')
    #     log.info('Remove them to re-run')
    #     return
    # execute
    # client = Client()
    badpix_arr = iofits.load_fits(badpix_path)[0].data.persist()
    inputs_coll = LazyPipelineCollection(all_files).map(iofits.load_fits)
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
    components, mean_sky, stddev_sky = pipelines.compute_sky_model(
        inputs_coll,
        badpix_arr,
        test_fraction,
        random_state,
        n_components,
        mask_dilate_iters,
    )
    dask.persist([components, mean_sky, stddev_sky])
    # save
    fits.PrimaryHDU(np.asarray(components.compute())).writeto(
        components_fn, overwrite=True
    )
    log.info(f"{n_components} components written to {components_fn}")
    fits.PrimaryHDU(np.asarray(mean_sky.compute())).writeto(mean_fn, overwrite=True)
    log.info(f"Mean sky written to {mean_fn}")
    fits.PrimaryHDU(np.asarray(stddev_sky.compute())).writeto(stddev_fn, overwrite=True)
    log.info(f"Stddev sky written to {stddev_fn}")


def clio_instrument_calibrate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "badpix", help="Path to FITS image of bad pixel map (1 is bad, 0 is good)"
    )
    parser.add_argument("sky_components", help="Path to FITS cube of sky eigenimages")
    parser.add_argument("sky_mean", help="Mean sky image")
    parser.add_argument("sky_std", help="Standard deviation sky image")
    parser.add_argument(
        "--sky-n-components",
        type=int,
        default=None,
        help="Number of PCA components to use in sky estimate (default: all)",
    )
    parser.add_argument(
        "--mask-dilate-iters",
        type=int,
        default=5,
        help="Number of times to grow mask regions to improve background estimates",
    )
    parser.add_argument(
        "--mask-n-sigma",
        type=float,
        default=1.5,
        help=unwrap(
            """
            Pixels excluded if mean science image (after mean background subtraction) has
            value p[y,x] > N * sigma[y,x] (from the sky standard deviation image)
        """
        ),
    )
    # parser.add_argument('psf_model', help='FITS image with unsaturated model PSF ("bottom" PSF for vAPP)')
    # parse and generate list of `all_files`
    args = _final_args_and_parse(parser)
    destination = args.destination
    os.makedirs(destination, exist_ok=True)
    sky_n_components = args.sky_n_components
    mask_dilate_iters = args.mask_dilate_iters
    mask_n_sigma = args.mask_n_sigma

    output_files = _generate_output_filenames(args.all_files, args.destination)
    if all(map(os.path.exists, output_files)):
        log.info(f"All outputs exist: {output_files}")
        log.info("Remove them to re-run")
        return
    badpix_arr = iofits.load_fits(args.badpix)[0].data
    sky_components_arr = iofits.get_data_from_disk(args.sky_components)[
        :sky_n_components
    ]
    coll = LazyPipelineCollection(args.all_files)
    coll_prelim = (
        coll.map(iofits.load_fits)
        .map(iofits.ensure_dq)
        .map(data_quality.set_dq_flag, badpix_arr, const.DQ_BAD_PIXEL)
        .map(
            detector.correct_linearity,
            clio.MORZINSKI_COEFFICIENTS,
            clio.MORZINSKI_DOMAIN,
        )
    )
    # Make sky background estimation mask
    delayed_prelim_data = coll_prelim.collect(iofits.hdulists_to_dask_cube)
    da_prelim = delayed_prelim_data.compute()
    # mean_star_arr = da.average(da_prelim, axis=0)
    mean_bg_arr = iofits.get_data_from_disk(args.sky_mean)
    std_bg_arr = iofits.get_data_from_disk(args.sky_std)
    # bg_goodpix = sky_model.generate_background_mask(
    #     mean_star_arr,
    #     mean_bg_arr,
    #     std_bg_arr,
    #     badpix_arr,
    #     mask_dilate_iters,
    #     mask_n_sigma
    # )
    # Background subtraction and centering
    d_results = (
        coll_prelim.map(
            sky_model.background_subtract,
            mean_bg_arr,
            std_bg_arr,
            sky_components_arr,
            badpix_arr,
            mask_dilate_iters,
            mask_n_sigma,
        )
        # .map(clio.rough_centers, mean_sky)
        # .map(clio.refine_centers, sky_model)
        # .map(clio.aligned_cutout)
        .zip_map(iofits.write_fits_to_disk, output_files).end()
    )
    # results = dask.compute(*d_results)
    results = dask.compute(d_results)
    log.info(results)
    # databag = dask.bag.from_sequence(args.all_files)
    # output_files_bag = dask.bag.from_sequence(output_files)
    # outputs_delayed = (databag
    #     .map(iofits.load_fits)
    #     # .map(iofits.ensure_dq)
    #     # .map(badpix.apply_mask, badpix_mask)
    #     # .map(clio.correct_linearity)
    #     # .map(iofits.write_fits, output_files_bag)
    #     .map(lambda x: x[0].data)
    # )
    # # writes = [
    # #     iofits.write_fits(fn, dhdul)
    # #     for fn, dhdul
    # #     in zip(output_files, outputs_delayed)
    # # ]
    # log.info(dask.compute(*da.stack(outputs_delayed)))
    # # result_filenames = dask.compute(*databag)
    # # log.info(f'Generated {result_filenames}')
