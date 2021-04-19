from astropy.io import fits
import numpy as np
import argparse
import dask
import dask.array as da
import fsspec.spec
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


class KLIP(BaseCommand):
    name = "klip"
    help = "Subtract starlight with KLIP"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--extname",
            default="SCI",
            help=unwrap("""
                FITS extension name containing data cube planes
                (used when single combined input file provided)
            """)
        )
        parser.add_argument(
            "--region-mask",
            default=None,
            help="Path to FITS image of region mask (1 is included, 0 is excluded)"
        )
        parser.add_argument(
            "--angle-keyword",
            default='ROTOFF',
            help="FITS keyword with rotation information (default: 'ROTOFF')"
        )
        parser.add_argument(
            "--angle-extname",
            default="ANGLES",
            help=unwrap("""
                FITS extension name containing 1D array of angles corresponding to
                data cube planes (used when single combined input file provided)
            """)
        )
        parser.add_argument(
            "--angle-scale",
            type=float,
            default=1.0,
            help="Scale factor relating keyword value to angle in degrees needed to rotate image North-up (default: 1.0)"
        )
        parser.add_argument(
            "--angle-constant",
            type=float,
            default=0.0,
            help="Constant factor added to (scale * keyword value) to get angle in degrees needed to rotate image North-up (default: 0.0)"
        )
        parser.add_argument(
            "--k-klip",
            type=int,
            default=10,
            help="Number of modes to subtract in starlight subtraction (default: 10)"
        )
        parser.add_argument(
            "--exclude-nearest-n-frames",
            type=int,
            default=0,
            help="Number of frames to exclude from the observations used to compute the target"
        )
        return super(KLIP, KLIP).add_arguments(parser)

    def main(self):
        destination = self.args.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        dest_fs.makedirs(destination, exist_ok=True)

        output_klip_final = utils.join(destination, "klip_final.fits")
        output_exptime_map = utils.join(destination, "klip_exptime_map.fits")
        if self.check_for_outputs([output_klip_final, output_exptime_map]):
            return
        if len(self.all_files) == 1:
            extname = self.args.extname
            rotation_extname = self.args.angle_extname
            input_cube_hdul = iofits.load_fits_from_path(self.all_files[0])
            sci_arr = da.from_array(input_cube_hdul[self.args.extname].data.astype('=f8')[::self.args.sample])
            rot_arr = da.from_array(input_cube_hdul[self.args.angle_extname].data.astype('=f8')[::self.args.sample])
            default_region_mask = np.ones_like(input_cube_hdul[extname].data[0])
        else:
            inputs = self.inputs_coll.map(iofits.load_fits_from_path)
            first_hdul = dask.persist(inputs.collection[0])[0].compute()
            plane_shape = first_hdul[0].data.shape
            sci_arr = sorted_inputs_collection.collect(iofits.hdulists_to_dask_cube, plane_shape)
            rot_arr = sorted_inputs_collection.collect(iofits.hdulists_keyword_to_dask_array, self.args.angle_keyword)
            default_region_mask = np.ones_like(first_hdul[0].data)

        if self.args.region_mask is not None:
            with open(self.args.region_mask, 'rb') as fh:
                region_mask = fits.open(fh)[0].data
        else:
            region_mask = default_region_mask

        out_image = pipelines.klip_adi(
            sci_arr,
            rot_arr,
            region_mask,
            self.args.angle_keyword,
            self.args.angle_scale,
            self.args.angle_constant,
            self.args.exclude_nearest_n_frames,
            self.args.k_klip
        )
        import time
        start = time.perf_counter()
        log.info(f"Computing klip pipeline result...")
        out_image = out_image.compute()
        elapsed = time.perf_counter() - start
        log.info(f"Computed in {elapsed} sec")
        output_file = iofits.write_fits(iofits.DaskHDUList([iofits.DaskHDU(out_image)]), output_klip_final)
        return dask.compute(output_file)
