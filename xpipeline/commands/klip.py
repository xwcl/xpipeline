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
from ..core import PipelineCollection
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
            "--k-klip-value",
            type=int,
            default=10,
            help="Number of modes to subtract in starlight subtraction (default: 10)"
        )
        return super(KLIP, KLIP).add_arguments(parser)

    def main(self):
        destination = self.args.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        dest_fs.makedirs(destination, exist_ok=True)

        output_klip_final = utils.join(destination, "klip_final.fits")
        inputs = self.inputs_coll.map(iofits.load_fits)

        if self.args.region_mask is not None:
            region_mask = iofits.load_fits_from_path(self.args.region_mask)[0].data.persist()
        else:
            first = dask.persist(inputs[0]).compute()
            region_mask = da.ones_like(first[0].data)

        out_image = pipelines.klip_adi(
            inputs,
            region_mask,
            self.args.rotation_keyword,
            self.args.rotation_scale,
            self.args.rotation_offset,
            self.args.rotation_exclusion_frames,
            self.args.k_klip
        )
        output_file = iofits.write_fits(out_image, output_klip_final)
        return dask.compute(output_file)
