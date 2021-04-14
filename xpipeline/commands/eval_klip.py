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
from .klip import KLIP

from .base import BaseCommand

log = logging.getLogger(__name__)


class EvalKLIP(KLIP):
    name = "eval_klip"
    help = "Inject and recover a companion in ADI data through KLIP"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "template_psf",
            help="Path to FITS image of template PSF, scaled to the average amplitude of the host star signal"
        )
        parser.add_argument(
            "--companion-spec",
            action='append',
            required=True,
            help="specification of the form 'contrast,r,theta', can be repeated (e.g. '0.0001,34,100' for 10^-4 contrast, 34 px, 100 deg E of N)"
        )
        return super(EvalKLIP, EvalKLIP).add_arguments(parser)

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
