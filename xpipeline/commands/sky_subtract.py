import sys
import argparse
import fsspec.spec
import logging
import dask

from .. import utils
from .. import pipelines
from ..ref import clio
from ..tasks import iofits, sky_model, improc


from .base import MultiInputCommand

log = logging.getLogger(__name__)


class SkySubtract(MultiInputCommand):
    name = "sky_subtract"
    help = "Subtract sky background with a PCA basis model file"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "sky_model",
            help="Path to FITS file with sky model basis",
        )
        parser.add_argument(
            "--mask-dilate-iters",
            default=3,
            type=int,
            help="Iterations to grow the mask that excludes pixels from background estimation",
        )
        parser.add_argument(
            "--n-sigma",
            default=3,
            type=float,
            help="Number of sigma (standard deviations of the background model input frames) beyond which pixel is considered illuminated and excluded from background estimation"
        )
        parser.add_argument(
            "--exclude-bbox",
            action="append",
            required=True,
            help=utils.unwrap(
                """
                specification of the form ``origin_y,origin_x,height,width``
                for regions presumed illuminated for exclusion from background
                estimation (can be repeated)
            """
            ),
        )
        return super(SkySubtract, SkySubtract).add_arguments(parser)

    def main(self):
        destination = self.args.destination
        log.debug(f"{destination=}")
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        n_output_files = len(self.all_files)
        output_filepaths = [utils.join(destination, f"sky_subtract_{i:04}.fits") for i in range(n_output_files)]
        for output_file in output_filepaths:
            if dest_fs.exists(output_file):
                log.error(f"Output exists: {output_file}")
                sys.exit(1)

        excluded_bboxes = []
        for bbox_spec in self.args.exclude_bbox:
            excluded_bboxes.append(improc.BBox.from_spec(bbox_spec))

        coll = self.inputs_coll.map(iofits.load_fits_from_path)
        hdul = iofits.load_fits_from_path(self.args.sky_model)
        model_sky = sky_model.SkyModel.from_hdulist(hdul)
        output_coll = pipelines.sky_subtract(coll, model_sky, self.args.mask_dilate_iters, self.args.n_sigma, excluded_bboxes)
        return output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True).compute()

