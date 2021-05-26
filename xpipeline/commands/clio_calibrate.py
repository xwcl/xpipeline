import sys
import argparse
import fsspec.spec
import logging
import dask

from .. import utils
from .. import pipelines
from ..ref import clio
from ..tasks import iofits


from .base import MultiInputCommand

log = logging.getLogger(__name__)


class ClioCalibrate(MultiInputCommand):
    name = "clio_calibrate"
    help = "Apply bad pixel map, linearity correction, and saturation flags"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--outname-prefix",
            default="clio_calibrate_",
            help="Prefix for output filenames (default: 'clio_calibrate_')",
        )
        parser.add_argument(
            "badpix", help="Path to FITS image of bad pixel map (1 is bad, 0 is good)"
        )
    #     for optflag, info in ClioSplit._keyword_override_options.items():
    #         parser.add_argument(optflag, default=None, help=info["help"])
        return super(ClioCalibrate, ClioCalibrate).add_arguments(parser)

    # def _normalize_extension_key(self, key):
    #     try:
    #         return int(key)
    #     except ValueError:
    #         return key
    #
    # def _normalize_extension_keys(self, keys):
    #     out = []
    #     for k in keys:
    #         out.append(self._normalize_extension_key(k))
    #     return out

    def main(self):
        destination = self.args.destination
        log.debug(f"{destination=}")
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        # infer planes per cube
        hdul = iofits.load_fits_from_path(self.all_files[0])
        plane_shape = hdul[0].data.shape

        n_output_files = len(self.all_files)
        input_names = [utils.basename(fn) for fn in self.all_files]
        output_filepaths = [utils.join(destination, f"{self.args.outname_prefix}{i:04}.fits") for i in range(n_output_files)]
        for output_file in output_filepaths:
            if dest_fs.exists(output_file):
                log.error(f"Output exists: {output_file}")
                sys.exit(1)

        coll = self.inputs_coll.map(iofits.load_fits_from_path)
        badpix_path = self.args.badpix
        full_badpix_arr = iofits.load_fits_from_path(badpix_path)[0].data
        badpix_arr = clio.badpix_for_shape(full_badpix_arr, plane_shape)
        output_coll = pipelines.clio_badpix_linearity(coll, badpix_arr, plane_shape)
        return output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True).compute()

