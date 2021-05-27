import sys
import argparse
import fsspec.spec
import logging

from .. import utils
from .. import pipelines
from ..tasks import iofits


from .base import MultiInputCommand

log = logging.getLogger(__name__)


class ClioSplit(MultiInputCommand):
    name = "clio_split"
    help = "Split Clio datacubes into frames and interpolate header telemetry values"

    _keyword_override_options = {
        "--telescope": {
            "keyword": "TELESCOP",
            "help": "Name of telescope where data were taken",
        },
        "--instrument": {
            "keyword": "INSTRUME",
            "help": "Name of instrument with which data were taken",
        },
        "--observer": {
            "keyword": "OBSERVER",
            "help": "Name of observer",
        },
        "--object": {
            "keyword": "OBJECT",
            "help": "Name object observed",
        },
    }

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--outname-prefix",
            default="clio_split_",
            help="Prefix for output filenames (default: 'clio_split_')",
        )
        for optflag, info in ClioSplit._keyword_override_options.items():
            parser.add_argument(optflag, default=None, help=info["help"])
        return super(ClioSplit, ClioSplit).add_arguments(parser)

    def _normalize_extension_key(self, key):
        try:
            return int(key)
        except ValueError:
            return key

    def _normalize_extension_keys(self, keys):
        out = []
        for k in keys:
            out.append(self._normalize_extension_key(k))
        return out

    def main(self):
        destination = self.args.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        # infer planes per cube
        hdul = iofits.load_fits_from_path(self.all_files[0])
        planes = hdul[0].data.shape[0]
        # plane_shape = hdul[0].data.shape[1:]

        n_output_files = len(self.all_files) * planes
        input_names = [utils.basename(fn) for fn in self.all_files]
        output_filepaths = [utils.join(destination, f"{self.args.outname_prefix}{i:04}.fits") for i in range(n_output_files)]
        for output_file in output_filepaths:
            if dest_fs.exists(output_file):
                log.error(f"Output exists: {output_file}")
                sys.exit(1)

        coll = self.inputs_coll.map(iofits.load_fits_from_path)
        output_coll = pipelines.clio_split(coll, input_names, frames_per_cube=planes)
        return output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True).compute()

