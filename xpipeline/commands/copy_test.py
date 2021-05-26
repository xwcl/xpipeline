from astropy.io import fits
import numpy as np
import argparse
import dask
import fsspec.spec
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


class CopyTest(MultiInputCommand):
    name = "copy_test"
    help = "Test CLI infrastructure by copying inputs to destination"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        return super(CopyTest, CopyTest).add_arguments(parser)

    def main(self):
        destination = self.args.destination
        log.debug(f"{destination=}")
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        output_files = (
            self.inputs_coll.map(utils.basename)
            .map(lambda x: utils.join(destination, x))
            .end()
        )
        log.debug(output_files)
        inputs = self.inputs_coll.map(iofits.load_fits_from_path)
        output_paths = inputs.zip_map(iofits.write_fits, output_files, overwrite=True)
        return output_paths.compute()
