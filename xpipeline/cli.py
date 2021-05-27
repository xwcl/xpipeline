import argparse
import logging
import coloredlogs
import os
import sys
import numpy

from dask.distributed import Client

from exao_dap_client.data_store import get_fs
from .commands import (
    # compute_sky_model,
    # copy_test,
    # klip,
    # eval_klip,
    # local_klip,
    # diagnostic,
    # collect_dataset,
    # clio_split,
    # clio_calibrate,
    # sky_subtract,
    aligned_cutouts
)

from . import utils
import xconf

log = logging.getLogger(__name__)

from .commands import base

COMMANDS = {
    # compute_sky_model.ComputeSkyModel,
    # copy_test.CopyTest,
    # collect_dataset.CollectDataset,
    # klip.KLIP,
    # eval_klip.EvalKLIP,
    # local_klip.LocalKLIP,
    # diagnostic.Diagnostic,
    # clio_split.ClioSplit,
    # clio_calibrate.ClioCalibrate,
    # sky_subtract.SkySubtract,
    aligned_cutouts.AlignedCutouts,
    base.BaseCommand,
    base.MultiInputCommand,
}

class Dispatcher(xconf.Dispatcher):
    def configure_logging(self, level):
        for name in ["xpipeline", "irods_fsspec", "exao_dap_client"]:
            logger = logging.getLogger(name)
            coloredlogs.install(level="DEBUG", logger=logger)
            logger.setLevel(level)

def main():
    d = Dispatcher(COMMANDS)
    sys.exit(d.main())
