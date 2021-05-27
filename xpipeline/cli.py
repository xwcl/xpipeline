import argparse
import logging
import coloredlogs
import os
import sys
import numpy
import xconf
from .commands import (
    # copy_test,
    # klip,
    # eval_klip,
    # local_klip,
    # diagnostic,
    # collect_dataset,
    # aligned_cutouts
    base,
    clio_split,
    clio_calibrate,
    compute_sky_model,
    sky_subtract,
    diagnostic,
)

log = logging.getLogger(__name__)

COMMANDS = {
    # copy_test.CopyTest,
    # collect_dataset.CollectDataset,
    # klip.KLIP,
    # eval_klip.EvalKLIP,
    # local_klip.LocalKLIP,
    # diagnostic.Diagnostic,
    # aligned_cutouts.AlignedCutouts,
    base.BaseCommand,
    base.MultiInputCommand,
    clio_split.ClioSplit,
    clio_calibrate.ClioCalibrate,
    compute_sky_model.ComputeSkyModel,
    sky_subtract.SkySubtract,
    diagnostic.Diagnostic
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
