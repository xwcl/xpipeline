import argparse
import logging
import coloredlogs
import os
import sys
import numpy
import xconf
from .commands import (
    # copy_test,
    # local_klip,
    base,
    update_headers,
    clio_split,
    clio_calibrate,
    compute_sky_model,
    sky_subtract,
    diagnostic,
    aligned_cutouts,
    combine_images,
    collect_dataset,
    scale_templates,
    inject,
    klip,
    eval_klip,
)

log = logging.getLogger(__name__)

COMMANDS = {
    # copy_test.CopyTest,
    # diagnostic.Diagnostic,
    base.BaseCommand,
    base.MultiInputCommand,
    diagnostic.Diagnostic,
    update_headers.UpdateHeaders,
    clio_split.ClioSplit,
    clio_calibrate.ClioCalibrate,
    compute_sky_model.ComputeSkyModel,
    sky_subtract.SkySubtract,
    aligned_cutouts.AlignedCutouts,
    combine_images.CombineImages,
    collect_dataset.CollectDataset,
    scale_templates.ScaleTemplates,
    inject.Inject,
    klip.Klip,
    eval_klip.EvalKlip,
}

class Dispatcher(xconf.Dispatcher):
    def configure_logging(self, level):
        for name in ["xpipeline", "irods_fsspec"]:
            logger = logging.getLogger(name)
            coloredlogs.install(level="DEBUG", logger=logger)
            logger.setLevel(level)

def main():
    d = Dispatcher(COMMANDS)
    sys.exit(d.main())
