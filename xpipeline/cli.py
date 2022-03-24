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
    combine,
    collect_dataset,
    scale_templates,
    inject,
    klip,
    eval_klip,
    subtract_starlight,
    stack,
    evaluate,
    vapp_trap,
    summarize_grid,
    klipt_fm,
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
    combine.Combine,
    collect_dataset.CollectDataset,
    scale_templates.ScaleTemplates,
    inject.Inject,
    klip.Klip,
    eval_klip.EvalKlip,
    subtract_starlight.SubtractStarlight,
    stack.Stack,
    evaluate.Evaluate,
    vapp_trap.VappTrap,
    summarize_grid.SummarizeGrid,
    klipt_fm.KlipTFm
}

class Dispatcher(xconf.Dispatcher):
    def configure_logging(self, level):
        # remove existing handlers
        root_logger = logging.getLogger()
        for h in root_logger.handlers:
            root_logger.removeHandler(h)
        # apply verbosity
        for logger_name in ['xpipeline', 'xconf']:
            pkglog = logging.getLogger(logger_name)
            pkglog.setLevel(level)
            # add colors (if a tty)
            coloredlogs.install(level=level, logger=pkglog)


def main():
    d = Dispatcher(COMMANDS)
    sys.exit(d.main())
