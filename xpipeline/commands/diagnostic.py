from astropy.io import fits
import numpy as np
import argparse
import dask
import fsspec.spec
import os.path
import coloredlogs
import logging
from .. import utils
from .. import tasks, ref
# from .ref import clio

from .base import BaseCommand

log = logging.getLogger(__name__)


class Diagnostic:
    name = "diagnostic"
    help = "Power on, self-test"

    def __init__(self, cli_args: argparse.Namespace):
        logger = logging.getLogger('xpipeline')
        coloredlogs.install(level='DEBUG', logger=logger)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        pass

    def main(self):
        task_names = [x for x in dir(tasks) if x[0] != '_']
        log.info(f'task modules: {task_names}')
        ref_names = [x for x in dir(ref) if x[0] != '_']
        log.info(f'ref modules: {ref_names}')
