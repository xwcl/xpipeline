import argparse
import logging

import numpy
import coloredlogs
from exao_dap_client.commands import base
from dask.distributed import Client

from fsspec.implementations.local import LocalFileSystem
import os.path


from ..core import PipelineCollection
from .. import utils


log = logging.getLogger(__name__)


DEFAULT_EXTENSIONS = ('fit', 'fits')


def _files_from_source(source, extensions):
    # source is a list of directories or files
    # directories should be globbed with a pattern
    # and filenames should be added as-is provided that
    # they exist
    # entries may be either paths or urls

    all_files_paths = []
    for entry in source:
        log.debug(f'Interpreting source entry {entry}')
        fs = utils.get_fs(entry)
        if isinstance(fs, LocalFileSystem):
            # relative paths are only a concern locally
            entry = os.path.realpath(entry)
        if not fs.exists(entry):
            raise RuntimeError(f"Cannot find file or directory {entry}")
        if fs.isdir(entry):
            log.debug(f'Globbing contents of {entry} for {extensions}')
            for extension in extensions:
                if extension[0] == '.':
                    extension = extension[1:]
                glob_result = fs.glob(utils.join(entry, f"*.{extension}"))
                # returned paths from glob won't have protocol string or host
                # so take the basenames of the files and we stick the other
                # part back on from `entry`
                all_files_paths.extend([utils.join(entry, os.path.basename(x)) for x in glob_result])
        else:
            all_files_paths.append(entry)
    # sort file paths lexically
    sorted_files_paths = list(sorted(all_files_paths))
    log.debug(f'Final source files set: {sorted_files_paths} on {fs}')
    if not len(sorted_files_paths):
        raise RuntimeError("Attempting to process empty set of input files")
    return sorted_files_paths


class BaseCommand(base.BaseCommand):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        for name in ['xpipeline', 'irods_fsspec', 'exao_dap_client']:
            logger = logging.getLogger(name)
            coloredlogs.install(level='DEBUG', logger=logger)
            logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
        log.debug(f'Verbose logging: {args.verbose}')

        numpy.random.seed(args.random_state)
        log.debug(f'Set random seed to {args.random_state}')
        
        if args.dask_scheduler is not None:
            log.info(f'Connecting to dask-scheduler at {args.dask_scheduler}')
            c = Client(args.dask_scheduler)  # registers with dask as a side-effect
        else:
            log.info('Starting Dask LocalCluster')
            c = Client()  # registers with dask as a side-effect
        log.info(f'Dask cluster: {c.scheduler.address} ({c.dashboard_link})')
        extensions = args.extension if len(args.extension) else DEFAULT_EXTENSIONS
        self.all_files = _files_from_source(args.source, extensions)[:: args.sample]
        self.inputs_coll = PipelineCollection(self.all_files)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "-v", "--verbose", help="Show all debugging output", action="store_true"
        )
        parser.add_argument(
            "-s", "--sample", help="Sample every Nth file in source", type=int, default=1
        )
        parser.add_argument(
            "-x", "--extension",
            help=utils.unwrap(
                '''
                Override file extensions to include in source, can be
                specified multiple times. Specify '' to match all.
                (default: ['fit', 'fits'])
                '''
            ),
            default=[],
            action='append'
        )
        parser.add_argument(
            "-d","--dask-scheduler", help="Address of existing dask-scheduler process as host:port"
        )
        parser.add_argument(
            "--random-state",
            type=int,
            default=0,
            help="Random seed state for reproducibility (default: 0)",
        )
        parser.add_argument("source", nargs="+")
        parser.add_argument("destination")
        return super(BaseCommand, BaseCommand).add_arguments(parser)
