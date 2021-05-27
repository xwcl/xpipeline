import typing
import argparse
import logging

import numpy
import coloredlogs
from exao_dap_client.commands import base
from dask.distributed import Client

from fsspec.spec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
import os
import os.path

# from dataclasses import dataclass
import dask

from ..core import LazyPipelineCollection
from .. import utils
import xconf
from ..utils import unwrap


log = logging.getLogger(__name__)


DEFAULT_EXTENSIONS = ("fit", "fits")
FITS_DEFAULT_EXT = 0

def _files_from_source(source, extensions):
    # source is a list of directories or files
    # directories should be globbed with a pattern
    # and filenames should be added as-is provided that
    # they exist
    # entries may be either paths or urls

    all_files_paths = []
    for entry in source:
        log.debug(f"Interpreting source entry {entry}")
        fs = utils.get_fs(entry)
        if isinstance(fs, LocalFileSystem):
            # relative paths are only a concern locally
            entry = os.path.realpath(entry)
        if not fs.exists(entry):
            raise RuntimeError(f"Cannot find file or directory {entry}")
        if fs.isdir(entry):
            log.debug(f"Globbing contents of {entry} for {extensions}")
            for extension in extensions:
                if extension[0] == ".":
                    extension = extension[1:]
                glob_result = fs.glob(utils.join(entry, f"*.{extension}"))
                # returned paths from glob won't have protocol string or host
                # so take the basenames of the files and we stick the other
                # part back on from `entry`
                all_files_paths.extend(
                    [utils.join(entry, os.path.basename(x)) for x in glob_result]
                )
        else:
            all_files_paths.append(entry)
    # sort file paths lexically
    sorted_files_paths = list(sorted(all_files_paths))
    log.debug(f"Final source files set: {sorted_files_paths} on {fs}")
    if not len(sorted_files_paths):
        raise RuntimeError("Attempting to process empty set of input files")
    return sorted_files_paths


def _determine_temporary_directory():
    if "OSG_WN_TMP" in os.environ:
        temp_dir = os.environ["OSG_WN_TMP"]
    elif "TMPDIR" in os.environ:
        temp_dir = os.environ["TMPDIR"]
    elif os.path.exists("/tmp"):
        temp_dir = "/tmp"
    else:
        temp_dir = None  # default behavior: use pwd
    return temp_dir


# class BaseCommand(base.BaseCommand):
#     """Base class for CLI commands"""

#     def __init__(self, cli_args: argparse.Namespace):
#         super().__init__(cli_args)
#         for name in ["xpipeline", "irods_fsspec", "exao_dap_client"]:
#             logger = logging.getLogger(name)
#             coloredlogs.install(level="DEBUG", logger=logger)
#             logger.setLevel(logging.DEBUG if cli_args.verbose else logging.INFO)
#         log.debug(f"Verbose logging: {cli_args.verbose}")

#         numpy.random.seed(cli_args.random_state)
#         log.debug(f"Set random seed to {cli_args.random_state}")

#         if not cli_args.disable_dask:
#             temp_dir = _determine_temporary_directory()
#             if temp_dir is not None:
#                 os.environ["DASK_TEMPORARY_DIRECTORY"] = temp_dir
#                 dask.config.refresh()
#             log.debug(f"{dask.config.config=}")
#             if cli_args.dask_scheduler is not None:
#                 log.info(f"Connecting to dask-scheduler at {cli_args.dask_scheduler}")
#                 c = Client(
#                     cli_args.dask_scheduler
#                 )  # registers with dask as a side-effect
#             else:
#                 log.info("Starting Dask LocalCluster")
#                 # registers with dask as a side-effect
#                 c = Client(
#                     silence_logs=logging.INFO
#                     if not cli_args.verbose
#                     else logging.DEBUG,
#                     processes=False,
#                 )
#             log.info(f"Dask cluster: {c.scheduler.address} ({c.dashboard_link})")

#     def check_for_outputs(self, output_files):
#         existing_files = [fn for fn in output_files if self.dest_fs.exists(fn)]
#         if len(existing_files):
#             log.info(f"Existing outputs: {existing_files}")
#             log.info("Remove them to re-run")
#             return True
#         else:
#             return False

#     @staticmethod
#     def add_arguments(parser: argparse.ArgumentParser):
#         parser.add_argument(
#             "-v", "--verbose", help="Show all debugging output", action="store_true"
#         )
#         parser.add_argument(
#             "--random-state",
#             type=int,
#             default=0,
#             help="Random seed state for reproducibility (default: 0)",
#         )
#         parser.add_argument(
#             "-d",
#             "--dask-scheduler",
#             help="Address of existing dask-scheduler process as host:port",
#         )
#         parser.add_argument(
#             "-D",
#             "--disable-dask",
#             help="Skip initializing dask, overrides ``--dask-scheduler``",
#             action="store_true",
#         )
#         return super(BaseCommand, BaseCommand).add_arguments(parser)

@xconf.config
class DaskConfig:
    distributed : bool = xconf.field(default=True, help="Whether to execute Dask workflows with a cluster of local threads to parallelize work")
    log_level : str = xconf.field(default='INFO', help="What level of logs to show from the scheduler and workers")

@xconf.config
class BaseCommand(xconf.Command):
    dask : DaskConfig = xconf.field(default_factory=lambda: DaskConfig(), help="Configure Dask executor")
    random_state : int = xconf.field(default=0, help="Initialize NumPy's random number generator with this seed")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        numpy.random.seed(self.random_state)
        log.debug(f"Set NumPy random seed to {self.random_state}")

        if self.dask.distributed:
            temp_dir = _determine_temporary_directory()
            if temp_dir is not None:
                os.environ["DASK_TEMPORARY_DIRECTORY"] = temp_dir
                dask.config.refresh()
            log.info("Starting Dask LocalCluster")
            # registers with dask as a side-effect
            c = Client(
                silence_logs=self.dask.log_level,
                processes=True,
            )
            log.info(f"Dask cluster: {c.scheduler.address} ({c.dashboard_link})")


@xconf.config
class MultiInputCommand(BaseCommand):
    input : str = xconf.field(help="Input file, directory, or wildcard pattern matching multiple files")
    destination : str = xconf.field(help="Output directory")
    sample_every_n : int = xconf.field(default=1, help="Take every Nth file from inputs (for speed of debugging)")
    file_extensions : list[str] = xconf.field(default=DEFAULT_EXTENSIONS, help="File extensions to match in the input (when given a directory)")
    ext : typing.Union[str,int] = xconf.field(default=FITS_DEFAULT_EXT, help="Extension index or name to load from input files")

    def get_all_inputs(self):
        src_fs = utils.get_fs(self.input)
        if '*' in self.input:
            # handle globbing
            all_inputs = src_fs.glob(self.input)
        else:
            # handle directory
            if src_fs.isdir(self.input):
                all_inputs = []
                for extension in self.file_extensions:
                    glob_result = src_fs.glob(utils.join(self.input, f"*{extension}"))
                    # returned paths from glob won't have protocol string or host
                    # so take the basenames of the files and we stick the other
                    # part back on from `entry`
                    all_inputs.extend(
                        [utils.join(self.input, utils.basename(x)) for x in glob_result]
                    )
            # handle single file
            else:
                all_inputs = [self.input]
        return all_inputs

# class MultiInputCommand(BaseCommand):
#     def __init__(self, cli_args: argparse.Namespace):
#         super().__init__(cli_args)
#         extensions = (
#             cli_args.extension if len(cli_args.extension) else DEFAULT_EXTENSIONS
#         )
#         self.all_files = _files_from_source(cli_args.source, extensions)[
#             :: cli_args.sample
#         ]

#         self.dest_fs = utils.get_fs(cli_args.destination)
#         assert isinstance(self.dest_fs, AbstractFileSystem)
#         self.dest_fs.makedirs(cli_args.destination, exist_ok=True)
#         self.destination = cli_args.destination

#         self.inputs_coll = LazyPipelineCollection(self.all_files)
#         self.sample = cli_args.sample

#     @staticmethod
#     def add_arguments(parser: argparse.ArgumentParser):
#         parser.add_argument(
#             "-s",
#             "--sample",
#             help="Sample every Nth file in source",
#             type=int,
#             default=1,
#         )
#         parser.add_argument(
#             "source",
#             nargs="+",
#             help=unwrap(
#                 """
#             One or more source files or directories to process.
#             directories are globbed using the value of the
#             ``--extension`` option to determine valid file extensions.
#             Note that some commands may operate differently when
#             a single input file is provided rather than a collection,
#             i.e., interpreting it as a data cube.

#             Paths may be given as local filesystem paths, or as any
#             protocol supported by fsspec (including ``irods://``)
#         """
#             ),
#         )
#         parser.add_argument(
#             "destination",
#             help=unwrap(
#                 """
#             Destination where command outputs are to be saved. Commands
#             will check for existing outputs and fail fast if they are found.
#             Path may be given as local filesystem paths, or as any
#             protocol supported by fsspec (including ``irods://``)
#         """
#             ),
#         )
#         parser.add_argument(
#             "-x",
#             "--extension",
#             help=utils.unwrap(
#                 """
#                 Override file extensions to include in source, can be
#                 specified multiple times. Specify '' to match all.
#                 (default: ['fit', 'fits'])
#                 """
#             ),
#             default=[],
#             action="append",
#         )
#         return super(MultiInputCommand, MultiInputCommand).add_arguments(parser)
