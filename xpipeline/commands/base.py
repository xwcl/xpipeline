import typing
import logging
import numpy
import os
import os.path
import xconf
from .. import utils

log = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = ("fit", "fits")
FITS_DEFAULT_EXT = 0

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
            import dask
            from dask.distributed import Client
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

    def check_for_outputs(self, output_paths):
        dest_fs = utils.get_fs(self.destination)
        for op in output_paths:
            if dest_fs.exists(op):
                return True
        return False
