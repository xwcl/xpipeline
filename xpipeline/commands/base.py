import typing
import logging
import numpy
import os
import sys
import os.path
import xconf
from .. import core, utils

log = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = ("fit", "fits")

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
    distributed : bool = xconf.field(default=False, help="Whether to execute Dask workflows with a cluster to parallelize work")
    log_level : str = xconf.field(default='WARN', help="What level of logs to show from the scheduler and workers")
    port : int = xconf.field(default=8786, help="Port to contact dask-scheduler")
    host : typing.Optional[str] = xconf.field(default=None, help="Hostname of running dask-scheduler")

@xconf.config
class BaseCommand(xconf.Command):
    dask : DaskConfig = xconf.field(default_factory=lambda: DaskConfig(), help="Configure Dask executor")
    random_state : int = xconf.field(default=0, help="Initialize NumPy's random number generator with this seed")
    cpus : int = xconf.field(default=utils.available_cpus(), help="Number of CPUs free for use")

    def __post_init__(self):
        numpy.random.seed(self.random_state)
        log.debug(f"Set NumPy random seed to {self.random_state}")

        if self.dask.distributed:
            import dask
            from dask.distributed import Client
            temp_dir = _determine_temporary_directory()
            if temp_dir is not None:
                dask.config.set({'temporary-directory': temp_dir})
            # registers with dask as a side-effect
            if self.dask.host is not None:
                log.info("Connecting to existing cluster")
                c = Client(address=f'{self.dask.host}:{self.dask.port}')
            else:
                log.info("Starting Dask LocalCluster")
                # This has to be done by us because Dask is using psutil process
                # affinity and on Puma that automatically detects only 1 CPU
                from distributed.deploy.utils import nprocesses_nthreads
                nproc, nthread = nprocesses_nthreads(self.cpus)
                log.info(f'Using {nproc} processes with {nthread} threads per process')
                c = Client(
                    n_workers=nproc,
                    threads_per_worker=nthread,
                    silence_logs=self.dask.log_level,
                )
            log.info("Preloading xpipeline in Dask workers")
            c.register_worker_plugin(core.DaskWorkerPreloadPlugin)
            log.info(f"Dask cluster: {c.scheduler.address} ({c.dashboard_link})")


@xconf.config
class InputCommand(BaseCommand):
    input : str = xconf.field(help="Input file path")
    destination : str = xconf.field(help="Output directory")

    def check_for_outputs(self, output_paths):
        dest_fs = utils.get_fs(self.destination)
        for op in output_paths:
            if dest_fs.exists(op):
                return True
        return False

    def quit_if_outputs_exist(self, output_paths):
        if self.check_for_outputs(output_paths):
            log.info(f"Outputs exist at {output_paths}")
            log.info(f"Remove to re-run")
            sys.exit(0)


@xconf.config
class MultiInputCommand(InputCommand):
    input : str = xconf.field(help="Input file, directory, or wildcard pattern matching multiple files")
    sample_every_n : int = xconf.field(default=1, help="Take every Nth file from inputs (for speed of debugging)")
    file_extensions : list[str] = xconf.field(default=DEFAULT_EXTENSIONS, help="File extensions to match in the input (when given a directory)")

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
        return list(sorted(all_inputs))[::self.sample_every_n]

