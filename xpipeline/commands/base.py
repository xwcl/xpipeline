import typing
import logging
import numpy
import os
import sys
import os.path
import xconf
from .. import core, utils, types

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
    n_processes : int = xconf.field(default=None, help="Override automatically selected number of processes to spawn")
    n_threads : int = xconf.field(default=None, help="Override automatically selected number of threads per process to spawn")
    synchronous : bool = xconf.field(default=True, help="Whether to disable Dask's parallelism in favor of lower levels (ignored when distributed=True)")

@xconf.config
class BaseCommand(xconf.Command):
    destination : str = xconf.field(help="Output directory")
    dask : DaskConfig = xconf.field(default_factory=lambda: DaskConfig(), help="Configure Dask executor")
    random_state : int = xconf.field(default=0, help="Initialize NumPy's random number generator with this seed")
    cpus : int = xconf.field(default=utils.available_cpus(), help="Number of CPUs free for use")

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

    def __post_init__(self):
        numpy.random.seed(self.random_state)
        log.debug(f"Set NumPy random seed to {self.random_state}")

        import dask
        temp_dir = _determine_temporary_directory()
        if temp_dir is not None:
            dask.config.set({'temporary-directory': temp_dir})
        if self.dask.distributed:
            from dask.distributed import Client
            # instantiating Client registers with dask as a side-effect
            if self.dask.host is not None:
                log.info("Connecting to existing cluster")
                c = Client(address=f'{self.dask.host}:{self.dask.port}')
            else:
                log.info("Starting Dask LocalCluster")
                from distributed.deploy.utils import nprocesses_nthreads
                # This has to be done by us because Dask is using psutil process
                # affinity and on Puma that automatically detects only 1 CPU:
                nproc, nthread = nprocesses_nthreads(self.cpus)
                if self.dask.n_processes is not None:
                    nproc = self.dask.n_processes
                if self.dask.n_threads is not None:
                    nthread = self.dask.n_threads
                log.info(f'Using {nproc} processes with {nthread} threads per process')
                c = Client(
                    n_workers=nproc,
                    threads_per_worker=nthread,
                    silence_logs=self.dask.log_level,
                )
            log.info("Preloading xpipeline in Dask workers")
            c.register_worker_plugin(core.DaskWorkerPreloadPlugin)
            log.info(f"Dask cluster: {c.scheduler.address} ({c.dashboard_link})")
        elif self.dask.synchronous:
            dask.config.set(scheduler='synchronous')


@xconf.config
class InputCommand(BaseCommand):
    input : str = xconf.field(help="Input file path")


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

@xconf.config
class MeasurementConfig:
    r_px : float = xconf.field(help="Radius of companion")
    pa_deg : float = xconf.field(help="Position angle of companion in degrees East of North")


@xconf.config
class CompanionConfig(MeasurementConfig):
    scale : float = xconf.field(help=utils.unwrap(
        """Scale factor multiplied by template (and optional template
        per-frame scale factor) to give companion image,
        i.e., contrast ratio. Can be negative or zero."""))
    def to_companionspec(self):
        from xpipeline.tasks.characterization import CompanionSpec
        return CompanionSpec(self.r_px, self.pa_deg, self.scale)

@xconf.config
class TemplateConfig:
    path : str = xconf.field(help=utils.unwrap(
        """Path to FITS image of template PSF, scaled to the
        average amplitude of the host star signal such that
        multiplying by the contrast gives an appropriately
        scaled planet PSF"""
    ))
    ext : typing.Optional[types.FITS_EXT] = xconf.field(default=None, help=utils.unwrap("""
        Extension containing the template data (default: same as template name)
    """))
    scale_factors_path : typing.Optional[str] = xconf.field(help=utils.unwrap(
        """Path to FITS file with extensions for each data extension
        containing 1D arrays of scale factors that match template PSF
        intensity to real PSF intensity per-frame"""
    ))
    scale_factors_ext : typing.Optional[types.FITS_EXT] = xconf.field(default=None, help=utils.unwrap("""
        Extension containing the per-frame scale factors by which
        the template data is multiplied before applying the
        companion scale factor (default: same as template name)
    """))


@xconf.config
class PixelRotationRangeConfig:
    delta_px : float = xconf.field(default=0, help="Maximum difference between target frame value and matching frames")
    r_px : float = xconf.field(default=None, help="Radius at which to calculate motion in pixels")

@xconf.config
class AngleRangeConfig:
    delta_deg : float = xconf.field(default=0, help="Maximum difference between target frame value and matching frames")

@xconf.config
class FileConfig:
    path : str = xconf.field(help="File path")

    def open(self, mode='r'):
        from ..utils import get_fs
        fs = get_fs(self.path)
        return fs.open(self.path, mode)

@xconf.config
class FitsConfig:
    path : str = xconf.field(help="Path from which to load the containing FITS file")
    ext : typing.Union[int,str] = xconf.field(default=0, help="Extension from which to load")

    def load(self):
        from ..tasks import iofits
        hdul = iofits.load_fits_from_path(self.path)
        return hdul[self.ext].data

@xconf.config
class FitsTableColumnConfig(FitsConfig):
    table_column : str = xconf.field(help="Path from which to load the containing FITS file")
    ext : typing.Union[int,str] = xconf.field(default="OBSTABLE", help="Extension from which to load")

    def load(self):
        from ..tasks import iofits
        hdul = iofits.load_fits_from_path(self.path)
        return hdul[self.ext].data[self.table_column]
