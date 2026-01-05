import pathlib
import typing
import logging
import numpy
import os
import sys
import time
import ray
import numpy as np
import os.path
import xconf
from xconf.contrib import LocalRayConfig as _LocalRayConfig
from xconf.contrib import RemoteRayConfig as _RemoteRayConfig
from .. import core, utils, types

log = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = ("fit", "fits")
BYTES_PER_MB = 1024 * 1024

def _measure_ram_for_step(func, *args, measure_gpu_ram=False, **kwargs):
    from memory_profiler import memory_usage
    gpu_prof = utils.CupyRamHook() if measure_gpu_ram else utils.DummyRamHook()
    time_sec = time.perf_counter()
    log.debug(f"{func=} inner timer start @ {time_sec}\n{ray.get_runtime_context().get()=}")
    with gpu_prof:
        mem_mb_series = memory_usage((func, args, kwargs))
    gpu_ram_usage_mb = gpu_prof.used_bytes / BYTES_PER_MB
    final_ram_mb = memory_usage(-1, max_usage=True)
    ram_usage_mb = np.max(mem_mb_series) - final_ram_mb
    end_time_sec = time.perf_counter()
    time_sec = end_time_sec - time_sec
    log.debug(f"{func=} inner timer end @ {end_time_sec}, duration {time_sec}. {ram_usage_mb=} {gpu_ram_usage_mb=}")
    return ram_usage_mb, gpu_ram_usage_mb, time_sec
measure_ram_for_step = ray.remote(_measure_ram_for_step)


def measure_ram(func, options, *args, ram_pad_factor=1.1, measure_gpu_ram=False, **kwargs):
    log.info(f"Submitting measure_ram_for_step for {func} with {options=}")
    measure_ref = measure_ram_for_step.options(**options).remote(
        func,
        *args,
        measure_gpu_ram=measure_gpu_ram,
        **kwargs
    )
    outside_time_sec = time.time()
    log.debug(f"{func} {outside_time_sec=} at start, ref is {measure_ref}")
    ram_usage_mb, gpu_ram_usage_mb, inside_time_sec = ray.get(measure_ref)
    end_time_sec = time.time()
    log.debug(f"{func} end time {end_time_sec}")
    outside_time_sec = time.time() - outside_time_sec
    log.info(f"Measured {func} RAM use of {ram_usage_mb:1.3f} MB, GPU RAM use of {gpu_ram_usage_mb:1.3f}, runtime of {inside_time_sec:1.2f} sec inside and {outside_time_sec:1.2f} sec outside")
    # surprise! "ValueError: Resource quantities >1 must be whole numbers." from ray
    ram_requirement_mb = int(np.ceil(ram_pad_factor * ram_usage_mb))
    gpu_ram_requirement_mb = int(np.ceil(ram_pad_factor * gpu_ram_usage_mb))
    log.info(f"Setting {func} RAM requirements {ram_requirement_mb:1.3f} MB RAM and {gpu_ram_requirement_mb:1.3f} MB GPU RAM (pad factor: {ram_pad_factor:1.2f})")
    return ram_requirement_mb, gpu_ram_requirement_mb


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
    destination : str = xconf.field(default=".", help="Output directory")
    random_state : int = xconf.field(default=0, help="Initialize NumPy's random number generator with this seed")
    cpus : int = xconf.field(default=utils.available_cpus(), help="Number of CPUs free for use")

    def get_dest_fs(self):
        dest_fs = utils.get_fs(self.destination)
        log.debug(f"calling makedirs on {dest_fs} at {self.destination}")
        dest_fs.makedirs(self.destination, exist_ok=True)
        return dest_fs

    def check_for_outputs(self, output_paths):
        dest_fs = self.get_dest_fs()
        for op in output_paths:
            if dest_fs.exists(op):
                return True
        return False

    def quit_if_outputs_exist(self, output_paths):
        if self.check_for_outputs(output_paths):
            log.info(f"Outputs exist at {output_paths}")
            log.info(f"Remove to re-run")
            sys.exit(0)

    def get_output_paths(self, *output_paths):
        output_paths = [utils.join(self.destination, op) for op in output_paths]
        return output_paths

    def __post_init__(self):
        numpy.random.seed(self.random_state)
        log.debug(f"Set NumPy random seed to {self.random_state}")


@xconf.config
class InputCommand(BaseCommand):
    input : str = xconf.field(help="Input file path")

@xconf.config
class MultiInputCommand(InputCommand):
    input : str = xconf.field(help="Input file, directory, or wildcard pattern matching multiple files")
    sample_every_n : int = xconf.field(default=1, help="Take every Nth file from inputs (for speed of debugging)")
    file_extensions : list[str] = xconf.field(default=DEFAULT_EXTENSIONS, help="File extensions to match in the input (when given a directory)")

    def get_all_inputs(self, input_str):
        src_fs = utils.get_fs(input_str)
        if '*' in input_str:
            # handle globbing
            all_inputs = src_fs.glob(input_str)
        else:
            # handle directory
            if src_fs.isdir(input_str):
                all_inputs = []
                for extension in self.file_extensions:
                    glob_result = src_fs.glob(utils.join(input_str, f"*{extension}"))
                    # returned paths from glob won't have protocol string or host
                    # so take the basenames of the files and we stick the other
                    # part back on from `entry`
                    all_inputs.extend(
                        [utils.join(input_str, utils.basename(x)) for x in glob_result]
                    )
            # handle single file
            else:
                all_inputs = [input_str]
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

    def open(self, mode='rb'):
        from ..utils import get_fs
        fs = get_fs(self.path)
        return fs.open(self.path, mode)

@xconf.config
class FitsConfig(FileConfig):  #TODO pathlibify
    path : str = xconf.field(help="Path from which to load the containing FITS file")
    ext : typing.Union[int,str] = xconf.field(default=0, help="Extension from which to load")

    def load(self):
        from ..tasks import iofits
        with self.open() as fh:
            hdul = iofits.load_fits(fh)
        return hdul[self.ext].data


@xconf.config
class FitsExtConfig:
    path : pathlib.Path = xconf.field(help="Path from which to load the containing FITS file")
    ext : typing.Union[int,str] = xconf.field(default=0, help="Extension from which to load")

    def load(self):
        from astropy.io import fits
        with self.path.open('rb') as fh:
            hdul = fits.open(fh)
        return hdul[self.ext]

@xconf.config
class FitsImageConfig(FitsExtConfig):
    def load(self):
        from astropy.io import fits
        with self.path.open('rb') as fh:
            hdul = fits.open(fh)
            image = hdul[self.ext].data.astype('=f8')
        return image

@xconf.config
class FitsTableColumnConfig(FitsConfig):
    table_column : str = xconf.field(help="Path from which to load the containing FITS file")
    ext : typing.Union[int,str] = xconf.field(default="OBSTABLE", help="Extension from which to load")

    def load(self):
        from ..tasks import iofits
        with self.open() as fh:
            hdul = iofits.load_fits(fh)
        return hdul[self.ext].data[self.table_column]

def init_worker():
    import matplotlib
    matplotlib.use("Agg")
    from xpipeline import cli
    cli._configure_logging('INFO')
    log.info(f"Worker logging initalized")

@xconf.config
class LocalRayConfig(_LocalRayConfig):
    setup_function_path : typing.ClassVar[str] = "xpipeline.commands.base.init_worker"

@xconf.config
class RemoteRayConfig(_RemoteRayConfig):
    setup_function_path : typing.ClassVar[str] = "xpipeline.commands.base.init_worker"

AnyRayConfig = typing.Union[LocalRayConfig, RemoteRayConfig]
