import pathlib
import toml
import time
import xconf
import ray
from astropy.io import fits
from astropy.convolution import convolve_fft
# from xconf import FileConfig, join, PathConfig, DirectoryConfig
from xconf.contrib import BaseRayGrid, FileConfig, PathConfig, DirectoryConfig
import sys
import logging
from typing import Optional, Union
from xpipeline.types import FITS_EXT
from enum import Enum
import numpy as np
from dataclasses import dataclass
from ..tasks import iofits, vapp, improc, starlight_subtraction, characterization
from ..ref import clio
from .. import pipelines
from .. import utils, constants
from ..core import cupy as cp

from .base import BaseCommand, AngleRangeConfig, PixelRotationRangeConfig, FitsConfig, FitsTableColumnConfig, AnyRayConfig, LocalRayConfig

log = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024

@xconf.config
class SamplingConfig:
    n_radii : int = xconf.field(help="Number of steps in radius at which to probe contrast")
    spacing_px : float = xconf.field(help="Spacing in pixels between contrast probes along circle (sets number of probes at radius by 2 * pi * r / spacing)")
    scales : list[float] = xconf.field(default_factory=lambda: [0.0], help="Probe contrast levels (C = companion / host)")
    iwa_px : float = xconf.field(help="Inner working angle (px)")
    owa_px : float = xconf.field(help="Outer working angle (px)")

    def __post_init__(self):
        # to make use of this for detection, we must also apply the matched
        # filter in the no-injection case for each combination of parameters
        self.scales = [float(s) for s in self.scales]
        if 0.0 not in self.scales:
            self.scales.insert(0, 0.0)

@xconf.config
class PerTaskConfig:
    generate : float = xconf.field(help="amount required in model generation")
    decompose : float = xconf.field(help="amount required in basis computation")
    evaluate : float = xconf.field(help="amount required in fitting")

@dataclass
class TaskOptions:
    generate : dict
    decompose : dict
    evaluate : dict

@dataclass
class ModelInputs:
    data_cube_shape : tuple[int,int,int]
    left_template : np.ndarray
    right_template : np.ndarray
    left_scales : np.ndarray
    right_scales : np.ndarray
    angles : np.ndarray
    mask : np.ndarray

def _measure_ram(func, options, *args, ram_pad_factor=1.1, _measure_gpu_ram=False, **kwargs):
    log.info(f"Submitting measure_ram_for_step for {func} with {options=}")
    measure_ref = measure_ram_for_step.options(**options).remote(
        func,
        *args,
        _measure_gpu_ram=_measure_gpu_ram,
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

def generate_model(model_inputs : ModelInputs, companion_r_px, companion_pa_deg):
    companion_spec = characterization.CompanionSpec(companion_r_px, companion_pa_deg, 1.0)
    # generate
    left_model_cube = characterization.generate_signals(
        model_inputs.data_cube_shape,
        [companion_spec],
        model_inputs.left_template,
        model_inputs.angles,
        model_inputs.left_scales
    )
    right_model_cube = characterization.generate_signals(
        model_inputs.data_cube_shape,
        [companion_spec],
        model_inputs.right_template,
        model_inputs.angles,
        model_inputs.right_scales
    )
    # stitch
    out_cube = pipelines.vapp_stitch(left_model_cube, right_model_cube, clio.VAPP_PSF_ROTATION_DEG)
    model_vecs = improc.unwrap_cube(out_cube, model_inputs.mask)
    return model_vecs

def _inject_model(image_vecs, model_inputs : ModelInputs, companion_r_px, companion_pa_deg, companion_scale):
    model_vecs = generate_model(model_inputs, companion_r_px, companion_pa_deg)
    return image_vecs + companion_scale * model_vecs
inject_model = ray.remote(_inject_model)

def _measure_ram_for_step(func, *args, _measure_gpu_ram=False, **kwargs):
    from memory_profiler import memory_usage
    gpu_prof = utils.CupyRamHook() if _measure_gpu_ram else utils.DummyRamHook()
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

def _precompute_basis(image_vecs, model_inputs : ModelInputs, r_px, ring_exclude_px, k_modes_max, decompose_on_gpu):
    rho, _ = improc.polar_coords(improc.arr_center(model_inputs.mask), model_inputs.mask.shape)
    included_pix_mask = np.abs(rho - r_px) > ring_exclude_px / 2
    included_vecs_mask = improc.unwrap_image(included_pix_mask, model_inputs.mask)
    assert included_vecs_mask.shape[0] == image_vecs.shape[0]
    ref_vecs = image_vecs[included_vecs_mask]
    params = starlight_subtraction.TrapParams(
        k_modes=k_modes_max,
        force_gpu_decomposition=decompose_on_gpu,
        return_basis=True,
    )
    precomputed_trap_basis = starlight_subtraction.compute_temporal_basis(ref_vecs, params)
    if decompose_on_gpu:
        precomputed_trap_basis.temporal_basis = precomputed_trap_basis.temporal_basis.get()
    return precomputed_trap_basis
precompute_basis = ray.remote(_precompute_basis)

def _evaluate_point_kt(
    row, 
    inject_image_vecs,
    model_inputs,
    precomputed_basis, 
    resel_px, 
    coverage_mask, 
    exclude_nearest=1, 
    save_to_dir: Optional[DirectoryConfig]=None, 
    evaluate_on_gpu=False
):
    # just use klip-transpose / ADI / matched filter, no simultaneous fit
    log.info(f"evaluate_point_kt start {time.time()=}")
    start = time.perf_counter()
    row = row.copy() # since ray is r/o
    companion_r_px, companion_pa_deg = float(row['r_px']), float(row['pa_deg'])  # convert to primitive type so numba doesn't complain
    model_gen_sec = time.perf_counter()
    model_vecs = generate_model(model_inputs, companion_r_px, companion_pa_deg)
    model_gen_sec = model_gen_sec - time.perf_counter()
    params_kt = starlight_subtraction.KlipTParams(
        k_modes=row['k_modes'],
        compute_residuals=True,
        precomputed_basis=precomputed_basis,
    )
    if evaluate_on_gpu:
        precomputed_basis.temporal_basis = cp.asarray(precomputed_basis.temporal_basis)
        inject_image_vecs = cp.asarray(inject_image_vecs)
        model_vecs = cp.asarray(model_vecs)
    image_resid_vecs, model_resid_vecs = starlight_subtraction.klip_transpose(inject_image_vecs, model_vecs, params_kt)
    if evaluate_on_gpu:
        image_resid_vecs, model_resid_vecs = image_resid_vecs.get(), model_resid_vecs.get()
    resid_cube_kt = improc.wrap_matrix(image_resid_vecs, model_inputs.mask)
    mf_cube_kt = improc.wrap_matrix(model_resid_vecs, model_inputs.mask)

    finim = pipelines.adi(resid_cube_kt, model_inputs.angles, pipelines.CombineOperation.SUM)
    mf_finim = pipelines.adi(mf_cube_kt, model_inputs.angles, pipelines.CombineOperation.SUM)
    dx, dy = characterization.r_pa_to_x_y(companion_r_px, companion_pa_deg, 0, 0)
    dx, dy = float(dx), float(dy)
    mf_ctr = improc.shift2(mf_finim, -dx, -dy)
    mf_ctr[np.abs(mf_ctr) < params_kt.model_trim_threshold * np.max(mf_ctr)] = 0
    mf_ctr /= np.nansum(mf_ctr**2)
    fltrd = convolve_fft(finim, np.flip(mf_ctr, axis=(0, 1)), normalize_kernel=False, nan_treatment='fill')
    fltrd[~coverage_mask] = np.nan

    snr, signal = characterization.snr_from_convolution(
        fltrd,
        companion_r_px,
        companion_pa_deg,
        aperture_diameter_px=resel_px,
        exclude_nearest=exclude_nearest
    )

    if save_to_dir is not None:
        save_to_dir.ensure_exists()
        destfs = save_to_dir.get_fs()
        destfs.makedirs(utils.join(save_to_dir.path, "gridpoints"), exist_ok=True)
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(finim, name='finim'),
            fits.ImageHDU(mf_finim, name='mf_finim'),
            fits.ImageHDU(mf_ctr, name='mf_ctr'),
            fits.ImageHDU(fltrd, name='fltrd')
            # fits.ImageHDU(planet_mask.astype(int), name='planet_mask'),
            # fits.ImageHDU(ring_mask.astype(int), name='ring_mask'),
        ])
        dest_path = utils.join(save_to_dir.path, "gridpoints", f"point_{int(row['index'])}.fits")
        with destfs.open(dest_path, mode='wb') as fh:
            hdul.writeto(fh)
        log.info(f"Wrote gridpoint image to {dest_path}")
    row['time_total_sec'] = time.perf_counter() - start
    row['time_generate_sec'] = time.perf_counter() - model_gen_sec
    row['time_decompose_sec'] = timers['time_svd_sec']
    row['pix_used'] = pix_used
    row['signal'] = signal
    row['snr'] = snr
    log.info(f"evaluate_point_kt end {time.time()=}")
    return row
evaluate_point_kt = ray.remote(_evaluate_point_kt)

@xconf.config
class KlipTFm(BaseCommand, BaseRayGrid):
    @classmethod
    @property
    def name(cls):
        return 'klipt_fm'

    # overridden from BaseRayGrid to incorporate custom worker init func
    ray : AnyRayConfig = xconf.field(
        default_factory=LocalRayConfig,
        help="Ray distributed framework configuration"
    )
    # TODO use DirectoryConfig more broadly, this just overrides the destination from BaseCommand
    destination : pathlib.Path = xconf.field(default=pathlib.Path("."), help="Directory for output files")
    
    # new options
    input : FileConfig = xconf.field(help="Path to input FITS file with collected dataset and metadata")
    checkpoint : str = xconf.field(default=None, help="Save checkpoints to this path, and/or resume grid from this checkpoint (no verification of parameters used to generate the grid is performed)")
    checkpoint_every_x : int = xconf.field(default=10, help="Write current state of grid to disk every X iterations")
    save_images : bool = xconf.field(default=False, help="Whether KLIP^T-FM images used for SNR estimation should be written to disk (warning: could be lots)")
    every_t_frames : int = xconf.field(default=1, help="Use every Tth frame as the input cube")
    k_modes_vals : list[int] = xconf.field(default_factory=lambda: [15, 100], help="")
    left_extname : FITS_EXT = xconf.field(help="")
    left_template : FitsConfig = xconf.field(help="")
    right_template : FitsConfig = xconf.field(help="")
    right_extname : FITS_EXT = xconf.field(help="")
    left_scales : FitsConfig = xconf.field(help="")
    right_scales : FitsConfig = xconf.field(help="")
    left_mask : FitsConfig = xconf.field(help="Mask that is 1 where pixels should be included in the final derotated image and 0 elsewhere")
    left_refpix_mask : Optional[FitsConfig] = xconf.field(help="Mask that is 1 where pixels may be included in the reference set and 0 elsewhere")
    right_mask : FitsConfig = xconf.field(help="Mask that is 1 where pixels should be included in the final derotated image and 0 elsewhere")
    right_refpix_mask : Optional[FitsConfig] = xconf.field(help="Mask that is 1 where pixels may be included in the reference set and 0 elsewhere")
    angles : Union[FitsConfig,FitsTableColumnConfig] = xconf.field(help="")
    sampling : SamplingConfig = xconf.field(help="Configure the sampling of the final derotated field for detection and contrast calibration")
    min_coverage_frac : float = xconf.field(help="")
    ring_exclude_px : float = xconf.field(default=12, help="When selecting reference pixel timeseries, determines width of ring centered at radius of interest for which pixel vectors are excluded")
    snr_exclude_nearest_apertures: int = xconf.field(default=1, help="When calculating signal-to-noise from resel_px-spaced 'apertures', exclude this many from each side of the signal aperture")
    resel_px : float = xconf.field(default=8, help="Resolution element in pixels for these data")
    gpu_ram_mb_per_task : Union[float, str, PerTaskConfig, None] = xconf.field(default=None, help="Maximum amount of GPU RAM used by a stage or grid point, or 'measure'")
    max_tasks_per_gpu : float = xconf.field(default=1, help="When GPU is utilized, this is the maximum number of tasks scheduled on the same GPU (RAM permitting)")
    ram_mb_per_task : Union[float, str, PerTaskConfig, None] = xconf.field(default=None, help="Maximum amount of RAM used by a stage or grid point, or 'measure'")
    decompose_on_gpu : bool = xconf.field(default=False, help="")
    evaluate_on_gpu : bool = xconf.field(default=False, help="")
    use_cgls : bool = xconf.field(default=False, help="")
    benchmark : bool = xconf.field(default=False, help="")
    benchmark_trials : int = xconf.field(default=2, help="")
    efficient_decomp_reuse : bool = xconf.field(default=True, help="Use a single decomposition for all PAs at given separation by masking a ring")

    def _initialize_task_options(self) -> tuple[TaskOptions, bool]:
        options = {'resources':{}}
        generate_options = options.copy()
        decompose_options = options.copy()
        evaluate_options = options.copy()
        measure_ram = False
        if isinstance(self.ram_mb_per_task, PerTaskConfig):
            generate_options['memory'] = self.ram_mb_per_task.generate * BYTES_PER_MB
            decompose_options['memory'] = self.ram_mb_per_task.decompose * BYTES_PER_MB
            evaluate_options['memory'] = self.ram_mb_per_task.evaluate * BYTES_PER_MB
        elif self.ram_mb_per_task == "measure":
            measure_ram = True
        elif self.ram_mb_per_task is not None:
            # number or None
            generate_options['memory'] = self.ram_mb_per_task * BYTES_PER_MB
            decompose_options['memory'] = self.ram_mb_per_task * BYTES_PER_MB
            evaluate_options['memory'] = self.ram_mb_per_task * BYTES_PER_MB

        if self.decompose_on_gpu:
            gpu_frac = 1 / self.max_tasks_per_gpu
            log.debug(f"Using {self.max_tasks_per_gpu} tasks per GPU, {gpu_frac=}")
            # generate_options['num_gpus'] = gpu_frac
            decompose_options['num_gpus'] = gpu_frac
            # evaluate_options['num_gpus'] = gpu_frac

            if self.gpu_ram_mb_per_task is None:
                raise RuntimeError(f"Specify GPU RAM per task")
            if isinstance(self.gpu_ram_mb_per_task, PerTaskConfig):
                generate_options['resources'] = {'gpu_memory_mb': self.gpu_ram_mb_per_task.generate}
                decompose_options['resources'] = {'gpu_memory_mb': self.gpu_ram_mb_per_task.decompose}
                evaluate_options['resources'] = {'gpu_memory_mb': self.gpu_ram_mb_per_task.evaluate}
            elif self.gpu_ram_mb_per_task == "measure":
                measure_ram = True
            else:
                # number or None
                generate_options['resources'] = {'gpu_memory_mb': self.gpu_ram_mb_per_task}
                decompose_options['resources'] = {'gpu_memory_mb': self.gpu_ram_mb_per_task}
                evaluate_options['resources'] = {'gpu_memory_mb': self.gpu_ram_mb_per_task}
        return TaskOptions(generate_options, decompose_options, evaluate_options), measure_ram

    def get_output_paths(self, *output_paths):
        self.destination.ensure_exists()
        output_paths = [utils.join(self.destination.path, op) for op in output_paths]
        return output_paths

    def load_coverage_map_or_generate(self, mask, angles):
        from .. import pipelines
        dest_fs = self.destination.get_fs()
        coverage_map_fn = f'coverage_t{self.every_t_frames}.fits'
        covered_pix_mask_fn = f'coverage_t{self.every_t_frames}_{self.min_coverage_frac}.fits'
        coverage_map_path, covered_pix_mask_path = self.get_output_paths(
            coverage_map_fn,
            covered_pix_mask_fn,
        )
        coverage_map_path = utils.join(self.destination.path, coverage_map_fn)
        if not dest_fs.exists(coverage_map_path):
            log.debug(f"Number of pixels retained per frame {np.count_nonzero(mask)=}")
            log.debug(f"Computing coverage map")
            final_coverage = pipelines.adi_coverage(mask, angles)
            iofits.write_fits(iofits.PicklableHDUList([iofits.PicklableHDU(final_coverage)]), coverage_map_path)
            log.debug(f"Wrote coverage map to {coverage_map_path}")
        else:
            final_coverage = iofits.load_fits_from_path(coverage_map_path)[0].data
        n_frames = len(angles)
        covered_pix_mask = final_coverage > int(n_frames * self.min_coverage_frac)
        from skimage.morphology import binary_closing
        covered_pix_mask = binary_closing(covered_pix_mask)
        log.debug(f"Coverage map with {self.min_coverage_frac} fraction gives {np.count_nonzero(covered_pix_mask)} possible pixels to analyze")
        if not dest_fs.exists(covered_pix_mask_path):
            iofits.write_fits(iofits.PicklableHDUList([iofits.PicklableHDU(covered_pix_mask.astype(int))]), covered_pix_mask_path)
            log.debug(f"Wrote covered pix mask to {covered_pix_mask_path}")
        return final_coverage, covered_pix_mask

    def generate_grid(self):
        cols_dtype = [
            ('index', int),
            ('time_total_sec', float),
            ('r_px', float),
            ('pa_deg', float),
            ('x', float),
            ('y', float),
            ('injected_scale', float),
            ('k_modes', int),
            ('pix_used', int),
            ('time_generate_sec', float),
            ('time_decompose_sec', float),
            ('signal', float),
            ('snr', float),
        ]

        left_mask = self.left_mask.load() == 1
        mask_shape = left_mask.shape

        probes = list(characterization.generate_probes(
            self.sampling.iwa_px,
            self.sampling.owa_px,
            self.sampling.n_radii,
            self.sampling.spacing_px,
            self.sampling.scales
        ))
        n_comp_rows = len(self.k_modes_vals) * len(probes)
        log.debug(f"Evaluating {len(probes)} positions/contrast levels at {len(self.k_modes_vals)} k values")
        comp_grid = np.zeros(n_comp_rows, dtype=cols_dtype)
        flattened_idx = 0
        for comp in probes:
            comp_x, comp_y = characterization.r_pa_to_x_y(comp.r_px, comp.pa_deg, *improc.arr_center(mask_shape))
            # for every number of modes:
            for k_modes in self.k_modes_vals:
                comp_grid[flattened_idx]['index'] = flattened_idx
                comp_grid[flattened_idx]['r_px'] = comp.r_px
                comp_grid[flattened_idx]['pa_deg'] = comp.pa_deg
                comp_grid[flattened_idx]['x'] = comp_x
                comp_grid[flattened_idx]['y'] = comp_y
                comp_grid[flattened_idx]['injected_scale'] = comp.scale
                comp_grid[flattened_idx]['k_modes'] = k_modes
                flattened_idx += 1
        return comp_grid

    def compare_grid_to_checkpoint(self, checkpoint_tbl, grid_tbl):
        parameters = ['index', 'r_px', 'pa_deg', 'x', 'y', 'injected_scale', 'k_modes']
        for param in parameters:
            # print(param, 'chk', np.unique(checkpoint_tbl[param]))
            # print(param, 'grid_tbl', np.unique(grid_tbl[param]))
            if not np.allclose(checkpoint_tbl[param], grid_tbl[param]):
                return False
        return True

    def filter_grid(self, tbl):
        if len(self.only_indices):
            indices_mask = np.zeros(len(tbl), dtype=bool)
            for idx in self.only_indices:
                indices_mask |= tbl['index'] == idx
            return tbl[indices_mask]
        mask = np.ones(len(tbl), dtype=bool)
        still_todo = tbl['time_total_sec'] == 0
        dest_fs = self.destination.get_fs()
        gridpoints_dir = utils.join(self.destination.path, "gridpoints")
        if self.save_images and dest_fs.exists(self.destination.path):
            log.debug(f"Checking {self.destination.path} for existing saved images")
            dest_fs.makedirs(gridpoints_dir, exist_ok=True)
            gridpoint_image_paths = dest_fs.ls(gridpoints_dir, detail=False)
            for i in range(len(tbl)):
                point_filepath = utils.join(self.destination.path, "gridpoints", f"point_{int(tbl[i]['index'])}.fits")
                if point_filepath not in gridpoint_image_paths:
                    still_todo[i] = True
        mask &= still_todo
        return tbl[mask]

    def measure_ram(self, task_options, image_vecs_ref, model_inputs_ref, coverage_mask_ref, tbl_row) -> tuple[dict,dict,dict]:
        # Measure injection using first location (assumes at least 1 unique value for
        # injection parameters, even if it's zero)
        log.debug(f"Measuring RAM use in inject_model()")
        injection_ram_mb, injection_gpu_ram_mb = _measure_ram(
            _inject_model,
            task_options.generate,
            image_vecs_ref,
            model_inputs_ref,
            tbl_row['r_px'],
            tbl_row['pa_deg'],
            tbl_row['injected_scale'],
        )
        task_options.generate['memory'] = injection_ram_mb * BYTES_PER_MB
        if injection_gpu_ram_mb > 0:
            resdict = task_options.generate.get('resources', {})
            resdict['gpu_memory_mb'] = injection_gpu_ram_mb
            task_options.generate['resources'] = resdict
        log.debug(f"{task_options.generate=}")

        # End up precomputing twice, but use the same args at least...
        precomp_args = (
            image_vecs_ref,  # no injection needed
            model_inputs_ref,
            0,  # smaller radii -> more reference pixels -> upper bound on time
            0,  # no exclusion -> more pixels
            tbl_row['k_modes'],
            self.decompose_on_gpu,
        )
        # precompute to measure
        log.debug(f"Measuring RAM use in precompute_basis()")
        precomp_ram_mb, precomp_gpu_ram_mb = _measure_ram(
            _precompute_basis,
            task_options.decompose,
            *precomp_args,
            _measure_gpu_ram=self.decompose_on_gpu,
        )
        task_options.decompose['memory'] = precomp_ram_mb * BYTES_PER_MB
        log.debug(f"After setting decompose_options {task_options.generate=} {task_options.decompose=} {task_options.evaluate=}")
        if precomp_gpu_ram_mb > 0:
            resdict = task_options.decompose.get('resources', {})
            resdict['gpu_memory_mb'] = precomp_gpu_ram_mb
            task_options.decompose['resources'] = resdict
        log.debug(f"{task_options.decompose=}")

        # precompute to use as input for evaluate_point
        temp_precompute_ref = precompute_basis.options(**task_options.decompose).remote(*precomp_args)
        log.debug(f"Submitting precompute as input to evaluate_point {temp_precompute_ref=}")
        ray.wait([temp_precompute_ref], fetch_local=False)
        log.debug(f"{temp_precompute_ref=} is ready")

        log.debug(f"Measuring RAM use in evaluate_point() {task_options.evaluate=}")
        ram_requirement_mb, gpu_ram_requirement_mb = _measure_ram(
            _evaluate_point_kt,
            task_options.evaluate,
            tbl_row,
            image_vecs_ref,  # no injection needed
            model_inputs_ref,
            temp_precompute_ref,
            self.resel_px,
            coverage_mask_ref,
            exclude_nearest=self.snr_exclude_nearest_apertures,
            evaluate_on_gpu=self.evaluate_on_gpu,
            _measure_gpu_ram=self.evaluate_on_gpu or (self.decompose_on_gpu and not self.efficient_decomp_reuse),
        )
        task_options.evaluate['memory'] = ram_requirement_mb * BYTES_PER_MB
        if gpu_ram_requirement_mb > 0:
            resdict = task_options.evaluate.get('resources', {})
            resdict['gpu_memory_mb'] = gpu_ram_requirement_mb
            task_options.evaluate['resources'] = resdict
        # ram_mb_per_task = { generate = 10768, decompose = 982, evaluate = 10768 }
        log.info("RAM config:\n\n" + toml.dumps({'ram_mb_per_task': {
            'generate': int(injection_ram_mb),
            'decompose': int(precomp_ram_mb),
            'evaluate': int(ram_requirement_mb),
        }}))
        log.info("GPU RAM config:\n\n" + toml.dumps({'gpu_ram_mb_per_task': {
            'generate': int(injection_gpu_ram_mb),
            'decompose': int(precomp_gpu_ram_mb),
            'evaluate': int(gpu_ram_requirement_mb),
        }}))
        log.debug(f"{task_options.generate=} {task_options.decompose=} {task_options.evaluate=}")
        return task_options

    def launch_grid(self, pending_tbl) -> list:
        hdul = iofits.load_fits(self.input.open())
        log.debug(f"Loaded from {self.input.path}")

        # decimate
        left_cube = hdul[self.left_extname].data[::self.every_t_frames]
        right_cube = hdul[self.right_extname].data[::self.every_t_frames]
        angles = self.angles.load()[::self.every_t_frames]
        left_scales = self.left_scales.load()[::self.every_t_frames]
        right_scales = self.right_scales.load()[::self.every_t_frames]
        log.debug(f"After decimation, {len(angles)} frames remain")

        # templates
        left_template = self.left_template.load()
        right_template = self.right_template.load()

        # get masks
        left_mask = self.left_mask.load() == 1
        right_mask = self.right_mask.load() == 1
        mask = left_mask | right_mask
        final_coverage, covered_pix_mask = self.load_coverage_map_or_generate(mask, angles)
        coverage_mask_ref = ray.put(covered_pix_mask)

        model_inputs = ModelInputs(
            data_cube_shape=left_cube.shape,
            left_template=left_template,
            right_template=right_template,
            left_scales=left_scales,
            right_scales=right_scales,
            angles=angles,
            mask=mask,
        )
        
        task_options, measure_ram = self._initialize_task_options()
        model_inputs_ref = ray.put(model_inputs)
        log.debug(f"Put model inputs into Ray.")

        # hold reused intermediate results in dicts keyed on the grid parameters
        # that affect them
        def key_maker_maker(*columns):
            def key_maker(row):
                # numpy scalars are not hashable, so we convert each one to a float
                return tuple(float(row[colname]) for colname in columns)
            return key_maker
        inject_refs = {}
        inject_key_maker = key_maker_maker('r_px', 'pa_deg', 'injected_scale')
        precompute_refs = {}
        precompute_key_maker = key_maker_maker('r_px', 'pa_deg', 'injected_scale')

        stitched_cube = pipelines.vapp_stitch(left_cube, right_cube, clio.VAPP_PSF_ROTATION_DEG)
        image_vecs = improc.unwrap_cube(stitched_cube, model_inputs.mask)
        del stitched_cube
        image_vecs_ref = ray.put(image_vecs)

        # try to avoid bottlenecking submission on ram measurement
        if measure_ram:
            test_row = pending_tbl[np.argmax(pending_tbl['k_modes'])]
            task_options = self.measure_ram(
                task_options,
                image_vecs_ref,
                model_inputs_ref, 
                coverage_mask_ref,
                test_row,
            )

        max_k_modes = np.max(pending_tbl['k_modes'])
        injected_count, decomposed_count = 0, 0
        remaining_point_refs = []
        for row in pending_tbl:
            inject_key = inject_key_maker(row)
            precompute_key = precompute_key_maker(row)
            if inject_key not in inject_refs:
                if row['injected_scale'] == 0:
                    inject_image_vecs_ref = image_vecs_ref
                else:
                    inject_image_vecs_ref = inject_model.options(**task_options.generate).remote(
                        image_vecs_ref,
                        model_inputs_ref,
                        row['r_px'],
                        row['pa_deg'],
                        row['injected_scale'],
                    )
                inject_refs[inject_key] = inject_image_vecs_ref
                injected_count += 1
            if precompute_key not in precompute_refs:
                precompute_ref = precompute_basis.options(**task_options.decompose).remote(
                    inject_image_vecs_ref,
                    model_inputs_ref,
                    row['r_px'],
                    self.ring_exclude_px,
                    max_k_modes,
                    self.decompose_on_gpu,
                )
                precompute_refs[precompute_key] = precompute_ref
                decomposed_count += 1
            point_ref = evaluate_point_kt.options(**task_options.evaluate).remote(
                row,
                inject_image_vecs_ref,
                model_inputs_ref,
                precompute_ref,
                self.resel_px,
                coverage_mask_ref,
                save_to_dir=self.destination if self.save_images else None,
            )
            remaining_point_refs.append(point_ref)
        log.debug(f"Generated {injected_count} fake-injected datasets, launched {decomposed_count} basis computations")
        log.debug(f"Applying for {len(pending_tbl)} points in the parameter grid")
        return remaining_point_refs
