from re import M
import toml
import time
from astropy.convolution import convolve_fft
from dataclasses import dataclass
from xpipeline.core import cupy as cp
import itertools
import shutil
import numpy as np
import xconf
import ray
import os.path
from astropy.io import fits
from typing import Optional, Union
from xpipeline.types import FITS_EXT
from xpipeline.commands.base import FitsConfig, FitsTableColumnConfig, CompanionConfig, InputCommand
from xpipeline.tasks import iofits, vapp, improc, starlight_subtraction, characterization
from xpipeline.ref import clio
from xpipeline import pipelines, utils
from tqdm import tqdm
import logging
log = logging.getLogger(__name__)
BYTES_PER_MB = 1024 * 1024
@xconf.config
class CommonRayConfig:
    env_vars : Optional[dict[str, str]] = xconf.field(default_factory=dict, help="Environment variables to set for worker processes")
@xconf.config
class LocalRayConfig(CommonRayConfig):
    cpus : Optional[int] = xconf.field(default=None, help="CPUs available to built-in Ray cluster (default is auto-detected)")
    gpus : Optional[int] = xconf.field(default=None, help="GPUs available to built-in Ray cluster (default is auto-detected)")
    resources : Optional[dict[str, float]] = xconf.field(default_factory=dict, help="Node resources available when running in standalone mode")
@xconf.config
class RemoteRayConfig(CommonRayConfig):
    url : str = xconf.field(help="URL to existing Ray cluster head node")

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
class ModelInputs:
    data_cube_shape : tuple[int,int,int]
    left_template : np.ndarray
    right_template : np.ndarray
    left_scales : np.ndarray
    right_scales : np.ndarray
    angles : np.ndarray
    mask : np.ndarray

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


def _precompute_basis(image_vecs, model_inputs : ModelInputs, r_px, ring_exclude_px, k_modes_max, force_gpu_decomposition):
    rho, _ = improc.polar_coords(improc.arr_center(model_inputs.mask), model_inputs.mask.shape)
    included_pix_mask = np.abs(rho - r_px) > ring_exclude_px / 2
    included_vecs_mask = improc.unwrap_image(included_pix_mask, model_inputs.mask)
    assert included_vecs_mask.shape[0] == image_vecs.shape[0]
    ref_vecs = image_vecs[included_vecs_mask]
    params = starlight_subtraction.TrapParams(
        k_modes=k_modes_max,
        force_gpu_decomposition=force_gpu_decomposition,
        return_basis=True,
    )
    precomputed_trap_basis = starlight_subtraction.trap_phase_1(ref_vecs, params)
    if force_gpu_decomposition:
        precomputed_trap_basis.temporal_basis = precomputed_trap_basis.temporal_basis.get()
    return precomputed_trap_basis
precompute_basis = ray.remote(_precompute_basis)

def _evaluate_point_kt(
        out_idx, 
        row, 
        inject_image_vecs, 
        model_inputs, 
        k_modes, 
        precomputed_trap_basis, 
        resel_px, 
        coverage_mask, 
        exclude_nearest=1, 
        save_images=False, 
        force_gpu_fit=False
    ):
    # just use klip-transpose / ADI / matched filter, no simultaneous fit
    print(f"evaluate_point_kt start {time.time()=}")
    log.info(f"evaluate_point_kt start {time.time()=}")
    start = time.perf_counter()
    row = row.copy() # since ray is r/o
    companion_r_px, companion_pa_deg = float(row['r_px']), float(row['pa_deg'])  # convert to primitive type so numba doesn't complain
    model_gen_sec = time.perf_counter()
    model_vecs = generate_model(model_inputs, companion_r_px, companion_pa_deg)
    model_gen_sec = model_gen_sec - time.perf_counter()
    params_kt = starlight_subtraction.TrapParams(
        k_modes=k_modes,
        compute_residuals=True,
        precomputed_basis=precomputed_trap_basis,
    )
    if force_gpu_fit:
        precomputed_trap_basis.temporal_basis = cp.asarray(precomputed_trap_basis.temporal_basis)
        inject_image_vecs = cp.asarray(inject_image_vecs)
        model_vecs = cp.asarray(model_vecs)
    image_resid_vecs, model_resid_vecs, timers, pix_used = starlight_subtraction.klip_transpose(inject_image_vecs, model_vecs, params_kt)
    if force_gpu_fit:
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

    if save_images:
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(finim, name='finim'),
            fits.ImageHDU(mf_finim, name='mf_finim'),
            fits.ImageHDU(mf_ctr, name='mf_ctr'),
            fits.ImageHDU(fltrd, name='fltrd')
            # fits.ImageHDU(planet_mask.astype(int), name='planet_mask'),
            # fits.ImageHDU(ring_mask.astype(int), name='ring_mask'),
        ])
        import os
        os.makedirs('./gridpoints/', exist_ok=True)
        outfile = f'./gridpoints/point_{int(out_idx)}.fits'
        hdul.writeto(outfile, overwrite=True)
        print(os.path.abspath(outfile))
    row['time_total_sec'] = time.perf_counter() - start
    row['time_generate_sec'] = time.perf_counter() - model_gen_sec
    row['time_decompose_sec'] = timers['time_svd_sec']
    row['pix_used'] = pix_used
    row['signal'] = signal
    row['snr'] = snr
    print(f"evaluate_point_kt end {time.time()=}")
    log.info(f"evaluate_point_kt end {time.time()=}")
    return out_idx, row
evaluate_point_kt = ray.remote(_evaluate_point_kt)

def grid_generate(k_modes_vals, mask_shape, sampling_config : SamplingConfig):
    cols_dtype = [
        ('r_px', float),
        ('pa_deg', float),
        ('x', float),
        ('y', float),
        ('injected_scale', float),
        ('k_modes', int),
        ('pix_used', int),
        ('time_total_sec', float),
        ('time_generate_sec', float),
        ('time_decompose_sec', float),
        ('signal', float),
        ('snr', float),
    ]

    probes = list(characterization.generate_probes(
        sampling_config.iwa_px,
        sampling_config.owa_px,
        sampling_config.n_radii,
        sampling_config.spacing_px,
        sampling_config.scales
    ))
    n_comp_rows = len(k_modes_vals) * len(probes)
    log.debug(f"Evaluating {len(probes)} positions/contrast levels at {len(k_modes_vals)} k values")
    comp_grid = np.zeros(n_comp_rows, dtype=cols_dtype)
    flattened_idx = 0
    for idx, comp in enumerate(probes):
        comp_x, comp_y = characterization.r_pa_to_x_y(comp.r_px, comp.pa_deg, *improc.arr_center(mask_shape))
        # for every number of modes:
        for k_modes in k_modes_vals:
            comp_grid[flattened_idx]['r_px'] = comp.r_px
            comp_grid[flattened_idx]['pa_deg'] = comp.pa_deg
            comp_grid[flattened_idx]['x'] = comp_x
            comp_grid[flattened_idx]['y'] = comp_y
            comp_grid[flattened_idx]['inject_scale'] = comp.scale
            comp_grid[flattened_idx]['k_modes'] = k_modes
            flattened_idx += 1
    return comp_grid

def init_worker():
    import matplotlib
    matplotlib.use("Agg")
    from xpipeline.cli import Dispatcher
    Dispatcher.configure_logging(None, 'INFO')
    log.info(f"Worker logging initalized")

def _measure_ram(func, options, *args, ram_pad_factor=1.1, measure_gpu_ram=False, **kwargs):
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


def launch_grid(grid,
                left_cube, right_cube,
                model_inputs, ring_exclude_px,
                force_gpu_decomposition, force_gpu_fit,
                generate_options=None, decompose_options=None, evaluate_options=None,
                measure_ram=False, efficient_decomp_reuse=False, resel_px=8, coverage_mask=None):
    if generate_options is None:
        generate_options = {}
    if decompose_options is None:
        decompose_options = {}
    if evaluate_options is None:
        evaluate_options = {}
    # filter points that have already been evaluated out
    # but keep the indices into the full grid for later
    remaining_idxs = np.argwhere(grid['time_total_sec'] == 0)
    n_remaining = len(remaining_idxs)
    if n_remaining == 0:
        log.info(f"All grid entries processed, nothing to do")
        return []
    elif n_remaining > 0:
        log.info(f"Loaded {len(grid) - n_remaining} completed grid points, {n_remaining} left to go")
    remaining_grid = grid[remaining_idxs]

    # precomputation will use the max number of modes, computed once, trimmed as needed
    log.debug(f"{np.unique(remaining_grid['k_modes'])=}")
    max_k_modes = int(max(remaining_grid['k_modes']))

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
    inject_key_maker = key_maker_maker('r_px', 'pa_deg', 'inject_scale')
    precompute_refs = {}
    precompute_key_maker = key_maker_maker('r_px', 'pa_deg', 'inject_scale')

    stitched_cube = pipelines.vapp_stitch(left_cube, right_cube, clio.VAPP_PSF_ROTATION_DEG)
    image_vecs = improc.unwrap_cube(stitched_cube, model_inputs.mask)
    del stitched_cube
    image_vecs_ref = ray.put(image_vecs)

    # unique (r, pa, scale) for *injected*
    injection_locs = np.unique(remaining_grid[['r_px', 'pa_deg', 'inject_scale']])
    # unique (target r, target pa, injection scale) for precomputing basis
    precompute_locs = np.unique(remaining_grid[['r_px', 'pa_deg', 'inject_scale']])

    # try to avoid bottlenecking submission on ram measurement
    if measure_ram:
        # Measure injection using first location (assumes at least 1 unique value for
        # injection parameters, even if it's zero)
        log.debug(f"Measuring RAM use in inject_model()")
        injection_ram_mb, injection_gpu_ram_mb = _measure_ram(
            _inject_model,
            generate_options,
            image_vecs_ref,
            model_inputs_ref,
            injection_locs[0]['r_px'],
            injection_locs[0]['pa_deg'],
            injection_locs[0]['inject_scale'],
        )
        generate_options['memory'] = injection_ram_mb * BYTES_PER_MB
        if injection_gpu_ram_mb > 0:
            resdict = generate_options.get('resources', {})
            resdict['gpu_memory_mb'] = injection_gpu_ram_mb
            generate_options['resources'] = resdict
        log.debug(f"{generate_options=}")

        # End up precomputing twice, but use the same args at least...
        precomp_args = (
            image_vecs_ref,  # no injection needed
            model_inputs_ref,
            0,  # smaller radii -> more reference pixels -> upper bound on time
            0,  # no exclusion -> more pixels
            max_k_modes,
            force_gpu_decomposition,
        )
        # precompute to measure
        log.debug(f"Measuring RAM use in precompute_basis()")
        precomp_ram_mb, precomp_gpu_ram_mb = _measure_ram(
            _precompute_basis,
            decompose_options,
            *precomp_args,
            measure_gpu_ram=force_gpu_decomposition,
        )
        decompose_options['memory'] = precomp_ram_mb * BYTES_PER_MB
        log.debug(f"After setting decompose_options {generate_options=} {decompose_options=} {evaluate_options=}")
        if precomp_gpu_ram_mb > 0:
            resdict = decompose_options.get('resources', {})
            resdict['gpu_memory_mb'] = precomp_gpu_ram_mb
            decompose_options['resources'] = resdict
        log.debug(f"{decompose_options=}")

        # precompute to use as input for evaluate_point
        temp_precompute_ref = precompute_basis.options(**decompose_options).remote(*precomp_args)
        log.debug(f"Submitting precompute as input to evaluate_point {temp_precompute_ref=}")
        ray.wait([temp_precompute_ref], fetch_local=False)
        log.debug(f"{temp_precompute_ref=} is ready")

        log.debug(f"Measuring RAM use in evaluate_point() {evaluate_options=}")
        ram_requirement_mb, gpu_ram_requirement_mb = _measure_ram(
            _evaluate_point_kt,
            evaluate_options,
            0,
            remaining_grid[0],
            image_vecs_ref,  # no injection needed
            model_inputs_ref,
            max_k_modes,
            temp_precompute_ref,
            resel_px,
            coverage_mask,
            force_gpu_fit=force_gpu_fit,
            measure_gpu_ram=force_gpu_fit or (force_gpu_decomposition and not efficient_decomp_reuse),
        )
        evaluate_options['memory'] = ram_requirement_mb * BYTES_PER_MB
        if gpu_ram_requirement_mb > 0:
            resdict = evaluate_options.get('resources', {})
            resdict['gpu_memory_mb'] = gpu_ram_requirement_mb
            evaluate_options['resources'] = resdict
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
        log.debug(f"{generate_options=} {decompose_options=} {evaluate_options=}")

    made = 0
    for row in injection_locs:
        inject_key = inject_key_maker(row)
        if row['inject_scale'] == 0:
            inject_refs[inject_key] = image_vecs_ref
        else:
            inject_refs[inject_key] = inject_model.options(**generate_options).remote(
                image_vecs_ref,
                model_inputs_ref,
                row['r_px'],
                row['pa_deg'],
                row['inject_scale'],
            )
            made += 1
    log.debug(f"Generating {made} datasets with model planet injection")

    if efficient_decomp_reuse:
        for row in precompute_locs:
            # submit initial_decomposition with max_k_modes
            precompute_key = precompute_key_maker(row)
            inject_key = inject_key_maker(row)
            inject_image_vecs_ref = inject_refs[inject_key]

            precompute_refs[precompute_key] = precompute_basis.options(**decompose_options).remote(
                inject_image_vecs_ref,
                model_inputs_ref,
                row['r_px'],
                ring_exclude_px,
                max_k_modes,
                force_gpu_decomposition,
            )
        log.debug(f"Precomputing {len(precompute_locs)} basis sets with {max_k_modes=}")

    remaining_point_refs = []

    for idx in remaining_idxs:
        row = grid[idx]
        inject_key = inject_key_maker(row)
        inject_ref = inject_refs[inject_key]
        if efficient_decomp_reuse:
            precompute_key = precompute_key_maker(row)
            precompute_ref = precompute_refs[precompute_key]
        else:
            precompute_ref = None
        k_modes = int(row['k_modes'])
        point_ref = evaluate_point_kt.options(**evaluate_options).remote(
            idx,
            row,
            inject_ref,
            model_inputs_ref,
            k_modes,
            precompute_ref,
            resel_px,
            coverage_mask,
        )
        remaining_point_refs.append(point_ref)
    log.debug(f"Applying for {len(remaining_idxs)} points in the parameter grid")
    return remaining_point_refs

@xconf.config
class VappTrap(InputCommand):
    checkpoint : str = xconf.field(default=None, help="Save checkpoints to this path, and/or resume grid from this checkpoint (no verification of parameters used to generate the grid is performed)")
    checkpoint_every_x : int = xconf.field(default=10, help="Write current state of grid to disk every X iterations")
    every_t_frames : int = xconf.field(default=1, help="Use every Tth frame as the input cube")
    k_modes_vals : list[int] = xconf.field(default_factory=lambda: [15, 100], help="")
    left_extname : FITS_EXT = xconf.field(help="")
    left_template : FitsConfig = xconf.field(help="")
    right_template : FitsConfig = xconf.field(help="")
    right_extname : FITS_EXT = xconf.field(help="")
    left_scales : FitsConfig = xconf.field(help="")
    right_scales : FitsConfig = xconf.field(help="")
    left_mask : FitsConfig = xconf.field(help="")
    right_mask : FitsConfig = xconf.field(help="")
    angles : Union[FitsConfig,FitsTableColumnConfig] = xconf.field(help="")
    sampling : SamplingConfig = xconf.field(help="Configure the sampling of the final derotated field for detection and contrast calibration")
    min_coverage_frac : float = xconf.field(help="")
    ray : Union[LocalRayConfig,RemoteRayConfig] = xconf.field(
        default=LocalRayConfig(),
        help="Ray distributed framework configuration"
    )
    ring_exclude_px : float = xconf.field(default=12, help="When selecting reference pixel timeseries, determines width of ring centered at radius of interest for which pixel vectors are excluded")
    resel_px : float = xconf.field(default=8, help="Resolution element in pixels for these data")
    gpu_ram_mb_per_task : Union[float, str, PerTaskConfig, None] = xconf.field(default=None, help="Maximum amount of GPU RAM used by a stage or grid point, or 'measure'")
    max_tasks_per_gpu : float = xconf.field(default=1, help="When GPU is utilized, this is the maximum number of tasks scheduled on the same GPU (RAM permitting)")
    ram_mb_per_task : Union[float, str, PerTaskConfig, None] = xconf.field(default=None, help="Maximum amount of RAM used by a stage or grid point, or 'measure'")
    use_gpu_decomposition : bool = xconf.field(default=False, help="")
    use_gpu_fit : bool = xconf.field(default=False, help="")
    use_cgls : bool = xconf.field(default=False, help="")
    benchmark : bool = xconf.field(default=False, help="")
    benchmark_trials : int = xconf.field(default=2, help="")
    efficient_decomp_reuse : bool = xconf.field(default=True, help="Use a single decomposition for all PAs at given separation by masking a ring")

    def main(self):
        dest_fs = self.get_dest_fs()
        coverage_map_fn = f'coverage_t{self.every_t_frames}.fits'
        covered_pix_mask_fn = f'coverage_t{self.every_t_frames}_{self.min_coverage_frac}.fits'
        output_filepath, coverage_map_path, covered_pix_mask_path = self.get_output_paths(
            "grid.fits",
            coverage_map_fn,
            covered_pix_mask_fn,
        )

        hdul = iofits.load_fits_from_path(self.input)
        log.debug(f"Loaded from {self.input}")

        # decimate
        left_cube = hdul[self.left_extname].data[::self.every_t_frames]
        right_cube = hdul[self.right_extname].data[::self.every_t_frames]
        angles = self.angles.load()[::self.every_t_frames]
        left_scales = self.left_scales.load()[::self.every_t_frames]
        right_scales = self.right_scales.load()[::self.every_t_frames]
        log.debug(f"After decimation, {len(angles)} frames remain")

        # get masks
        left_mask = self.left_mask.load() == 1
        right_mask = self.right_mask.load() == 1
        mask = left_mask | right_mask
        log.debug(f"{np.count_nonzero(mask)=}")

        # templates
        left_template = self.left_template.load()
        right_template = self.right_template.load()

        model_inputs = ModelInputs(
            data_cube_shape=left_cube.shape,
            left_template=left_template,
            right_template=right_template,
            left_scales=left_scales,
            right_scales=right_scales,
            angles=angles,
            mask=mask,
        )

        # init ray
        ray_init_kwargs = {'runtime_env': {
            'env_vars': {
                'RAY_USER_SETUP_FUNCTION': 'xpipeline.commands.vapp_trap.init_worker',
                'MKL_NUM_THREADS': '1',
                'OMP_NUM_THREADS': '1',
                'NUMBA_NUM_THREADS': '1',
            }}
        }
        if isinstance(self.ray, RemoteRayConfig):
            ray.init(self.ray.url, **ray_init_kwargs)
        else:
            ray.init(
                num_cpus=self.ray.cpus,
                num_gpus=self.ray.gpus,
                resources=self.ray.resources,
                **ray_init_kwargs
            )
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

        if self.use_gpu_decomposition:
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


        if not dest_fs.exists(coverage_map_path):
            log.debug(f"Computing coverage map")
            final_coverage = pipelines.adi_coverage(mask, angles)
            iofits.write_fits(iofits.DaskHDUList([iofits.DaskHDU(final_coverage)]), coverage_map_path)
            log.debug(f"Wrote coverage map to {coverage_map_path}")
        else:
            final_coverage = iofits.load_fits_from_path(coverage_map_path)[0].data

        n_frames = len(angles)
        covered_pix_mask = final_coverage > int(n_frames * self.min_coverage_frac)
        from skimage.morphology import binary_closing
        covered_pix_mask = binary_closing(covered_pix_mask)
        log.debug(f"Coverage map with {self.min_coverage_frac} fraction gives {np.count_nonzero(covered_pix_mask)} possible pixels to analyze")
        if not dest_fs.exists(covered_pix_mask_path):
            iofits.write_fits(iofits.DaskHDUList([iofits.DaskHDU(covered_pix_mask.astype(int))]), covered_pix_mask_path)
            log.debug(f"Wrote covered pix mask to {covered_pix_mask_path}")

        log.debug("Generating grid")
        grid = grid_generate(
            self.k_modes_vals,
            covered_pix_mask.shape,
            self.sampling,
        )
        if not self.benchmark:
            if self.checkpoint is not None and utils.get_fs(self.checkpoint).exists(self.checkpoint):
                try:
                    hdul = iofits.load_fits_from_path(self.checkpoint)
                    grid = np.asarray(hdul['grid'].data)
                    log.debug(f"Loaded checkpoint successfully")
                except Exception as e:
                    log.exception("Checkpoint loading failed, starting with empty grid")
        else:
            # select most costly points
            bench_mask = grid['k_modes'] == max(self.k_modes_vals)
            grid = grid[bench_mask][:self.benchmark_trials]

        # submit tasks for every grid point
        start_time = time.time()
        result_refs = launch_grid(
            grid,
            left_cube,
            right_cube,
            model_inputs,
            self.ring_exclude_px,
            self.use_gpu_decomposition,
            self.use_gpu_fit,
            generate_options=generate_options,
            decompose_options=decompose_options,
            evaluate_options=evaluate_options,
            measure_ram=measure_ram,
            efficient_decomp_reuse=self.efficient_decomp_reuse,
            resel_px=self.resel_px,
            coverage_mask=covered_pix_mask,
        )
        if self.benchmark:
            log.debug(f"Running {self.benchmark_trials} trials...")
            for i in range(self.benchmark_trials):
                print(ray.get(result_refs[i]))
            ray.shutdown()
            return 0

        # Wait for results, checkpointing as we go
        pending = result_refs
        total = len(grid)
        restored_from_checkpoint = np.count_nonzero(grid['time_total_sec'] != 0)

        def make_hdul(grid):
            return iofits.DaskHDUList([
                iofits.DaskHDU(None, kind="primary"),
                iofits.DaskHDU(grid, kind="bintable", name="grid")
            ])

        with tqdm(total=total) as pbar:
            pbar.update(restored_from_checkpoint)
            while pending:
                complete, pending = ray.wait(pending, timeout=5, num_returns=min(self.checkpoint_every_x, len(pending)))
                for (idx, result) in ray.get(complete):
                    grid[idx] = result
                if len(complete):
                    pbar.update(len(complete))
                    if self.checkpoint is not None:
                        iofits.write_fits(make_hdul(grid), self.checkpoint, overwrite=True)
        iofits.write_fits(make_hdul(grid), output_filepath, overwrite=True)
        ray.shutdown()
