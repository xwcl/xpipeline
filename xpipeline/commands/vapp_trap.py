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
from xpipeline.commands.base import FitsConfig, FitsTableColumnConfig, CompanionConfig
from xpipeline.tasks import iofits, vapp, improc, starlight_subtraction, characterization
from xpipeline.ref import clio
from xpipeline import pipelines, utils
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
class ContrastProbesConfig:
    n_radii : int = xconf.field(help="Number of steps in radius at which to probe contrast")
    spacing_px : float = xconf.field(help="Spacing in pixels between contrast probes along circle (sets number of probes at radius by 2 * pi * r / spacing)")
    scales : list[float] = xconf.field(help="Probe contrast levels (C = companion / host)")
    iwa_px : float = xconf.field(help="Inner working angle (px)")
    owa_px : float = xconf.field(help="Outer working angle (px)")


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

def generate_probes(config : Optional[ContrastProbesConfig]):
    '''Generator returning CompanionSpec objects for
    radii / PA / contrast scales that cover the region from iwa to owa
    in steps as specificed by `config`

    When config.n_radii == 1, only iwa_px matters
    '''
    if config is None:
        return []
    iwa_px, owa_px = config.iwa_px, config.owa_px
    radii_dpx = (owa_px - iwa_px) / (config.n_radii - 1)
    for i in range(config.n_radii):
        r_px = iwa_px + i * radii_dpx
        circumference = np.pi * 2 * r_px
        n_probes = int(circumference // config.spacing_px)
        angles_ddeg = 360 / n_probes
        for j in range(n_probes):
            pa_deg = j * angles_ddeg
            for scl in config.scales:
                yield characterization.CompanionSpec(r_px, pa_deg, scl)


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

def test_particle(n_sec):
    print(f"Going to sleep for {n_sec} sec")
    time.sleep(5)
    print(f"Done sleeping")

def _measure_ram_for_step(func, *args, measure_gpu_ram=False, **kwargs):
    from memory_profiler import memory_usage
    gpu_prof = utils.CupyRamHook() if measure_gpu_ram else utils.DummyRamHook()
    initial_ram_mb = memory_usage(-1, max_usage=True)
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
    return precomputed_trap_basis
precompute_basis = ray.remote(_precompute_basis)

def _evaluate_point(out_idx, row, inject_image_vecs, model_inputs, k_modes, precomputed_trap_basis, left_pix_vec, force_gpu_fit, use_cgls):
    start = time.perf_counter()
    row = row.copy() # since ray is r/o
    model_vecs = generate_model(model_inputs, row['r_px'], row['pa_deg'])
    params = starlight_subtraction.TrapParams(
        k_modes=k_modes,
        compute_residuals=False,
        precomputed_basis=precomputed_trap_basis,
        background_split_mask=left_pix_vec,
        force_gpu_fit=force_gpu_fit,
        use_cgls=use_cgls,
    )
    model_coeff, timers, pix_used, _ = starlight_subtraction.trap_mtx(inject_image_vecs, model_vecs, params)
    row['model_coeff'] = model_coeff
    row['pix_used'] = pix_used
    row['time_precompute_svd_sec'] = timers['time_svd_sec']
    row['time_invert_sec'] = timers['invert']
    row['time_total_sec'] = time.perf_counter() - start
    log.info(f"Evaluated {out_idx=} in {row['time_total_sec']} sec")
    return out_idx, row
evaluate_point = ray.remote(_evaluate_point)

def _evaluate_point_kt(out_idx, row, inject_image_vecs, model_inputs, k_modes, precomputed_trap_basis, resel_px, coverage_mask, resels_planet=2, resels_ring=2, exclude_nearest=1):
    # just use klip-transpose / ADI / matched filter, no simultaneous fit
    start = time.perf_counter()
    row = row.copy() # since ray is r/o
    companion_r_px, companion_pa_deg = float(row['r_px']), float(row['pa_deg'])  # convert to primitive type so numba doesn't complain
    model_vecs = generate_model(model_inputs, companion_r_px, companion_pa_deg)
    params_kt = starlight_subtraction.TrapParams(
        k_modes=k_modes,
        compute_residuals=True,
        precomputed_basis=precomputed_trap_basis,
    )
    image_resid_vecs, model_resid_vecs, timers, pix_used = starlight_subtraction.klip_transpose(inject_image_vecs, model_vecs, params_kt)
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
    row['time_precompute_svd_sec'] = timers['time_svd_sec']
    row['pix_used'] = pix_used
    row['signal'] = signal
    row['snr'] = snr
    return out_idx, row
evaluate_point_kt = ray.remote(_evaluate_point_kt)

def grid_generate(k_modes_vals, covered_pix_mask, every_n_pix, contrast_probes_config, use_klip_transpose):
    log.debug(f"{contrast_probes_config=}")
    rhos, pa_degs, xx, yy = improc.downsampled_grid_r_pa(covered_pix_mask, every_n_pix)
    # static: every_n_pix, every_t_frames, min_coverage
    # varying: k_modes, inject_r_px, inject_pa_deg, inject_scale, r_px, pa_deg, x, y
    # computed: model_coeff, time_total, time_model, time_decomp, time_fit
    n_rows = len(k_modes_vals) * len(rhos)
    cols_dtype = [
        ('inject_r_px', float),
        ('inject_pa_deg', float),
        ('inject_scale', float),
        ('k_modes', int),
        ('r_px', float),
        ('pa_deg', float),
        ('x', int),
        ('y', int),
        ('time_total_sec', float),
        ('pix_used', int),
        ('time_precompute_svd_sec', float),
    ]
    if use_klip_transpose:
        cols_dtype.extend([
            ('signal', float),
            ('noise', float),
            ('snr', float),
        ])
    else:
        cols_dtype.extend([
            ('model_coeff', float),
            ('time_invert_sec', float),
        ])
    grid = np.zeros(n_rows, dtype=cols_dtype)
    for idx, (k_modes, inner_idx) in enumerate(itertools.product(k_modes_vals, range(len(rhos)))):
        grid[idx]['k_modes'] = k_modes
        grid[idx]['r_px'] = rhos[inner_idx]
        grid[idx]['pa_deg'] = pa_degs[inner_idx]
        grid[idx]['x'] = xx[inner_idx]
        grid[idx]['y'] = yy[inner_idx]

    probes = list(generate_probes(contrast_probes_config))
    # n_comp_rows = 2 * len(k_modes_vals) * len(probes)
    n_comp_rows = len(k_modes_vals) * len(probes)
    log.debug(f"Evaluating {len(probes)} positions/contrast levels at {len(k_modes_vals)} k values and 2 map locations")
    comp_grid = np.zeros(n_comp_rows, dtype=cols_dtype)
    # for every probe location and value for scale:
    for cidx, companion in enumerate(probes):
        comp_x, comp_y = characterization.r_pa_to_x_y(companion.r_px, companion.pa_deg, *improc.arr_center(covered_pix_mask))
        nearest_grid_idx = np.argmin((grid['x'] - comp_x)**2 + (grid['y'] - comp_y)**2)
        nearest_grid_point = grid[nearest_grid_idx]
        # log.debug(f"{nearest_grid_point=}")
        assert nearest_grid_point['inject_scale'] == 0
        # for every number of modes:
        for kidx, k_modes in enumerate(k_modes_vals):
            # grid_idx = cidx * (len(k_modes_vals) * 2) + 2 * kidx
            grid_idx = cidx * len(k_modes_vals) + kidx
            print(f"{grid_idx=} {cidx=} {kidx=}")
            assert comp_grid[grid_idx]['inject_scale'] == 0, "visiting same index more than once in grid gen"

            # - compute fit coefficient for nearest r, pa from overall grid
            comp_grid[grid_idx]['inject_r_px'] = companion.r_px
            comp_grid[grid_idx]['inject_pa_deg'] = companion.pa_deg
            comp_grid[grid_idx]['inject_scale'] = companion.scale
            comp_grid[grid_idx]['k_modes'] = k_modes
            comp_grid[grid_idx]['r_px'] = nearest_grid_point['r_px']
            comp_grid[grid_idx]['pa_deg'] = nearest_grid_point['pa_deg']
            comp_grid[grid_idx]['x'] = nearest_grid_point['x']
            comp_grid[grid_idx]['y'] = nearest_grid_point['y']
            # print(grid_idx, f"{comp_grid[grid_idx]=}")
            # - compute fit coefficient for exact r, pa
            # comp_grid[grid_idx + 1]['inject_r_px'] = companion.r_px
            # comp_grid[grid_idx + 1]['inject_pa_deg'] = companion.pa_deg
            # comp_grid[grid_idx + 1]['inject_scale'] = companion.scale
            # comp_grid[grid_idx + 1]['k_modes'] = k_modes
            # comp_grid[grid_idx + 1]['r_px'] = companion.r_px
            # comp_grid[grid_idx + 1]['pa_deg'] = companion.pa_deg
            # comp_grid[grid_idx + 1]['x'] = comp_x
            # comp_grid[grid_idx + 1]['y'] = comp_y
            # print(grid_idx + 1, f"{comp_grid[grid_idx + 1]=}")
    grid = np.concatenate((grid, comp_grid))
    return grid

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
    outside_time_sec = time.perf_counter()
    log.debug(f"{func} {outside_time_sec=} at start, ref is {measure_ref}")
    ram_usage_mb, gpu_ram_usage_mb, inside_time_sec = ray.get(measure_ref)
    end_time_sec = time.perf_counter()
    log.debug(f"{func} end time {end_time_sec}")
    outside_time_sec = time.perf_counter() - outside_time_sec
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
                measure_ram=False, efficient_decomp_reuse=False, split_bg=False, use_cgls=False,
                use_klip_transpose=False, resel_px=8, coverage_mask=None):
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

    # left_pix_vec is true where vector entries correspond to pixels
    # used from the left half of the image in the final stitched cube
    # for purposes of creating the two background offset terms
    if split_bg:
        left_half, _ = vapp.mask_along_angle(model_inputs.mask.shape, clio.VAPP_PSF_ROTATION_DEG)
        left_pix_vec = improc.unwrap_image(left_half, model_inputs.mask)
    else:
        left_pix_vec = None

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
    inject_key_maker = key_maker_maker('inject_r_px', 'inject_pa_deg', 'inject_scale')
    precompute_refs = {}
    precompute_key_maker = key_maker_maker('inject_r_px', 'inject_pa_deg', 'inject_scale', 'r_px')

    stitched_cube = pipelines.vapp_stitch(left_cube, right_cube, clio.VAPP_PSF_ROTATION_DEG)
    image_vecs = improc.unwrap_cube(stitched_cube, model_inputs.mask)
    del stitched_cube
    image_vecs_ref = ray.put(image_vecs)

    # unique (r, pa, scale) for *injected*
    injection_locs = np.unique(remaining_grid[['inject_r_px', 'inject_pa_deg', 'inject_scale']])
    # unique (inject_r, inject_pa, inject_scale, r, pa) for precomputing basis
    precompute_locs = np.unique(remaining_grid[['inject_r_px', 'inject_pa_deg', 'inject_scale', 'r_px']])
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
            injection_locs[0]['inject_r_px'],
            injection_locs[0]['inject_pa_deg'],
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

        log.debug(f"Measuring RAM use in evaluate_point() {evaluate_options=}")
        if use_klip_transpose:
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
                measure_gpu_ram=force_gpu_decomposition and not efficient_decomp_reuse,
            )
        else:
            ram_requirement_mb, gpu_ram_requirement_mb = _measure_ram(
                _evaluate_point,
                evaluate_options,
                0,
                remaining_grid[0],
                image_vecs_ref,  # no injection needed
                model_inputs_ref,
                max_k_modes,
                temp_precompute_ref,
                left_pix_vec,
                force_gpu_fit,
                use_cgls,
                measure_gpu_ram=force_gpu_fit or not efficient_decomp_reuse,
            )
        evaluate_options['memory'] = ram_requirement_mb * BYTES_PER_MB
        if gpu_ram_requirement_mb > 0:
            resdict = evaluate_options.get('resources', {})
            resdict['gpu_memory_mb'] = gpu_ram_requirement_mb
            evaluate_options['resources'] = resdict
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
                row['inject_r_px'],
                row['inject_pa_deg'],
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
        if use_klip_transpose:
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
        else:
            point_ref = evaluate_point.options(**evaluate_options).remote(
                idx,
                row,
                inject_ref,
                model_inputs_ref,
                k_modes,
                precompute_ref,
                left_pix_vec,
                force_gpu_fit,
                use_cgls,
            )
        remaining_point_refs.append(point_ref)
    log.debug(f"Applying TRAP++ for {len(remaining_idxs)} points in the parameter grid")
    return remaining_point_refs

@xconf.config
class VappTrap(xconf.Command):
    every_n_pix : int = xconf.field(default=1, help="Evaluate a forward model centered on every Nth pixel")
    every_t_frames : int = xconf.field(default=1, help="Use every Tth frame as the input cube")
    k_modes_vals : list[int] = xconf.field(default_factory=lambda: [15, 100], help="")
    dataset : str = xconf.field(help="")
    left_extname : FITS_EXT = xconf.field(help="")
    left_template : FitsConfig = xconf.field(help="")
    right_template : FitsConfig = xconf.field(help="")
    right_extname : FITS_EXT = xconf.field(help="")
    left_scales : FitsConfig = xconf.field(help="")
    right_scales : FitsConfig = xconf.field(help="")
    left_mask : FitsConfig = xconf.field(help="")
    right_mask : FitsConfig = xconf.field(help="")
    angles : Union[FitsConfig,FitsTableColumnConfig] = xconf.field(help="")
    contrast_probes : Optional[ContrastProbesConfig] = xconf.field(default=None, help="")
    # companions : list[CompanionConfig] = xconf.field(default_factory=list, help="")
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
    use_klip_transpose : bool = xconf.field(default=False, help="")
    benchmark : bool = xconf.field(default=False, help="")
    benchmark_trials : int = xconf.field(default=2, help="")
    split_bg_model : bool = xconf.field(default=True, help="Use split BG model along clio symmetry ax for vAPP")
    efficient_decomp_reuse : bool = xconf.field(default=True, help="Use a single decomposition for all PAs at given separation by masking a ring")
    # ray_url : Optional[str] = xconf.field(help="")
    checkpoint_every_x : int = xconf.field(default=10, help="Write current state of grid to disk every X iterations")
    # max_jobs_chunk_size : int = xconf.field(default=1, help="Number of grid points to submit as single Ray task")

    def main(self):
        hdul = iofits.load_fits_from_path(self.dataset)
        log.debug(f"Loaded from {self.dataset}")

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
        ray_init_kwargs = {'runtime_env': {'env_vars': {'RAY_USER_SETUP_FUNCTION': 'xpipeline.commands.vapp_trap.init_worker'}}}
        if isinstance(self.ray, RemoteRayConfig):
            ray.init(self.ray.url, **ray_init_kwargs)
        else:
            resources = {}
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

        if self.use_gpu_decomposition or self.use_gpu_fit:
            gpu_frac = 1 / self.max_tasks_per_gpu
            log.debug(f"Using {self.max_tasks_per_gpu} tasks per GPU, {gpu_frac=}")
            # generate_options['num_gpus'] = gpu_frac
            decompose_options['num_gpus'] = gpu_frac
            evaluate_options['num_gpus'] = gpu_frac

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

        coverage_map_path = f'./coverage_t{self.every_t_frames}.fits'
        if not os.path.exists(coverage_map_path):
            log.debug(f"Computing coverage map")
            final_coverage = pipelines.adi_coverage(mask, angles)
            fits.PrimaryHDU(final_coverage).writeto(coverage_map_path, overwrite=True)
        else:
            final_coverage = fits.getdata(coverage_map_path)

        n_frames = len(angles)
        covered_pix_mask = final_coverage > int(n_frames * self.min_coverage_frac)
        from skimage.morphology import binary_closing
        covered_pix_mask = binary_closing(covered_pix_mask)
        log.debug(f"Coverage map with {self.min_coverage_frac} fraction gives {np.count_nonzero(covered_pix_mask)} possible pixels to analyze")
        fits.PrimaryHDU(covered_pix_mask.astype(int)).writeto(coverage_map_path.replace('.fits', f'_{self.min_coverage_frac}.fits'), overwrite=True)

        log.debug("Generating grid")
        grid = grid_generate(
            self.k_modes_vals,
            covered_pix_mask,
            self.every_n_pix,
            self.contrast_probes,
            self.use_klip_transpose,
        )
        if not self.benchmark:
            try:
                # load checkpoint
                with open("./grid.fits", 'rb') as fh:
                    hdul = fits.open(fh)
                    grid = np.asarray(hdul['grid'].data)
                log.debug(f"Loaded checkpoint successfully")
            except FileNotFoundError:
                with open("./grid.fits", 'wb') as fh:
                    hdu = fits.BinTableHDU(grid, name='grid')
                    hdu.writeto('./grid.fits')
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
            split_bg=self.split_bg_model,
            use_cgls=self.use_cgls,
            use_klip_transpose=self.use_klip_transpose,
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
        total = len(result_refs)
        n_complete = 0
        while pending:
            complete, pending = ray.wait(pending, timeout=5, num_returns=min(self.checkpoint_every_x, len(pending)))
            for (idx, result) in ray.get(complete):
                grid[idx] = result
            if len(complete):
                n_complete += len(complete)
                dt = time.time() - start_time
                complete_per_sec = n_complete / dt
                sec_per_point = 1/complete_per_sec
                est_remaining = sec_per_point * (total - n_complete)
                log.debug(f"{n_complete=} / {total=} ({complete_per_sec=} {sec_per_point=}, {est_remaining=} sec or {est_remaining/60} min)")
                with open("./grid.fits~", 'wb') as fh:
                    hdu = fits.BinTableHDU(grid, name='grid')
                    hdu.writeto(fh)
                    shutil.move("./grid.fits~", "./grid.fits")
        ray.shutdown()
