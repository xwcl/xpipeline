import time
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

@xconf.config
class LocalRayConfig:
    cpus : Optional[int] = xconf.field(default=None, help="CPUs available to built-in Ray cluster (default is auto-detected)")
    gpus : Optional[int] = xconf.field(default=None, help="GPUs available to built-in Ray cluster (default is auto-detected)")
    ram_gb : Optional[float] = xconf.field(default=None, help="RAM available to built-in Ray cluster")

@xconf.config
class RemoteRayConfig:
    url : str = xconf.field(help="URL to existing Ray cluster head node")

@xconf.config
class PerTaskConfig:
    generate : float = xconf.field(help="amount required in model generation")
    decompose : float = xconf.field(help="amount required in basis computation")
    evaluate : float = xconf.field(help="amount required in fitting")



def _generate_model(cube_shape, left_template, right_template,
                   left_scales, right_scales, angles, companion_r_px, companion_pa_deg, mask):
    companion_spec = characterization.CompanionSpec(companion_r_px, companion_pa_deg, 1.0)
    # generate
    left_model_cube = characterization.generate_signals(cube_shape, [companion_spec], left_template, angles, left_scales)
    right_model_cube = characterization.generate_signals(cube_shape, [companion_spec], right_template, angles, right_scales)
    # stitch
    out_cube = pipelines.vapp_stitch(left_model_cube, right_model_cube, clio.VAPP_PSF_ROTATION_DEG)
    model_vecs, _ = improc.unwrap_cube(out_cube, mask)
    return model_vecs
generate_model = ray.remote(_generate_model)

def _inject_model(image_vecs, model_vecs, companion_scale):
    return image_vecs + companion_scale * model_vecs
inject_model = ray.remote(_inject_model)

def _precompute_basis(image_vecs, model_vecs, k_modes_max, force_gpu_decomposition):
    params = starlight_subtraction.TrapParams(
        k_modes=k_modes_max,
        force_gpu_decomposition=force_gpu_decomposition,
        return_basis=True,
    )
    precomputed_trap_basis = starlight_subtraction.trap_mtx(image_vecs, model_vecs, params)
    return precomputed_trap_basis
precompute_basis = ray.remote(_precompute_basis)

def _evaluate_point(out_idx, row, inject_image_vecs, model_vecs, k_modes, precomputed_trap_basis, left_pix_vec, force_gpu_inversion):
    start = time.perf_counter()
    row = row.copy() # since ray is r/o
    params = starlight_subtraction.TrapParams(
        k_modes=k_modes,
        compute_residuals=False,
        precomputed_basis=precomputed_trap_basis,
        background_split_mask=left_pix_vec,
        force_gpu_inversion=force_gpu_inversion,
    )
    model_coeff, timers, pix_used, _ = starlight_subtraction.trap_mtx(inject_image_vecs, model_vecs, params)
    row['model_coeff'] = model_coeff
    row['pix_used'] = pix_used
    row['time_precompute_svd_sec'] = precomputed_trap_basis.time_sec
    row['time_invert_sec'] = timers['invert']
    row['time_total_sec'] = time.perf_counter() - start
    log.info(f"Evaluated {out_idx=} in {row['time_total_sec']} sec")
    return out_idx, row
evaluate_point = ray.remote(_evaluate_point)

def grid_generate(k_modes_vals, covered_pix_mask, every_n_pix, companions):
    # static: every_n_pix, every_t_frames, min_coverage
    # varying: k_modes, inject_r_px, inject_pa_deg, inject_scale, r_px, pa_deg, x, y
    # computed: model_coeff, time_total, time_model, time_decomp, time_fit
    companions.append(characterization.CompanionSpec(0, 0, 0))
    rhos, pa_degs, xx, yy = improc.downsampled_grid_r_pa(covered_pix_mask, every_n_pix)
    grid = np.zeros(len(companions) * len(k_modes_vals) * len(rhos), dtype=[
        ('inject_r_px', float),
        ('inject_pa_deg', float),
        ('inject_scale', float),
        ('k_modes', int),
        ('r_px', float),
        ('pa_deg', float),
        ('x', int),
        ('y', int),
        ('pix_used', int),
        ('model_coeff', float),
        ('time_total_sec', float),
        # ('time_gen_model_sec', float),
        ('time_precompute_svd_sec', float),
        ('time_invert_sec', float),
    ])

    for idx, (companion, k_modes, inner_idx) in enumerate(itertools.product(companions, k_modes_vals, range(len(rhos)))):
        grid[idx]['k_modes'] = k_modes
        grid[idx]['inject_r_px'] = companion.r_px
        grid[idx]['inject_pa_deg'] = companion.pa_deg
        grid[idx]['inject_scale'] = companion.scale
        grid[idx]['r_px'] = rhos[inner_idx]
        grid[idx]['pa_deg'] = pa_degs[inner_idx]
        grid[idx]['x'] = xx[inner_idx]
        grid[idx]['y'] = yy[inner_idx]
    return grid

def worker_init(num_cpus):
    import matplotlib
    matplotlib.use("Agg")
    import mkl
    mkl.set_num_threads(num_cpus)
    import numba
    numba.set_num_threads(num_cpus)
    from xpipeline.core import torch, HAVE_TORCH
    if HAVE_TORCH:
        torch.set_num_threads(num_cpus)
    from xpipeline.cli import Dispatcher
    Dispatcher.configure_logging(None, 'INFO')

def launch_grid(grid, 
                left_cube, right_cube,
                left_template, right_template,
                left_scales, right_scales,
                angles, mask,
                force_gpu_decomposition, force_gpu_inversion,
                generate_options=None, decompose_options=None, evaluate_options=None):
    if generate_options is None:
        generate_options = {}
    if decompose_options is None:
        decompose_options = {}
    if evaluate_options is None:
        evaluate_options = {}
    # filter points that have already been evaluated out
    # but keep the indices into the full grid for later
    remaining_idxs = np.argwhere(grid['time_total_sec'] == 0)
    remaining_grid = grid[remaining_idxs]

    # precomputation will use the max number of modes, computed once, trimmed as needed
    max_k_modes = int(max(remaining_grid['k_modes']))

    # left_pix_vec is true where vector entries correspond to pixels 
    # used from the left half of the image in the final stitched cube
    # for purposes of creating the two background offset terms
    left_half, right_half = vapp.mask_along_angle(mask.shape, clio.VAPP_PSF_ROTATION_DEG)
    left_pix_vec, _ = improc.unwrap_image(left_half, mask)


    # put: templates, scales, angles, masks
    log.debug(f"Putting data into Ray...")
    left_template_ref = ray.put(left_template)
    right_template_ref = ray.put(right_template)
    left_scales_ref = ray.put(left_scales)
    right_scales_ref = ray.put(right_scales)
    mask_ref = ray.put(mask)
    angles_ref = ray.put(angles)
    log.debug(f"Put data into Ray.")

    # hold reused intermediate results in dicts keyed on the grid parameters
    # that affect them
    model_vecs_refs = {}
    def key_maker_maker(*columns):
        def key_maker(row):
            # numpy scalars are not hashable, so we convert each one to a float
            return tuple(float(row[colname]) for colname in columns)
        return key_maker
    model_key_maker = key_maker_maker('r_px', 'pa_deg')
    model_key_maker_inj = key_maker_maker('inject_r_px', 'inject_pa_deg')
    inject_refs = {}
    inject_key_maker = key_maker_maker('inject_r_px', 'inject_pa_deg', 'inject_scale')
    precompute_refs = {}
    precompute_key_maker = key_maker_maker('inject_r_px', 'inject_pa_deg', 'inject_scale', 'r_px', 'pa_deg')

    data_cube_shape = left_cube.shape
    stitched_cube = pipelines.vapp_stitch(left_cube, right_cube, clio.VAPP_PSF_ROTATION_DEG)
    image_vecs, indices = improc.unwrap_cube(stitched_cube, mask)
    del indices, stitched_cube
    image_vecs_ref = ray.put(image_vecs)
    # unique (r, pa, scale) for *injected*
    for row in np.unique(remaining_grid[['inject_r_px', 'inject_pa_deg', 'inject_scale']]):
        # submit model
        model_key = model_key_maker_inj(row)
        model_vecs_refs[model_key] = generate_model.options(**generate_options).remote(
            data_cube_shape,
            left_template_ref, right_template_ref,
            left_scales_ref, right_scales_ref, angles_ref,
            row['inject_r_px'], row['inject_pa_deg'],
            mask_ref
        )
        # submit inject
        inject_key = inject_key_maker(row)
        inject_refs[inject_key] = inject_model.remote(
            image_vecs_ref,
            model_vecs_refs[model_key],
            row['inject_scale'],
        )
    # unique (r, pa) for *modeled*
    for row in np.unique(remaining_grid[['r_px', 'pa_deg']]):
        # submit model
        model_key = model_key_maker(row)
        if model_key not in model_vecs_refs:
            model_vecs_refs[model_key] = generate_model.options(**generate_options).remote(
                data_cube_shape, left_template, right_template,
                left_scales, right_scales, angles_ref,
                row['r_px'], row['pa_deg'],
                mask_ref
            )
        # store ref by 2-tuple
    # unique (inject_r, inject_pa, inject_scale, r, pa) for precomputing basis
    for row in np.unique(remaining_grid[['inject_r_px', 'inject_pa_deg', 'inject_scale', 'r_px', 'pa_deg']]):
        # submit initial_decomposition with max_k_modes
        precompute_key = precompute_key_maker(row)
        model_key = model_key_maker(row)
        model_vecs_ref = model_vecs_refs[model_key]
        inject_key = inject_key_maker(row)
        inject_image_vecs_ref = inject_refs[inject_key]

        precompute_refs[precompute_key] = precompute_basis.options(**decompose_options).remote(
            inject_image_vecs_ref,
            model_vecs_ref,
            max_k_modes,
            force_gpu_decomposition,
        )

    remaining_point_refs = []
    for idx in remaining_idxs:
        row = grid[idx]
        inject_key = inject_key_maker(row)
        inject_ref = inject_refs[inject_key]
        model_key = model_key_maker(row)
        model_vecs_ref = model_vecs_refs[model_key]
        precompute_key = precompute_key_maker(row)
        precompute_ref = precompute_refs[precompute_key]
        k_modes = int(row['k_modes'])
        point_ref = evaluate_point.options(**evaluate_options).remote(
            idx,
            row,
            inject_ref,
            model_vecs_ref,
            k_modes,
            precompute_ref,
            left_pix_vec,
            force_gpu_inversion,
        )
        remaining_point_refs.append(point_ref)
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
    companions : list[CompanionConfig] = xconf.field(default_factory=list, help="")
    min_coverage_frac : float = xconf.field(help="")
    ray : Union[LocalRayConfig,RemoteRayConfig] = xconf.field(
        default=LocalRayConfig(),
        help="Ray distributed framework configuration"
    )
    gpus_per_task : Optional[Union[float, PerTaskConfig]] = xconf.field(default=None, help="")
    ram_gb_per_task : Optional[Union[float, PerTaskConfig]] = xconf.field(default=None, help="")
    # ram_gb_for_generate : Optional[float] = xconf.field(default=None, help="")
    # ram_gb_for_decompose : Optional[float] = xconf.field(default=None, help="")
    # ram_gb_for_evaluate : Optional[float] = xconf.field(default=None, help="")
    use_gpu_decomposition : bool = xconf.field(default=False, help="")
    use_gpu_inversion : bool = xconf.field(default=False, help="")
    benchmark : bool = xconf.field(default=False, help="")
    benchmark_trials : int = xconf.field(default=2, help="")
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
        combo_mask = left_mask | right_mask
        log.debug(f"{np.count_nonzero(combo_mask)=}")
        
        # templates
        left_template = self.left_template.load()
        right_template = self.right_template.load()

        # init ray
        if isinstance(self.ray, RemoteRayConfig):
            ray.init(self.ray.url)
        else:
            resources = {}
            if self.ray.ram_gb is not None:
                resources['ram_gb'] = self.ray.ram_gb
            ray.init(
                num_cpus=self.ray.cpus,
                num_gpus=self.ray.gpus,
                resources=resources
            )
        options = {'resources':{}}
        generate_options = options.copy()
        decompose_options = options.copy()
        evaluate_options = options.copy()
        if isinstance(self.ram_gb_per_task, PerTaskConfig):
            generate_options['resources']['ram_gb'] = self.ram_gb_per_task.generate
            decompose_options['resources']['ram_gb'] = self.ram_gb_per_task.decompose
            evaluate_options['resources']['ram_gb'] = self.ram_gb_per_task.evaluate
        else:
            # number or None
            generate_options['resources']['ram_gb'] = self.ram_gb_per_task
            decompose_options['resources']['ram_gb'] = self.ram_gb_per_task
            evaluate_options['resources']['ram_gb'] = self.ram_gb_per_task

        if self.use_gpu_decomposition or self.use_gpu_inversion:
            if self.gpus_per_task is None:
                raise RuntimeError(f"Specify GPUs per task")
            if isinstance(self.gpus_per_task, PerTaskConfig):
                generate_options['num_gpus'] = self.gpus_per_task.generate
                decompose_options['num_gpus'] = self.gpus_per_task.decompose
                evaluate_options['num_gpus'] = self.gpus_per_task.evaluate
            else:
                # number or None
                generate_options['num_gpus'] = self.gpus_per_task
                decompose_options['num_gpus'] = self.gpus_per_task
                evaluate_options['num_gpus'] = self.gpus_per_task
        
        coverage_map_path = f'./coverage_t{self.every_t_frames}.fits'
        if not os.path.exists(coverage_map_path):
            log.debug(f"Computing coverage map")
            final_coverage = pipelines.adi_coverage(combo_mask, angles)
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
            self.companions
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
            grid = grid[:self.benchmark_trials]
        result_refs = launch_grid(
            grid,
            left_cube,
            right_cube,
            left_template,
            right_template,
            left_scales,
            right_scales,
            angles,
            combo_mask,
            self.use_gpu_decomposition,
            self.use_gpu_inversion,
            generate_options=generate_options,
            decompose_options=decompose_options,
            evaluate_options=evaluate_options,
        )
        if self.benchmark:
            log.debug(f"Running {self.benchmark_trials} trials...")
            for i in range(self.benchmark_trials):
                print(ray.get(result_refs[i]))
            ray.shutdown()
            return 0
        # for idx, grid_point in enumerate(grid):
        #     if grid_point['time_total_sec'] != 0:
        #         continue
        #     companion_spec = characterization.CompanionSpec(
        #         grid_point['inject_r_px'],
        #         grid_point['inject_pa_deg'],
        #         grid_point['inject_scale']
        #     )
        #     if last_companion != companion_spec:
        #         last_companion = companion_spec
        #         # inject
        #         inject_left_cube = inject(left_cube, left_template, angles, left_scales, companion_spec)
        #         inject_right_cube = inject(right_cube, right_template, angles, right_scales, companion_spec)

        #         # stitch
        #         out_cube = pipelines.vapp_stitch(inject_left_cube, inject_right_cube, clio.VAPP_PSF_ROTATION_DEG)
        #         out_cube_ref = ray.put(out_cube)
        
        #     result_ref = evaluate_grid_point_remote.options(**options).remote(
        #         idx,
        #         out_cube_ref,
        #         grid_point,
        #         left_template_ref,
        #         right_template_ref,
        #         left_scales_ref,
        #         right_scales_ref,
        #         mask_ref,
        #         angles_ref,
        #         self.use_gpu_decomposition,
        #         self.use_gpu_inversion
        #     )
        #     result_refs.append(result_ref)
        # if self.benchmark:
        #     pending = result_refs[:self.benchmark_trials]
        #     while pending:
        #         complete, pending = ray.wait(pending, timeout=5)
        #         for ref in complete:
        #             idx, result = ray.get(ref)
        #             benchmark_out = {name: result[name] for name in result.dtype.names}
        #             benchmark_out['idx'] = idx
        #             print(benchmark_out)
        #     return
        pending = result_refs
        total = len(result_refs)
        n_complete = 0
        while pending:
            complete, pending = ray.wait(pending, timeout=5, num_returns=min(self.checkpoint_every_x, len(pending)))
            for (idx, result) in ray.get(complete):
                grid[idx] = result
            if len(complete):
                n_complete += len(complete)
                log.debug(f"{n_complete=} / {total=}")
                with open("./grid.fits~", 'wb') as fh:
                    hdu = fits.BinTableHDU(grid, name='grid')
                    hdu.writeto(fh)
                    shutil.move("./grid.fits~", "./grid.fits")
        ray.shutdown()
