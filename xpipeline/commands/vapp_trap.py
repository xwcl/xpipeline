import time
from xpipeline.core import cupy as cp
import itertools
import shutil
import numpy as np
import xconf
import ray
from astropy.io import fits
from typing import Optional
from xpipeline.types import FITS_EXT
from xpipeline.commands.base import FitsConfig, CompanionConfig
from xpipeline.tasks import iofits, vapp, improc, starlight_subtraction, characterization
from xpipeline.ref import clio
from xpipeline import pipelines

def inject(cube, template, angles, scale_factors, companion):
    return characterization.inject_signals(cube, [companion], template, angles, scale_factors)

from dataclasses import dataclass
import dataclasses

def grid_generate(k_modes_vals, covered_pix_mask, every_n_pix, companions):
    # static: every_n_pix, every_t_frames, min_coverage
    # varying: k_modes, inject_r_px, inject_pa_deg, inject_scale, r_px, pa_deg, x, y
    # computed: model_coeff, time_total, time_model, time_decomp, time_fit
    rhos, pa_degs, xx, yy = improc.downsampled_grid_r_pa(covered_pix_mask, every_n_pix)
    grid = np.zeros(len(companions) * len(k_modes_vals) * len(rhos), dtype=[
        ('inject_r_px', float),
        ('inject_pa_deg', float),
        ('inject_scale', float),
        ('k_modes', int),
        ('r_px', float),
        ('pa_deg', float),
        ('x', int),
        ('y', float),
        ('model_coeff', float),
        ('time_total_sec', float),
        ('time_gen_model_sec', float),
    ])
    for idx, (k_modes, companion, inner_idx) in enumerate(itertools.product(k_modes_vals, companions, range(len(rhos)))):
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

def evaluate_grid_point(idx, cube, grid_point, left_template, right_template,
                        left_scales, right_scales, mask, angles, use_gpu):
    worker_init(num_cpus=1)
    start = time.perf_counter()
    r_px, pa_deg = grid_point['r_px'], grid_point['pa_deg']
    scale_free_spec = characterization.CompanionSpec(r_px=r_px, pa_deg=pa_deg, scale=1)
    left_signal = characterization.generate_signals(cube.shape, [scale_free_spec], left_template, angles, left_scales)
    right_signal = characterization.generate_signals(cube.shape, [scale_free_spec], right_template, angles, right_scales)
    signal = pipelines.vapp_stitch(left_signal, right_signal, clio.VAPP_ROTATION_ANGLE_DEG)
    grid_point['time_gen_model_sec'] = time.perf_counter() - start
    image_vecs, _ = improc.unwrap_cube(cube, mask)
    model_vecs, _ = improc.unwrap_cube(signal, mask)
    params = starlight_subtraction.TrapParams(
        modes=grid_point['k_modes'],
        compute_residuals=False,
    )
    if use_gpu:
        gpu_image_vecs, gpu_model_vecs = cp.asarray(image_vecs), cp.asarray(model_vecs)
        del image_vecs, model_vecs
        image_vecs, model_vecs = gpu_image_vecs, gpu_model_vecs
    model_coeff = starlight_subtraction.trap_mtx(image_vecs, model_vecs, params)
    grid_point['model_coeff'] = model_coeff
    grid_point['time_total_sec'] = time.perf_counter() - start
    return idx, grid_point

@xconf.config
class VappTrap(xconf.Command):
    every_n_pix : int = xconf.field(default=1, help="Evaluate a forward model centered on every Nth pixel")
    every_t_frames : int = xconf.field(default=1, help="Use every Tth frame as the input cube")
    k_modes : list[int] = xconf.field(default=(15, 100), help="")
    dataset : str = xconf.field(help="")
    left_extname : FITS_EXT = xconf.field(help="")
    left_template : FitsConfig = xconf.field(help="")
    right_template : FitsConfig = xconf.field(help="")
    right_extname : FITS_EXT = xconf.field(help="")
    left_scales : FitsConfig = xconf.field(help="")
    right_scales : FitsConfig = xconf.field(help="")
    left_mask : FitsConfig = xconf.field(help="")
    right_mask : FitsConfig = xconf.field(help="")
    angles : FitsConfig = xconf.field(help="")
    companions : list[CompanionConfig] = xconf.field(help="")
    min_coverage : float = xconf.field(help="")
    cpus_per_gpu : int = xconf.field(default=1, help="")
    use_gpu : bool = xconf.field(default=False, help="")
    ray_url : Optional[str] = xconf.field(help="")
    checkpoint_every_x : int = xconf.field(default=10, help="Write current state of grid to disk every X iterations")

    def main(self):
        if self.ray_url is not None:
            ray.init(self.ray_url)
        else:
            ray.init()
        
        hdul = iofits.load_fits_from_path(self.dataset)

        # decimate
        left_cube = hdul[self.left_extname].data[::self.every_t_frames]
        right_cube = hdul[self.right_extname].data[::self.every_t_frames]
        angles = self.angles.load()[::self.every_t_frames]
        left_scales = self.left_scales.load()[::self.every_t_frames]
        right_scales = self.right_scales.load()[::self.every_t_frames]

        # get masks
        left_mask = self.left_mask.load() == 1
        right_mask = self.right_mask.load() == 1
        combo_mask = left_mask | right_mask
        
        # templates
        left_template = self.left_template.load()
        right_template = self.right_template.load()

        # put: templates, scales, angles, masks
        left_template_ref = ray.put(left_template)
        right_template_ref = ray.put(right_template)
        left_scales_ref = ray.put(left_scales)
        right_scales_ref = ray.put(right_scales)
        mask_ref = ray.put(combo_mask)
        angles_ref = ray.put(angles)

        
        final_coverage = pipelines.adi_coverage(combo_mask, angles)
        n_frames = len(angles)
        covered_pix_mask = final_coverage > int(n_frames * self.min_coverage_pct)
        from skimage.morphology import binary_closing
        covered_pix_mask = binary_closing(covered_pix_mask)
        fits.PrimaryHDU(covered_pix_mask).writeto('./coverage.fits', overwrite=True)

        try:
            # load checkpoint
            with open("./grid.fits", 'rb') as fh:
                hdul = fits.open(fh)
                grid = hdul['grid'].data
        except FileNotFoundError:
            grid = grid_generate(
                self.k_modes_vals,
                covered_pix_mask,
                self.every_n_pix,
                self.every_t_frames,
                self.min_coverage_pct,
                self.companions
            )
            with open("./grid.fits", 'wb') as fh:
                hdu = fits.BinTableHDU(grid, name='grid')
                hdu.writeto('./grid.fits')
        
        # submit to ray
        last_companion = None
        out_cube_ref = None
        result_refs = []
        for idx, grid_point in enumerate(grid):
            if grid_point['time_total_sec'] != 0:
                continue
            companion_spec = characterization.CompanionSpec(
                grid_point.inject_r_px,
                grid_point.inject_pa_deg,
                grid_point.inject_scale
            )
            if last_companion != companion_spec:
                last_companion = companion_spec
                # inject
                left_cube = inject(left_cube, left_template, angles, left_scales, companion_spec)
                right_cube = inject(right_cube, right_template, angles, right_scales, companion_spec)

                # stitch
                out_cube = pipelines.vapp_stitch(left_cube, right_cube, clio.VAPP_ROTATION_ANGLE_DEG)
                out_cube_ref = ray.put(out_cube)
        
            result_ref = evaluate_grid_point(
                idx,
                out_cube_ref,
                grid_point,
                left_template_ref,
                right_template_ref,
                left_scales_ref,
                right_scales_ref,
                mask_ref,
                angles_ref,
            )
            result_refs.append(result_ref)
        pending = result_refs
        total = len(result_refs)
        n_complete = 0
        while pending:
            complete, pending = ray.wait(pending, timeout=5)
            for ref in complete:
                idx, result = ray.get(ref)
                grid[idx] = result
                n_complete += 1
                # every X points, write checkpoint
                if n_complete % self.checkpoint_every_x == 0:
                    print(f"{idx} / {total}")
                    with open("./grid.checkpoint.fits", 'wb') as fh:
                        hdu = fits.BinTableHDU(grid, name='grid')
                        hdu.writeto(fh)
                        shutil.move("./grid.checkpoint.fits", "./grid.fits")
