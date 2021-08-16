import xconf
import sys
import logging
from typing import Optional, Union
from enum import Enum
from .. import utils, constants

from .base import BaseCommand, AngleRangeConfig, PixelRotationRangeConfig, FitsConfig

log = logging.getLogger(__name__)

@xconf.config
class ExcludeRangeConfig:
    angle : Union[AngleRangeConfig,PixelRotationRangeConfig] = xconf.field(default=AngleRangeConfig(), help="Apply exclusion to derotation angles")
    nearest_n_frames : int = xconf.field(default=0, help="Number of additional temporally-adjacent frames on either side of the target frame to exclude from the sequence when computing the KLIP eigenimages")

@xconf.config
class KlipInputConfig:
    sci_arr: FitsConfig
    estimation_mask: FitsConfig
    combination_mask: Optional[FitsConfig]
    initial_decomposition_path : str = None
    mask_min_r_px : Union[int,float] = xconf.field(default=None, help="Apply radial mask excluding pixels < mask_min_r_px from center")
    mask_max_r_px : Union[int,float] = xconf.field(default=None, help="Apply radial mask excluding pixels > mask_max_r_px from center")

@xconf.config
class Klip(BaseCommand):
    "Subtract starlight with KLIP"
    inputs : list[KlipInputConfig] = xconf.field(help="Input data to simultaneously reduce")
    obstable : FitsConfig = xconf.field(help="Metadata table in FITS")
    k_klip : int = xconf.field(default=10, help="Number of modes to subtract in starlight subtraction")
    exclude : ExcludeRangeConfig = xconf.field(default=ExcludeRangeConfig(), help="How to exclude frames from reference sample")
    strategy : constants.KlipStrategy = xconf.field(default=constants.KlipStrategy.DOWNDATE_SVD, help="Implementation of KLIP to use")
    reuse_eigenimages : bool = xconf.field(default=False, help="Apply KLIP without adjusting the eigenimages at each step (much faster, less powerful)")
    # combine_output_by : constants.CombineOperation = xconf.field(default=constants.CombineOperation.MEAN, help="Operation used to combine final derotated frames into a single output frame")
    # saturation_threshold : Optional[float] = xconf.field(default=None, help="Value in counts above which pixels should be considered saturated and ignored")
    # estimation_mask_path : str = xconf.field(default=None, help="Path to file shaped like single plane of input with 1s where pixels should be included in starlight estimation (intersected with saturation and annular mask)")
    # combination_mask_path : str = xconf.field(default=None, help="Path to file shaped like single plane of input with 1s where pixels should be included in final combination (intersected with other masks)")
    # vapp_mask_angle_deg : float = xconf.field(default=0, help="Angle in degrees E of N (+Y) of axis of symmetry for paired gvAPP-180 data")
    # scale_factors_path : Optional[str] = xconf.field(help=utils.unwrap(
    #     """Path to FITS file with extensions for each data extension
    #     containing 1D arrays of scale factors whose ratios will match
    #     relative amplitude between extensions (e.g. vAPP PSFs). If
    #     extension A element N is 2.0, and extension B element N is 4.0,
    #     then frame N from extension B will be scaled by 1/2."""
    # ))
    initial_decomposition_only : bool = xconf.field(default=False, help="Whether to output initial decomposition and exit")
    # initial_decomposition_path : Optional[str] = xconf.field(help="Path from which to load initial decomposition")
    # output_mean_image : bool = xconf.field(default=True, help="Whether to output the mean un-KLIPped image")
    # output_coverage_map : bool = xconf.field(default=True, help="Whether to output a coverage map image showing how many input images contributed to each pixel")

    # def _get_derotation_angles(self, input_cube_hdul, obs_method):
    #     derotation_angles_where = obs_method["adi"]["derotation_angles"]
    #     if derotation_angles_where in input_cube_hdul:
    #         derotation_angles = input_cube_hdul[derotation_angles_where].data
    #     elif "." in derotation_angles_where:
    #         derot_ext, derot_col = derotation_angles_where.rsplit(".", 1)
    #         derotation_angles = input_cube_hdul[derot_ext].data[derot_col]
    #     else:
    #         raise RuntimeError(
    #             f"Combined dataset must contain angles as data or table column {derotation_angles_where}"
    #         )
    #     return derotation_angles[:: self.sample_every_n]

    # def _get_sci_arr(self, input_cube_hdul, extname):
    #     return input_cube_hdul[extname].data

    def _make_masks(self, sci_arr, extname):
        import numpy as np
        from .. import pipelines
        from ..tasks import iofits, improc

        # mask file(s)

        if self.estimation_mask_path is not None:
            estimation_mask_hdul = iofits.load_fits_from_path(self.estimation_mask_path)
            if len(estimation_mask_hdul) > 1:
                estimation_mask = estimation_mask_hdul[extname].data
            else:
                estimation_mask = estimation_mask_hdul[0].data
        else:
            estimation_mask = np.ones(sci_arr.shape[1:])
        # coerce to bools
        estimation_mask = estimation_mask == 1

        # mask radial
        rho, _ = improc.polar_coords(
            improc.arr_center(sci_arr.shape[1:]), sci_arr.shape[1:]
        )
        if self.mask_min_r_px is not None:
            iwa_mask = rho >= self.mask_min_r_px
            estimation_mask &= iwa_mask
        if self.mask_max_r_px is not None:
            owa_mask = rho <= self.mask_max_r_px
            estimation_mask &= owa_mask

        # load combination mask + intersect with others
        if self.combination_mask_path is not None:
            combination_mask_hdul = iofits.load_fits_from_path(
                self.combination_mask_path
            )
            if len(combination_mask_hdul) > 0:
                combination_mask = combination_mask_hdul[extname].data
            else:
                combination_mask = combination_mask_hdul[0].data
        else:
            combination_mask = np.ones(sci_arr.shape[1:])
        # coerce to bools
        combination_mask = combination_mask == 1
        combination_mask &= estimation_mask
        return estimation_mask, combination_mask

    def _klip(self, klip_inputs, klip_params, obs_method: dict, left_over_right_ratios):
        from .. import pipelines
        if "vapp" in obs_method:
            left_input, right_input = klip_inputs
            result = pipelines.klip_vapp_separately(left_input, right_input, klip_params, self.vapp_mask_angle_deg, left_over_right_ratios)
            if klip_params.warmup:
                return result
            outcube, mean = result
            outcubes, means = [outcube], [mean]
        else:
            result = pipelines.klip_multi(klip_inputs, klip_params)
            if klip_params.warmup:
                return result
            outcubes, means = result
        return outcubes, means

    def _load_initial_decomposition(self, path):
        from ..tasks import iofits, starlight_subtraction
        decomp_hdul = iofits.load_fits_from_path(path)
        return starlight_subtraction.InitialDecomposition(
            mtx_u0=decomp_hdul['MTX_U0'].data,
            diag_s0=decomp_hdul['DIAG_S0'].data,
            mtx_v0=decomp_hdul['MTX_V0'].data,
        )

    # def _save_warmup(self, result, output_initial_decomp_fn, obs_method):
    #     from ..tasks import iofits
    #     if 'vapp' in obs_method:
    #         results = result
    #     else:
    #         results = [result]
    #     hdus = [iofits.DaskHDU(data=None, kind='primary')]
    #     for idx, result in enumerate(results):
    #         mtx_u0, diag_s0, mtx_v0 = result.mtx_u0, result.diag_s0, result.mtx_v0
    #         hdus.extend([
    #             iofits.DaskHDU(data=mtx_u0, name=f'{idx:02}_MTX_U0'),
    #             iofits.DaskHDU(data=diag_s0, name=f'{idx:02}_DIAG_S0'),
    #             iofits.DaskHDU(data=mtx_v0, name=f'{idx:02}_MTX_V0'),
    #         ])

    #     # save initial decomposition
    #     iofits.write_fits(
    #         iofits.DaskHDUList(hdus), output_initial_decomp_fn
    #     )
    #     return 0

    # def _load_dataset(self, dataset_path):
    #     from ..tasks import iofits
    #     input_cube_hdul = iofits.load_fits_from_path(dataset_path)
    #     obs_method = utils.parse_obs_method(input_cube_hdul[0].header["OBSMETHD"])
    #     return input_cube_hdul, obs_method

    def _assemble_klip_input(self, inputcfg : KlipInputConfig):
        from ..tasks import iofits, starlight_subtraction
        input_hdul = iofits.load_fits_from_path(inputcfg.sci_arr.path)
        sci_arr = input_hdul[inputcfg.sci_arr.ext].data
        initial_decomposition = self._load_initial_decomposition(inputcfg.initial_decomposition_path)
        emask_hdul = iofits.load_fits_from_path(inputcfg.estimation_mask.path)
        estimation_mask = emask_hdul[inputcfg.estimation_mask.ext].data
        return starlight_subtraction.KlipInput(sci_arr, estimation_mask, initial_decomposition)

    def main(self):
        from ..tasks import iofits
        output_klip_final_fn = utils.join(self.destination, "klip_final.fits")
        output_mean_image_fn = utils.join(self.destination, "mean_image.fits")
        output_coverage_map_fn = utils.join(self.destination, "coverage_map.fits")
        output_initial_decomp_fn = utils.join(self.destination, "initial_decomposition.fits")
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)
        if self.initial_decomposition:
            outputs = [output_initial_decomp_fn]
        else:
            outputs = [output_klip_final_fn]
            if self.output_mean_image:
                outputs.append(output_mean_image_fn)
            if self.output_coverage_map:
                outputs.append(output_coverage_map_fn)
            
        self.quit_if_outputs_exist(outputs)

        klip_inputs = []
        for inputcfg in self.inputs:
            klip_inputs.append(self._assemble_klip_input(inputcfg))

        # input_cube_hdul, obs_method = self._load_dataset(self.input)
        # left_over_right_ratios is only non-None for vAPP
        # if self.initial_decomposition_path is not None:
        #     initial_decompositions = self._load_initial_decomposition(self.initial_decomposition_path)
        # else:
        #     initial_decompositions = None
        # klip_inputs, obs_method, derotation_angles, left_over_right_ratios = self._assemble_klip_inputs(input_cube_hdul, obs_method, initial_decompositions)
        klip_params = self._assemble_klip_params(klip_inputs, derotation_angles)
        import time
        start = time.perf_counter()
        result = self._klip(klip_inputs, klip_params, obs_method, left_over_right_ratios)
        if klip_params.warmup:
            self._save_warmup(result, output_initial_decomp_fn, obs_method)
            return 0
        else:
            outcubes, outmeans = result
        out_image, mean_image, coverage_image = self._assemble_out_images(klip_inputs, obs_method, outcubes, outmeans, derotation_angles)
        elapsed = time.perf_counter() - start
        log.info(f"Computed in {elapsed} sec")

        iofits.write_fits(
            iofits.DaskHDUList([iofits.DaskHDU(out_image)]), output_klip_final_fn
        )
        if self.output_mean_image:
            iofits.write_fits(
                iofits.DaskHDUList([iofits.DaskHDU(mean_image)]), output_mean_image_fn
            )
        if self.output_coverage_map:
            iofits.write_fits(
                iofits.DaskHDUList([iofits.DaskHDU(coverage_image)]), output_coverage_map_fn
            )


    def _make_exclusions(self, exclude : ExcludeRangeConfig, derotation_angles):
        import numpy as np
        from ..tasks import starlight_subtraction
        exclusions = []
        if exclude.nearest_n_frames > 0:
            indices = np.arange(derotation_angles.shape[0])
            exc = starlight_subtraction.ExclusionValues(
                exclude_within_delta=exclude.nearest_n_frames,
                values=indices,
                num_excluded_max=2 * exclude.nearest_n_frames + 1
            )
            exclusions.append(exc)
        if isinstance(exclude.angle, PixelRotationRangeConfig) and exclude.angle.delta_px > 0:
            exc = starlight_subtraction.ExclusionValues(
                exclude_within_delta=exclude.angle.delta_px,
                values=exclude.angle.r_px * np.unwrap(np.deg2rad(derotation_angles))
            )
            exclusions.append(exc)
        elif isinstance(exclude.angle, AngleRangeConfig) and exclude.angle.delta_deg > 0:
            exc = starlight_subtraction.ExclusionValues(
                exclude_within_delta=exclude.angle.delta_deg,
                values=derotation_angles
            )
            exclusions.append(exc)
        else:
            pass  # not an error to have delta of zero, just don't exclude based on rotation
        return exclusions

    def _assemble_klip_params(self, klip_inputs, derotation_angles):
        import numpy as np
        from ..tasks import starlight_subtraction
        exclusions = self._make_exclusions(self.exclude, derotation_angles)
        klip_params = starlight_subtraction.KlipParams(
            self.k_klip,
            exclusions,
            decomposer=starlight_subtraction.DEFAULT_DECOMPOSERS[self.strategy],
            strategy=self.strategy,
            reuse=self.reuse_eigenimages,
            warmup=self.initial_decomposition_only
        )
        log.debug(klip_params)
        return klip_params

    # def _load_input(self, dataset_path):
    #     from ..tasks import iofits
    #     input_cube_hdul = iofits.load_fits_from_path(dataset_path)
    #     obs_method = utils.parse_obs_method(input_cube_hdul[0].header["OBSMETHD"])
    #     return input_cube_hdul, obs_method

    # def _assemble_klip_inputs(self, input_cube_hdul, obs_method, initial_decompositions):
    #     import numpy as np
    #     from .. import pipelines
    #     from ..tasks import iofits


    #     derotation_angles = self._get_derotation_angles(input_cube_hdul, obs_method)

    #     klip_inputs = []

    #     if self.scale_factors_path is not None:
    #         scale_factors_hdul = iofits.load_fits_from_path(self.scale_factors_path)

    #     if "vapp" in obs_method:
    #         left_extname = obs_method["vapp"]["left"]
    #         right_extname = obs_method["vapp"]["right"]

    #         sci_arr_left = self._get_sci_arr(input_cube_hdul, left_extname)
    #         sci_arr_right = self._get_sci_arr(input_cube_hdul, right_extname)
    #         if self.scale_factors_path is not None:
    #             scale_left = self._get_sci_arr(scale_factors_hdul, left_extname)
    #             scale_right = self._get_sci_arr(scale_factors_hdul, right_extname)
    #         else:
    #             log.info("Using ratio of per-frame median values as proxy for inter-PSF scaling")
    #             scale_left = np.nanmedian(sci_arr_left, axis=(1,2))
    #             scale_right = np.nanmedian(sci_arr_right, axis=(1,2))
    #         # right * (scale_left / scale_right) = scaled right
    #         left_over_right_ratios = scale_left / scale_right

    #         estimation_mask_left, combination_mask_left = self._make_masks(
    #             sci_arr_left, left_extname
    #         )
    #         estimation_mask_right, combination_mask_right = self._make_masks(
    #             sci_arr_right, right_extname
    #         )

    #         klip_inputs.append(
    #             pipelines.KlipInput(
    #                 # da.from_array(sci_arr_left),
    #                 sci_arr_left,
    #                 estimation_mask_left,
    #                 combination_mask_left,
    #                 initial_decompositions[0] if initial_decompositions is not None else None
    #             )
    #         )
    #         klip_inputs.append(
    #             pipelines.KlipInput(
    #                 # da.from_array(sci_arr_right),
    #                 sci_arr_right,
    #                 estimation_mask_right,
    #                 combination_mask_right,
    #                 initial_decompositions[1] if initial_decompositions is not None else None
    #             )
    #         )
    #     else:
    #         extname = obs_method.get("sci", "SCI")
    #         if extname not in input_cube_hdul:
    #             log.error(
    #                 f"Supply a 'SCI' extension, or use sci= or vapp.left= / vapp.right= in OBSMETHD"
    #             )
    #             sys.exit(1)
    #         sci_arr = self._get_sci_arr(input_cube_hdul, extname)
    #         left_over_right_ratios = None
    #         estimation_mask, combination_mask = self._make_masks(sci_arr, extname)
    #         klip_inputs.append(
    #             pipelines.KlipInput(
    #                 # da.from_array(sci_arr),
    #                 sci_arr,
    #                 estimation_mask,
    #                 combination_mask,
    #                 initial_decompositions[0] if initial_decompositions is not None else None
    #             )
    #         )
    #     return klip_inputs, obs_method, derotation_angles, left_over_right_ratios

    # def _assemble_out_images(self, klip_inputs, obs_method, outcubes, outmeans, derotation_angles):
    #     import numpy as np
    #     from .. import pipelines
    #     out_image = pipelines.adi(
    #         outcubes[0], derotation_angles, operation=self.combine_output_by
    #     )
    #     out_mean_image = outmeans[0]
    #     if "vapp" in obs_method:
    #         left_coverage = pipelines.adi_coverage(
    #             klip_inputs[0].combination_mask,
    #             derotation_angles
    #         )
    #         right_coverage = pipelines.adi_coverage(
    #             klip_inputs[1].combination_mask,
    #             derotation_angles
    #         )
    #         out_coverage_image = pipelines.vapp_stitch(left_coverage[np.newaxis, :, :], right_coverage[np.newaxis, :, :], self.vapp_mask_angle_deg)
    #     else:
    #         out_coverage_image = np.zeros_like(out_image)
    #         for klip_in in klip_inputs:
    #             if klip_in.combination_mask is not None:
    #                 out_coverage_image += pipelines.adi_coverage(
    #                     klip_in.combination_mask,
    #                     derotation_angles
    #                 )

    #     return out_image, out_mean_image, out_coverage_image
