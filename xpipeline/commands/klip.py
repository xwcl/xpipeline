from multiprocessing import Value
import xconf
import sys
import logging
from typing import Optional
from enum import Enum
from .. import utils, constants

from .base import InputCommand

log = logging.getLogger(__name__)

class RotationUnit(Enum):
    PX = 'px'
    DEG = 'deg'

@xconf.config
class AngleExclusionConfig:
    unit : RotationUnit = xconf.field(default=RotationUnit.DEG, help="Interpret limits as either px or deg")
    delta : float = xconf.field(default=0, help="Minimum absolute difference between target frame value and nearest included reference")
    r_px : float = xconf.field(default=None, help="Radius at which to calculate motion in pixels, ignored when unit == 'deg'")

    def __post_init__(self):
        if self.r_px is None and self.unit is RotationUnit.PX:
            raise ValueError("Cannot apply angular exclusion with units of pixels without r_px set")

@xconf.config
class ExclusionConfig:
    angle : AngleExclusionConfig = xconf.field(default=AngleExclusionConfig(), help="Apply exclusion to derotation angles")
    nearest_n_frames : int = xconf.field(default=0, help="Number of additional temporally-adjacent frames (besides the target frame) to exclude from the sequence when computing the KLIP eigenimages")

@xconf.config
class Klip(InputCommand):
    "Subtract starlight with KLIP"
    k_klip : int = xconf.field(default=10, help="Number of modes to subtract in starlight subtraction")
    exclude : ExclusionConfig = xconf.field(default=ExclusionConfig(), help="How to exclude frames from reference sample")
    strategy : constants.KlipStrategy = xconf.field(default=constants.KlipStrategy.DOWNDATE_SVD, help="Implementation of KLIP to use")
    reuse_eigenimages : bool = xconf.field(default=False, help="Apply KLIP without adjusting the eigenimages at each step (much faster, less powerful)")
    combine_by : constants.CombineOperation = xconf.field(default=constants.CombineOperation.MEAN, help="Operation used to combine final derotated frames into a single output frame")
    saturation_threshold : Optional[float] = xconf.field(default=None, help="Value in counts above which pixels should be considered saturated and ignored")
    mask_iwa_px : int = xconf.field(default=None, help="Apply radial mask excluding pixels < iwa_px from center")
    mask_owa_px : int = xconf.field(default=None, help="Apply radial mask excluding pixels > owa_px from center")
    estimation_mask_path : str = xconf.field(default=None, help="Path to file shaped like single plane of input with 1s where pixels should be included in starlight estimation (intersected with saturation and annular mask)")
    combination_mask_path : str = xconf.field(default=None, help="Path to file shaped like single plane of input with 1s where pixels should be included in final combination (intersected with other masks)")
    vapp_mask_angle_deg : float = xconf.field(default=0, help="Angle in degrees E of N (+Y) of axis of symmetry for paired gvAPP-180 data")
    sample_every_n : int = xconf.field(default=1, help="Take every Nth file from inputs (for speed of debugging)")
    scale_factors_path : Optional[str] = xconf.field(help=utils.unwrap(
        """Path to FITS file with extensions for each data extension
        containing 1D arrays of scale factors whose ratios will match
        relative amplitude between extensions (e.g. vAPP PSFs). If
        extension A element N is 2.0, and extension B element N is 4.0,
        then frame N from extension B will be scaled by 1/2."""
    ))
    output_mean_image : bool = xconf.field(default=True, help="Whether to output the mean un-KLIPped image")
    output_coverage_map : bool = xconf.field(default=True, help="Whether to output a coverage map image showing how many input images contributed to each pixel")

    def _get_derotation_angles(self, input_cube_hdul, obs_method):
        derotation_angles_where = obs_method["adi"]["derotation_angles"]
        if derotation_angles_where in input_cube_hdul:
            derotation_angles = input_cube_hdul[derotation_angles_where].data
        elif "." in derotation_angles_where:
            derot_ext, derot_col = derotation_angles_where.rsplit(".", 1)
            derotation_angles = input_cube_hdul[derot_ext].data[derot_col]
        else:
            raise RuntimeError(
                f"Combined dataset must contain angles as data or table column {derotation_angles_where}"
            )
        return derotation_angles[:: self.sample_every_n]

    def _get_sci_arr(self, input_cube_hdul, extname):
        return input_cube_hdul[extname].data.astype("=f4")[:: self.sample_every_n]

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
        if self.mask_iwa_px is not None:
            iwa_mask = rho >= self.mask_iwa_px
            estimation_mask &= iwa_mask
        if self.mask_owa_px is not None:
            owa_mask = rho <= self.mask_owa_px
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

    def _klip(self, klip_inputs, klip_params, obs_method: dict):
        from .. import pipelines
        if "vapp" in obs_method:
            left_input, right_input = klip_inputs
            outcube, mean = pipelines.klip_vapp_separately(left_input, right_input, klip_params, self.vapp_mask_angle_deg)
            outcubes, means = [outcube], [mean]
        else:
            outcubes, means = pipelines.klip_multi(klip_inputs, klip_params)
        return outcubes, means

    def main(self):
        from ..tasks import iofits
        output_klip_final_fn = utils.join(self.destination, "klip_final.fits")
        output_mean_image_fn = utils.join(self.destination, "mean_image.fits")
        output_coverage_map_fn = utils.join(self.destination, "coverage_map.fits")
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)
        outputs = [output_klip_final_fn]
        if self.output_mean_image:
            outputs.append(output_mean_image_fn)
        if self.output_coverage_map:
            outputs.append(output_coverage_map_fn)
        self.quit_if_outputs_exist(outputs)

        klip_inputs, obs_method, derotation_angles = self._assemble_klip_inputs(self.input)
        klip_params = self._assemble_klip_params(klip_inputs, derotation_angles)
        import time
        start = time.perf_counter()
        outcubes, outmeans = self._klip(klip_inputs, klip_params, obs_method)
        out_image, mean_image, coverage_image = self._assemble_out_images(klip_inputs, obs_method, outcubes, outmeans, derotation_angles)
        elapsed = time.perf_counter() - start
        log.info(f"Computed in {elapsed} sec")

        # log.info(f"Computing klip pipeline result...")
        # out_image = out_image.compute()
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


    def _make_exclusions(self, exclude : ExclusionConfig, derotation_angles):
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
        if exclude.angle.delta > 0:
            if exclude.angle.unit is RotationUnit.DEG:
                exc = starlight_subtraction.ExclusionValues(
                    exclude_within_delta=exclude.angle.delta,
                    values=derotation_angles
                )
            elif exclude.angle.unit is RotationUnit.PX:
                exc = starlight_subtraction.ExclusionValues(
                    exclude_within_delta=exclude.angle.delta,
                    values=exclude.angle.r_px * np.unwrap(np.deg2rad(derotation_angles))
                )
            else:
                raise ValueError("Invalid exclude.angle.unit shouldn't happen")
            exclusions.append(exc)
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
        )
        return klip_params

    def _assemble_klip_inputs(self, dataset_path):
        import numpy as np
        from .. import pipelines
        from ..tasks import iofits, improc
        input_cube_hdul = iofits.load_fits_from_path(dataset_path)
        if self.scale_factors_path is not None:
            scale_factors_hdul = iofits.load_fits_from_path(self.scale_factors_path)

        obs_method = utils.parse_obs_method(input_cube_hdul[0].header["OBSMETHD"])
        derotation_angles = self._get_derotation_angles(input_cube_hdul, obs_method)

        klip_inputs = []

        if "vapp" in obs_method:
            left_extname = obs_method["vapp"]["left"]
            right_extname = obs_method["vapp"]["right"]

            sci_arr_left = self._get_sci_arr(input_cube_hdul, left_extname)
            sci_arr_right = self._get_sci_arr(input_cube_hdul, right_extname)
            if self.scale_factors_path is not None:
                scale_left = self._get_sci_arr(scale_factors_hdul, left_extname)
                scale_right = self._get_sci_arr(scale_factors_hdul, right_extname)
            else:
                log.info("Using ratio of per-frame median values as proxy for inter-PSF scaling")
                scale_left = np.nanmedian(sci_arr_left, axis=(1,2))
                scale_right = np.nanmedian(sci_arr_right, axis=(1,2))
            sci_arr_right = sci_arr_right * (scale_left / scale_right)

            estimation_mask_left, combination_mask_left = self._make_masks(
                sci_arr_left, left_extname
            )
            estimation_mask_right, combination_mask_right = self._make_masks(
                sci_arr_right, right_extname
            )

            klip_inputs.append(
                pipelines.KlipInput(
                    # da.from_array(sci_arr_left),
                    sci_arr_left,
                    estimation_mask_left,
                    combination_mask_left,
                )
            )
            klip_inputs.append(
                pipelines.KlipInput(
                    # da.from_array(sci_arr_right),
                    sci_arr_right,
                    estimation_mask_right,
                    combination_mask_right,
                )
            )
        else:
            extname = obs_method.get("sci", "SCI")
            if extname not in input_cube_hdul:
                log.error(
                    f"Supply a 'SCI' extension, or use sci= or vapp.left= / vapp.right= in OBSMETHD"
                )
                sys.exit(1)
            sci_arr = self._get_sci_arr(input_cube_hdul, extname)
            estimation_mask, combination_mask = self._make_masks(sci_arr, extname)
            klip_inputs.append(
                pipelines.KlipInput(
                    # da.from_array(sci_arr),
                    sci_arr,
                    estimation_mask, combination_mask
                )
            )
        return klip_inputs, obs_method, derotation_angles

    def _assemble_out_images(self, klip_inputs, obs_method, outcubes, outmeans, derotation_angles):
        import numpy as np
        from .. import pipelines
        out_image = pipelines.adi(
            outcubes[0], derotation_angles, operation=self.combine_by
        )
        out_mean_image = outmeans[0]
        if "vapp" in obs_method:
            left_coverage = pipelines.adi_coverage(
                klip_inputs[0].combination_mask,
                derotation_angles
            )
            right_coverage = pipelines.adi_coverage(
                klip_inputs[1].combination_mask,
                derotation_angles
            )
            out_coverage_image = pipelines.vapp_stitch(left_coverage[np.newaxis, :, :], right_coverage[np.newaxis, :, :], self.vapp_mask_angle_deg)
        else:
            out_coverage_image = np.zeros_like(out_image)
            for klip_in in klip_inputs:
                if klip_in.combination_mask is not None:
                    out_coverage_image += pipelines.adi_coverage(
                        klip_in.combination_mask,
                        derotation_angles
                    )

        return out_image, out_mean_image, out_coverage_image
