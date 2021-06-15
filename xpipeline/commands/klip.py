import xconf
import sys
import logging
from typing import Optional
from .. import utils, constants

from .base import InputCommand

log = logging.getLogger(__name__)


@xconf.config
class Klip(InputCommand):
    "Subtract starlight with KLIP"
    k_klip : int = xconf.field(default=10, help="Number of modes to subtract in starlight subtraction")
    exclude_nearest_n_frames : int = xconf.field(default=0, help="Number of additional temporally-adjacent frames to exclude from the sequence when computing the KLIP eigenimages")
    strategy : constants.KlipStrategy = xconf.field(default=constants.KlipStrategy.DOWNDATE_SVD, help="Implementation of KLIP to use")
    reuse_eigenimages : bool = xconf.field(default=False, help="Apply KLIP without adjusting the eigenimages at each step (much faster, less powerful)")
    combine_by : str = xconf.field(default="sum", help="Operation used to combine final derotated frames into a single output frame")
    saturation_threshold : Optional[float] = xconf.field(default=None, help="Value in counts above which pixels should be considered saturated and ignored")
    mask_iwa_px : int = xconf.field(default=None, help="Apply radial mask excluding pixels < iwa_px from center")
    mask_owa_px : int = xconf.field(default=None, help="Apply radial mask excluding pixels > owa_px from center")
    estimation_mask_path : str = xconf.field(default=None, help="Path to file shaped like single plane of input with 1s where pixels should be included in starlight estimation (intersected with saturation and annular mask)")
    combination_mask_path : str = xconf.field(default=None, help="Path to file shaped like single plane of input with 1s where pixels should be included in final combination (intersected with other masks)")
    vapp_mask_angle : float = xconf.field(default=0, help="Angle in degrees E of N (+Y) of axis of symmetry for paired gvAPP-180 data")
    sample_every_n : int = xconf.field(default=1, help="Take every Nth file from inputs (for speed of debugging)")

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
        return input_cube_hdul[extname].data.astype("=f8")[:: self.sample_every_n]

    def _make_masks(self, sci_arr, extname):
        import numpy as np
        from .. import pipelines
        from ..tasks import iofits, improc

        # mask file(s)

        if self.estimation_mask_path is not None:
            estimation_mask_hdul = iofits.load_fits_from_path(self.estimation_mask_path)
            if len(estimation_mask_hdul) > 0:
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

    def main(self):
        from .. import pipelines
        from ..tasks import iofits, improc
        output_klip_final = utils.join(self.destination, "klip_final.fits")
        output_exptime_map = utils.join(self.destination, "exptime_map.fits")
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)
        self.quit_if_outputs_exist([output_klip_final, output_exptime_map])

        klip_inputs, obs_method, derotation_angles = self._assemble_klip_inputs(self.input)
        klip_params = self._assemble_klip_params()
        outcubes = pipelines.klip_multi(klip_inputs, klip_params)
        out_image = self._assemble_out_image(obs_method, outcubes, derotation_angles)

        import time

        start = time.perf_counter()
        log.info(f"Computing klip pipeline result...")
        out_image = out_image.compute()
        elapsed = time.perf_counter() - start
        log.info(f"Computed in {elapsed} sec")
        output_file = iofits.write_fits(
            iofits.DaskHDUList([iofits.DaskHDU(out_image)]), output_klip_final
        )
        return output_file

    def _assemble_klip_params(self):
        from ..tasks import starlight_subtraction
        klip_params = starlight_subtraction.KlipParams(
            self.k_klip,
            self.exclude_nearest_n_frames,
            strategy=self.strategy,
            reuse=self.reuse_eigenimages,
            decomposer=starlight_subtraction.DEFAULT_DECOMPOSERS[self.strategy]
        )
        return klip_params

    def _assemble_klip_inputs(self, dataset_path):
        import dask.array as da
        from .. import pipelines
        from ..tasks import iofits, improc
        input_cube_hdul = iofits.load_fits_from_path(dataset_path)

        obs_method = utils.parse_obs_method(input_cube_hdul[0].header["OBSMETHD"])
        derotation_angles = self._get_derotation_angles(input_cube_hdul, obs_method)

        klip_inputs = []

        if "vapp" in obs_method:
            left_extname = obs_method["vapp"]["left"]
            right_extname = obs_method["vapp"]["right"]

            sci_arr_left = self._get_sci_arr(input_cube_hdul, left_extname)
            sci_arr_right = self._get_sci_arr(input_cube_hdul, right_extname)

            estimation_mask_left, combination_mask_left = self._make_masks(
                sci_arr_left, left_extname
            )
            estimation_mask_right, combination_mask_right = self._make_masks(
                sci_arr_right, right_extname
            )

            klip_inputs.append(
                pipelines.KlipInput(
                    da.from_array(sci_arr_left),
                    estimation_mask_left,
                    combination_mask_left,
                )
            )
            klip_inputs.append(
                pipelines.KlipInput(
                    da.from_array(sci_arr_right),
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
                    da.from_array(sci_arr), estimation_mask, combination_mask
                )
            )
        return klip_inputs, obs_method, derotation_angles

    def _assemble_out_image(self, obs_method, outcubes, derotation_angles):
        from .. import pipelines
        from ..tasks import iofits, improc
        if "vapp" in obs_method:
            left_cube, right_cube = outcubes
            out_image = pipelines.vapp_stitch(
                left_cube, right_cube, self.vapp_mask_angle
            )
        else:
            out_image = pipelines.adi(
                outcubes[0], derotation_angles, operation=self.combine_by
            )

        return out_image
