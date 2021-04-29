import sys
from astropy.io import fits
import numpy as np
import argparse
import dask
import dask.array as da
import fsspec.spec
import os.path
import logging
# from . import constants as const
from ..utils import unwrap
from .. import utils
from .. import pipelines #, irods
from ..core import LazyPipelineCollection
from ..tasks import iofits, improc, vapp # obs_table, iofits, sky_model, detector, data_quality
# from .ref import clio

from .base import BaseCommand, MultiInputCommand

log = logging.getLogger(__name__)


def _docs_args(parser):
    # needed for sphinx-argparse support
    return KLIP.add_arguments(parser)


class KLIP(MultiInputCommand):
    name = "klip"
    help = "Subtract starlight with KLIP"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--k-klip",
            type=int,
            default=10,
            help="Number of modes to subtract in starlight subtraction (default: 10)"
        )
        parser.add_argument(
            "--exclude-nearest-n-frames",
            type=int,
            default=0,
            help="Number of frames to exclude from the observations used to compute the target"
        )
        parser.add_argument(
            "--combine-by",
            default="sum",
            choices=["sum", "average"],
            help="Operation used to combine final derotated frames into a single output frame"
        )
        parser.add_argument(
            "--mask-saturated",
            action="store_true",
            help="Whether to mask saturated pixels"
        )
        parser.add_argument(
            "--mask-saturated-level",
            type=float,
            help="Level in counts to mask as a saturated pixel from percentile image"
        )
        parser.add_argument(
            "--mask-saturated-percentile",
            type=float,
            default=99.5,
            help="Pixel-wise percentile level for image used to determine saturation"
        )
        parser.add_argument(
            "--mask-iwa-px",
            type=float,
            default=None,
            help="Apply radial mask excluding pixels < iwa_px from center"
        )
        parser.add_argument(
            "--mask-owa-px",
            type=float,
            default=None,
            help="Apply radial mask excluding pixels > owa_px from center"
        )
        parser.add_argument(
            "--estimation-mask",
            default=None,
            help="Path to file shaped like single plane of input with 1s where pixels should be included in starlight estimation (intersected with saturation and annular mask)"
        )
        parser.add_argument(
            "--combination-mask",
            default=None,
            help="Path to file shaped like single plane of input with 1s where pixels should be included in final combination (intersected with other masks)"
        )
        parser.add_argument(
            "--vapp-symmetry-angle",
            type=float,
            default=0,
            help="Angle in degrees E of N (+Y) of axis of symmetry for paired gvAPP-180 data (default: 0)"
        )
        return super(KLIP, KLIP).add_arguments(parser)

    def _get_derotation_angles(self, input_cube_hdul, obs_method):
        derotation_angles_where = obs_method['adi']['derotation_angles']
        if derotation_angles_where in input_cube_hdul:
            derotation_angles = input_cube_hdul[derotation_angles_where].data
        elif '.' in derotation_angles_where:
            derot_ext, derot_col = derotation_angles_where.rsplit('.', 1)
            derotation_angles = input_cube_hdul[derot_ext].data[derot_col]
        else:
            raise RuntimeError(f"Combined dataset must contain angles as data or table column {derotation_angles_where}")
        return derotation_angles[::self.sample]

    def _get_sci_arr(self, input_cube_hdul, extname):
        return input_cube_hdul[extname].data.astype('=f8')[::self.sample]

    def _make_masks(self, sci_arr, extname):
        # mask file(s)

        if self.args.estimation_mask is not None:
            estimation_mask_hdul = iofits.load_fits_from_path(self.args.estimation_mask)
            if len(estimation_mask_hdul) > 0:
                estimation_mask = estimation_mask_hdul[extname].data
            else:
                estimation_mask = estimation_mask_hdul[0].data
        else:
            estimation_mask = np.ones(sci_arr.shape[1:])
        # coerce to bools
        estimation_mask = estimation_mask == 1

        # mask saturated
        if self.args.mask_saturated:
            percentile_image = np.nanpercentile(sci_arr, self.args.mask_saturated_percentile, axis=0)
            saturation_mask = percentile_image < self.args.mask_saturated_level
            estimation_mask &= saturation_mask

        # mask radial
        rho, theta = improc.polar_coords(improc.center(sci_arr.shape[1:]), sci_arr.shape[1:])
        if self.args.mask_iwa_px is not None:
            iwa_mask = rho >= self.args.mask_iwa_px
            estimation_mask &= iwa_mask
        if self.args.mask_owa_px is not None:
            owa_mask = rho <= self.args.mask_owa_px
            estimation_mask &= owa_mask

        # load combination mask + intersect with others
        if self.args.combination_mask is not None:
            combination_mask_hdul = iofits.load_fits_from_path(self.args.combination_mask)
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
        output_klip_final = utils.join(self.destination, "klip_final.fits")
        output_exptime_map = utils.join(self.destination, "exptime_map.fits")
        if self.check_for_outputs([output_klip_final, output_exptime_map]):
            return

        if len(self.all_files) > 1:
            raise RuntimeError(f"Not sure what to do with multiple inputs: {self.all_files}")
        input_cube_hdul = iofits.load_fits_from_path(self.all_files[0])

        obs_method = utils.parse_obs_method(input_cube_hdul[0].header['OBSMETHD'])

        derotation_angles = self._get_derotation_angles(input_cube_hdul, obs_method)
        exclude_nearest_n_frames = self.args.exclude_nearest_n_frames
        k_klip = self.args.k_klip
        klip_params = pipelines.KLIPParams(k_klip, exclude_nearest_n_frames)

        if 'vapp' in obs_method:
            left_extname = obs_method['vapp']['left']
            right_extname = obs_method['vapp']['right']

            sci_arr_left = self._get_sci_arr(input_cube_hdul, left_extname)
            sci_arr_right = self._get_sci_arr(input_cube_hdul, right_extname)

            estimation_mask_left, combination_mask_left = self._make_masks(sci_arr_left, left_extname)
            estimation_mask_right, combination_mask_right = self._make_masks(sci_arr_right, right_extname)

            outcube = pipelines.vapp_klip(
                pipelines.KLIPInput(da.from_array(sci_arr_left), estimation_mask_left, combination_mask_left),
                pipelines.KLIPInput(da.from_array(sci_arr_right), estimation_mask_right, combination_mask_right),
                pipelines.KLIPParams(k_klip_value=self.args.k_klip, exclude_nearest_n_frames=self.args.exclude_nearest_n_frames),
                self.args.vapp_symmetry_angle,
            )
        else:
            extname = obs_method.get('sci', 'SCI')
            if extname not in input_cube_hdul:
                log.error(f"Supply a 'SCI' extension, or use sci= or vapp.left= / vapp.right= in OBSMETHD")
                sys.exit(1)
            sci_arr = self._get_sci_arr(input_cube_hdul, extname)
            estimation_mask, combination_mask = self._make_masks(sci_arr, extname)
            outcube = pipelines.klip_one(
                pipelines.KLIPInput(sci_arr, estimation_mask, combination_mask),
                klip_params
            )

        out_image = pipelines.adi(outcube, derotation_angles, operation=self.args.combine_by)
        import time
        start = time.perf_counter()
        log.info(f"Computing klip pipeline result...")
        out_image = out_image.compute()
        elapsed = time.perf_counter() - start
        log.info(f"Computed in {elapsed} sec")
        output_file = iofits.write_fits(iofits.DaskHDUList([iofits.DaskHDU(out_image)]), output_klip_final)
        return output_file
