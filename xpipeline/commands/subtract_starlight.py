import xconf
import sys
import logging
from typing import Optional, Union
from enum import Enum
import numpy as np

from .. import utils, constants

from .base import BaseCommand, AngleRangeConfig, PixelRotationRangeConfig, FitsConfig

log = logging.getLogger(__name__)

@xconf.config
class ExcludeRangeConfig:
    angle_deg_col : str = xconf.field(default="derotation_angle_deg", help="Column with angle values per frame to use for exclusion")
    angle : Union[AngleRangeConfig,PixelRotationRangeConfig] = xconf.field(default=AngleRangeConfig(), help="Apply exclusion to derotation angles")
    nearest_n_frames : int = xconf.field(default=0, help="Number of additional temporally-adjacent frames on either side of the target frame to exclude from the sequence when computing the KLIP eigenimages")

@xconf.config
class KlipInputConfig:
    sci_arr: FitsConfig
    signal_arr: Optional[FitsConfig]
    estimation_mask: Optional[FitsConfig]
    mask_min_r_px : Union[int,float] = xconf.field(default=0, help="Apply radial mask excluding pixels < mask_min_r_px from center")
    mask_max_r_px : Union[int,float,None] = xconf.field(default=None, help="Apply radial mask excluding pixels > mask_max_r_px from center")

@xconf.config
class SubtractStarlight(BaseCommand):
    "Subtract starlight with KLIP"
    inputs : list[KlipInputConfig] = xconf.field(help="Input data to simultaneously reduce")
    destination : str = xconf.field(help="Destination path to save results")
    obstable : FitsConfig = xconf.field(help="Metadata table in FITS")
    k_klip : int = xconf.field(default=10, help="Number of modes to subtract in starlight subtraction")
    exclude : ExcludeRangeConfig = xconf.field(default=ExcludeRangeConfig(), help="How to exclude frames from reference sample")
    strategy : constants.KlipStrategy = xconf.field(default=constants.KlipStrategy.DOWNDATE_SVD, help="Implementation of KLIP to use")
    reuse_eigenimages : bool = xconf.field(default=False, help="Apply KLIP without adjusting the eigenimages at each step (much faster, less powerful)")
    initial_decomposition_only : bool = xconf.field(default=False, help="Whether to output initial decomposition and exit")
    initial_decomposition_path : Optional[str] = xconf.field(default=None, help="Initial decomposition as FITS file")

    def _load_obstable(self):
        from ..tasks import iofits
        obstable_hdul = iofits.load_fits_from_path(self.obstable.path)
        obstable = obstable_hdul[self.obstable.ext].data
        return obstable

    def _load_initial_decomposition(self, path):
        from ..tasks import iofits, starlight_subtraction
        decomp_hdul = iofits.load_fits_from_path(path)
        return starlight_subtraction.InitialDecomposition(
            mtx_u0=decomp_hdul['MTX_U0'].data,
            diag_s0=decomp_hdul['DIAG_S0'].data,
            mtx_v0=decomp_hdul['MTX_V0'].data,
        )

    def _save_initial_decomposition(self, initial_decomposition, output_initial_decomp_fn):
        from ..tasks import iofits
        hdus = [
            iofits.DaskHDU(data=None, kind="primary"),
            iofits.DaskHDU(initial_decomposition.mtx_u0, name="MTX_U0"),
            iofits.DaskHDU(initial_decomposition.diag_s0, name="DIAG_S0"),
            iofits.DaskHDU(initial_decomposition.mtx_v0, name="MTX_V0"),
        ]
        iofits.write_fits(iofits.DaskHDUList(hdus), output_initial_decomp_fn)


    def _make_mask(self, klip_input_cfg : KlipInputConfig, sci_arr : np.ndarray):
        from ..tasks import iofits, improc

        # mask file(s)
        if klip_input_cfg.estimation_mask is not None:
            estimation_mask_hdul = iofits.load_fits_from_path(klip_input_cfg.estimation_mask.path)
            estimation_mask = estimation_mask_hdul[klip_input_cfg.estimation_mask.ext].data
        else:
            estimation_mask = np.ones(sci_arr.shape[1:])
        # coerce to bools
        estimation_mask = estimation_mask == 1

        if klip_input_cfg.mask_min_r_px == 0 and klip_input_cfg.mask_max_r_px is None:
            return estimation_mask
        else:
            ctr = improc.arr_center(sci_arr.shape[1:])
            frame_shape = sci_arr.shape[1:]
            if klip_input_cfg.mask_min_r_px is None:
                min_r = 0
            else:
                min_r = klip_input_cfg.mask_min_r_px
            annular_mask = improc.mask_arc(ctr, frame_shape, from_radius=min_r, to_radius=klip_input_cfg.mask_max_r_px)
            estimation_mask &= annular_mask
            return estimation_mask

    def _assemble_klip_input(self, klip_input_cfg : KlipInputConfig, obstable : np.ndarray):
        from ..tasks import iofits, starlight_subtraction
        input_hdul = iofits.load_fits_from_path(klip_input_cfg.sci_arr.path)
        sci_arr = input_hdul[klip_input_cfg.sci_arr.ext].data
        if klip_input_cfg.signal_arr is not None:
            if klip_input_cfg.signal_arr.path != klip_input_cfg.sci_arr.path:
                signal_hdul = iofits.load_fits_from_path(klip_input_cfg.sci_arr.path) 
            else:
                signal_hdul = input_hdul
            signal_arr = signal_hdul[klip_input_cfg.signal_arr.ext].data
        else:
            signal_arr = None
        estimation_mask = self._make_mask(klip_input_cfg, sci_arr)
        return starlight_subtraction.KlipInput(sci_arr, obstable, estimation_mask, signal_arr)


    def main(self):
        from ..tasks import iofits
        output_subtracted_fn = utils.join(self.destination, "starlight_subtracted.fits")
        output_initial_decomp_fn = utils.join(self.destination, "initial_decomposition.fits")
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)
        if self.initial_decomposition_only:
            if dest_fs.exists(output_initial_decomp_fn):
                raise FileExistsError(f"Output exists at {output_initial_decomp_fn}")
        else:
            if dest_fs.exists(output_subtracted_fn):
                raise FileExistsError(f"Output exists at {output_subtracted_fn}")


        obstable = self._load_obstable()
        klip_inputs = []
        for inputcfg in self.inputs:
            klip_inputs.append(self._assemble_klip_input(inputcfg, obstable))
        
        if self.initial_decomposition_path is not None:
            initial_decomposition = self._load_initial_decomposition(self.initial_decomposition_path)
        else:
            initial_decomposition = None
        
        klip_params = self._assemble_klip_params(obstable, initial_decomposition)
        import time
        start = time.perf_counter()

        from ..pipelines import klip_multi
        result = klip_multi(klip_inputs, klip_params)
        if klip_params.initial_decomposition_only:
            self._save_initial_decomposition(result, output_initial_decomp_fn)
            return 0
        else:
            outcubes, outmeans = result
        elapsed = time.perf_counter() - start
        log.info(f"Computed in {elapsed} sec")
        hdus = [iofits.DaskHDU(data=None, kind="primary")]
        for idx in range(len(outcubes)):
            hdus.append(iofits.DaskHDU(outcubes[idx], name=f"SCI_{idx}"))
            hdus.append(iofits.DaskHDU(outmeans[idx], name=f"MEAN_{idx}"))

        iofits.write_fits(
            iofits.DaskHDUList(hdus), output_subtracted_fn
        )

    def _make_exclusions(self, exclude : ExcludeRangeConfig, obstable):
        import numpy as np
        from ..tasks import starlight_subtraction
        exclusions = []
        derotation_angles = obstable[exclude.angle_deg_col]
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

    def _assemble_klip_params(self, obstable, initial_decomposition):
        import numpy as np
        from ..tasks import starlight_subtraction
        exclusions = self._make_exclusions(self.exclude, obstable)
        klip_params = starlight_subtraction.KlipParams(
            self.k_klip,
            exclusions,
            decomposer=starlight_subtraction.DEFAULT_DECOMPOSERS[self.strategy],
            strategy=self.strategy,
            reuse=self.reuse_eigenimages,
            initial_decomposition_only=self.initial_decomposition_only,
            initial_decomposition=initial_decomposition
        )
        log.debug(klip_params)
        return klip_params
