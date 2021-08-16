import logging
from typing import Optional, Union
import xconf
from .. import utils, constants
from .base import InputCommand

log = logging.getLogger(__name__)

@xconf.config
class MakeTemplates(InputCommand):
    "Turn multi-extension FITS file of observation sequences into templates"
    combine_input_by : constants.CombineOperation = xconf.field(default=constants.CombineOperation.MEAN, help="Operation used to combine input frames to make templates")
    
    def _load_dataset(self, dataset_path):
        from ..tasks import iofits
        input_cube_hdul = iofits.load_fits_from_path(dataset_path)
        obs_method = utils.parse_obs_method(input_cube_hdul[0].header["OBSMETHD"])
        return input_cube_hdul, obs_method

    def _get_derotation_angles(self, dataset_hdul, obs_method):
        if "adi" in obs_method:
            where_angles = obs_method["adi"]["derotation_angles"]
            if '.' in where_angles:
                tbl_ext, col_name = where_angles.split('.', 1)
                derotation_angles = dataset_hdul[tbl_ext].data[col_name]
            else:
                derotation_angles = dataset_hdul[where_angles].data
        else:
            derotation_angles = None
        return derotation_angles
    
    def _set_derotation_angles(self, dataset_hdul, obs_method, derotation_angles):
        if "adi" in obs_method:
            where_angles = obs_method["adi"]["derotation_angles"]
            if '.' in where_angles:
                tbl_ext, col_name = where_angles.split('.', 1)
                dataset_hdul[tbl_ext].data[col_name] = derotation_angles
            else:
                dataset_hdul[where_angles].data = derotation_angles

    def _combine_obs(self, dataset_hdul, obs_method):
        import numpy as np
        from ..tasks import improc
        # what to combine
        obs_to_combine = []
        for idx, hdu in enumerate(dataset_hdul):
            if hdu.kind != "bintable" and hdu.data is not None and len(hdu.data.shape) >= 1:
                obs_to_combine.append((idx, hdu.data))
        obs_table = dataset_hdul['OBSTABLE'].data
        
        # create range spec
        total_n_frames = obs_to_combine[0][1].shape[0]
        squeeze_axes = False
        if hasattr(self.combine, 'n_frames'):
            range_spec = improc.FrameIndexRangeSpec(n_frames=self.combine.n_frames)
        elif hasattr(self.combine, "angle"):
            if hasattr(self.combine.angle, "delta_deg"):
                range_spec = improc.AngleRangeSpec(delta_deg=self.combine.angle.delta_deg)
            else:
                range_spec = improc.PixelRotationRangeSpec(delta_px=self.combine.angle.delta_px, r_px=self.combine.angle.r_px)
        elif hasattr(self.combine, 'all'):
            if not self.combine:
                raise ValueError("Supply a range config or all = true")
            squeeze_axes = True
            range_spec = improc.FrameIndexRangeSpec(n_frames=total_n_frames)
        else:
            raise RuntimeError("self.combine isn't a CombineConfig")
        if range_spec == improc.FrameIndexRangeSpec(n_frames=1):
            log.debug(f"Got {range_spec=}, skipping combine")
            return dataset_hdul
        log.debug(f"Combining observations according to {range_spec=}")
        
        # do combination
        combined_obs, combined_angles, combined_metadata = improc.combine_ranges(
            [obs for _, obs in obs_to_combine],
            obs_table,
            range_spec,
            operation=self.combine_input_by
        )
        for new_idx, (old_idx, _) in enumerate(obs_to_combine):
            result = combined_obs[new_idx]
            if squeeze_axes:
                result = np.squeeze(combined_obs[new_idx])
            dataset_hdul.hdus[old_idx].data = result
        if 'OBSTABLE' in dataset_hdul:
            dataset_hdul['OBSTABLE'].data = combined_metadata
        log.info(f"Combined {total_n_frames} observations to get {combined_obs[0].shape[0]} observations")
        self._set_derotation_angles(dataset_hdul, obs_method, derotation_angles=combined_angles)
        return dataset_hdul

    def main(self):
        from ..tasks import iofits
        output_combined_data_fn = utils.join(self.destination, "combined_dataset.fits")
        outputs = [output_combined_data_fn]
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)
        self.quit_if_outputs_exist(outputs)

        dataset_hdul, obs_method = self._load_dataset(self.input)
        dataset_hdul = self._combine_obs(dataset_hdul, obs_method)
        iofits.write_fits(dataset_hdul, output_combined_data_fn)
