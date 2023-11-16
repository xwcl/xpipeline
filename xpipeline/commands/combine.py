import logging
from typing import Optional, Union
import xconf
from .. import utils, constants, types
from .base import InputCommand
from .base import CompanionConfig, TemplateConfig, AngleRangeConfig, PixelRotationRangeConfig

log = logging.getLogger(__name__)

@xconf.config
class CombineFramesConfig:
    n_frames : int = xconf.field(default=1, help="Maximum number of frames per chunk (final chunk may be smaller)")

@xconf.config
class CombineWallTimeConfig:
    delta_t_sec : float = xconf.field(help="Interval of wall time (not open shutter time) to cover")
    time_secs_col : str = xconf.field(default='DATE-OBS_TS', help="Observation table column containing time in seconds")


@xconf.config
class CombineAllConfig:
    all : bool = xconf.field(default=False, help="Whether to collapse all frames to one, removing time dimension (i.e. cube to image)")

@xconf.config
class CombineAnglesConfig:
    angle : Union[AngleRangeConfig,PixelRotationRangeConfig] = xconf.field(default=AngleRangeConfig(), help="Combine frames with derotation angles within range")

CombineConfig = Union[CombineFramesConfig, CombineWallTimeConfig, CombineAnglesConfig, CombineAllConfig]

@xconf.config
class Combine(InputCommand):
    "Downsample dataset according to various criteria for ranges to average"
    combine : CombineConfig = xconf.field(help="Criteria by which to combine chunks of frames")
    combine_input_by : constants.CombineOperation = xconf.field(default=constants.CombineOperation.MEAN, help="Operation used to combine ranges of input frames to downsample before reduction")
    
    def _load_dataset(self, dataset_path):
        from ..tasks import iofits
        input_cube_hdul = iofits.load_fits_from_path(dataset_path)
        obs_method = utils.parse_obs_method(input_cube_hdul[0].header["OBSMETHD"])
        return input_cube_hdul, obs_method

    def _combine_obs(self, dataset_hdul, obs_method):
        import numpy as np
        from ..tasks import improc
        # what to combine
        obs_to_combine = []
        for idx, hdu in enumerate(dataset_hdul):
            if hdu.kind != "bintable" and hdu.data is not None and len(hdu.data.shape) >= 1:
                obs_to_combine.append((idx, hdu.data))
        if 'OBSTABLE' in dataset_hdul:
            metadata = dataset_hdul['OBSTABLE'].data
        else:
            metadata = None
        
        # create range spec
        total_n_frames = obs_to_combine[0][1].shape[0]
        combine_all = False
        if hasattr(self.combine, 'n_frames'):
            range_spec = improc.FrameIndexRangeSpec(n_frames=self.combine.n_frames)
        elif hasattr(self.combine, "angle"):
            if hasattr(self.combine.angle, "delta_deg"):
                range_spec = improc.AngleRangeSpec(delta_deg=self.combine.angle.delta_deg, angle_deg_column_name='derotation_angle_deg')
            else:
                range_spec = improc.PixelRotationRangeSpec(delta_px=self.combine.angle.delta_px, r_px=self.combine.angle.r_px)
        elif hasattr(self.combine, 'all'):
            if not self.combine:
                raise ValueError("Supply a range config or all = true")
            combine_all = True
            range_spec = improc.FrameIndexRangeSpec(n_frames=total_n_frames)
        elif hasattr(self.combine, 'delta_t_sec'):
            range_spec = improc.WallTimeRangeSpec(delta_t_sec=self.combine.delta_t_sec, time_secs_col=self.combine.time_secs_col)
        else:
            raise RuntimeError("self.combine isn't a CombineConfig")
        if range_spec == improc.FrameIndexRangeSpec(n_frames=1):
            log.debug(f"Got {range_spec=}, skipping combine")
            return dataset_hdul
        log.debug(f"Combining observations according to {range_spec=}")
        
        # do combination
        if combine_all:
            combined_obs = [improc.combine(obs, self.combine_input_by) for _, obs in obs_to_combine]
            for x in combined_obs:
                x /= np.sum(x)
            log.info(f"Combined {total_n_frames} observations to make a normalized template")
        else:
            combined_obs, combined_metadata = improc.combine_ranges(
                [obs for _, obs in obs_to_combine],
                metadata,
                range_spec,
                operation=self.combine_input_by
            )
            if 'OBSTABLE' in dataset_hdul:
                dataset_hdul['OBSTABLE'].data = combined_metadata
            log.info(f"Combined {total_n_frames} observations to get {combined_obs[0].shape[0]} observations")
        for new_idx, (old_idx, _) in enumerate(obs_to_combine):
            result = combined_obs[new_idx]
            dataset_hdul.hdus[old_idx].data = result
        return dataset_hdul

    def main(self):
        from ..tasks import iofits
        output_combined_data_fn = utils.join(self.destination, "combine.fits")
        outputs = [output_combined_data_fn]
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)
        self.quit_if_outputs_exist(outputs)

        dataset_hdul, obs_method = self._load_dataset(self.input)
        dataset_hdul = self._combine_obs(dataset_hdul, obs_method)
        iofits.write_fits(dataset_hdul, output_combined_data_fn)
