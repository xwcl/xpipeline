import logging
from typing import Optional, Union
import xconf
import pandas as pd
from xpipeline.ref import clio, magellan
from xpipeline.tasks import characterization
from .. import utils, constants, types
from .base import FitsConfig, InputCommand
import astropy.units as u

log = logging.getLogger(__name__)

import numpy as np

@xconf.config
class GridColumnsConfig:
    r_px : str = xconf.field(default="r_px", help="Radius (separation) in pixels")
    pa_deg : str = xconf.field(default="pa_deg", help="PA (angle) in degrees E of N (+Y)")
    snr : str = xconf.field(default="snr", help="Signal-to-noise ratio for a given location and parameters")
    injected_scale : str = xconf.field(default="injected_scale", help="(companion)/(host) contrast of injection")
    hyperparameters : list[str] = xconf.field(default_factory=lambda: ['k_modes'], help="List of columns holding hyperparameters varied in grid")


@xconf.config
class SummarizeGrid(InputCommand):
    ext : str = xconf.field(default="grid", help="FITS binary table extension with calibration grid")
    columns : GridColumnsConfig = xconf.field(default=GridColumnsConfig(), help="Grid column names")
    arcsec_per_pixel : float = xconf.field(default=clio.CLIO2_PIXEL_SCALE.value)
    wavelength_um : float = xconf.field(help="Wavelength in microns")
    primary_diameter_m : float = xconf.field(default=magellan.PRIMARY_MIRROR_DIAMETER.to(u.m).value)
    coverage_mask : FitsConfig = xconf.field(help="Mask image with 1s where pixels have observation coverage and 0 elsewhere")
    min_snr_for_injection: float = xconf.field(default=5, help="Minimum SNR to recover in order to trust the 5sigma contrast value")

    def main(self):
        import pandas as pd
        from ..tasks import iofits, improc
        output_filepath, = self.get_output_paths("summarize_grid.fits")
        self.quit_if_outputs_exist([output_filepath])

        coverage_mask = self.coverage_mask.load() == 1
        yc, xc = improc.arr_center(coverage_mask)

        grid_hdul = iofits.load_fits_from_path(self.input)
        grid_tbl = grid_hdul[self.ext].data
        log.info(f"Loaded {len(grid_tbl)} points for evaluation")
        grid_df = pd.DataFrame(grid_tbl)

        import pandas as pd

        limits_df, detections_df = characterization.summarize_grid(
            grid_df,
            r_px_colname=self.columns.r_px,
            pa_deg_colname=self.columns.pa_deg,
            snr_colname=self.columns.snr,
            injected_scale_colname=self.columns.injected_scale,
            hyperparameter_colnames=self.columns.hyperparameters,
            min_snr_for_injection=self.min_snr_for_injection,
        )
        limits_df['delta_mag_contrast_limit_5sigma'] = characterization.contrast_to_deltamag(limits_df['contrast_limit_5sigma'].to_numpy())
        for df in [limits_df, detections_df]:
            df['r_as'] = (df['r_px'].to_numpy() * u.pix * self.arcsec_per_pixel).value
            df['r_lambda_over_d'] = characterization.arcsec_to_lambda_over_d(
                df['r_as'].to_numpy() * u.arcsec,
                self.wavelength_um * u.um,
                d=self.primary_diameter_m * u.m
            )

        log.info(f"Sampled {len(limits_df)} locations for contrast limits and detection")

        lim_xs, lim_ys = characterization.r_pa_to_x_y(limits_df[self.columns.r_px], limits_df[self.columns.pa_deg], xc, yc)
        contrast_lim_map = characterization.points_to_map(lim_xs, lim_ys, limits_df['contrast_limit_5sigma'], coverage_mask)
        det_xs, det_ys = characterization.r_pa_to_x_y(detections_df[self.columns.r_px], detections_df[self.columns.pa_deg], xc, yc)
        detection_map = characterization.points_to_map(det_xs, det_ys, detections_df['snr'], coverage_mask)

        hdus = [iofits.DaskHDU(None, kind="primary")]
        hdus.append(iofits.DaskHDU(contrast_lim_map, name="limits_5sigma_contrast_map"))
        hdus.append(iofits.DaskHDU(utils.convert_obj_cols_to_str(limits_df.to_records(index=False)), kind="bintable", name="limits"))
        hdus.append(iofits.DaskHDU(detection_map, name="detection_snr_map"))
        hdus.append(iofits.DaskHDU(utils.convert_obj_cols_to_str(detections_df.to_records(index=False)), kind="bintable", name="detection"))
        iofits.write_fits(iofits.DaskHDUList(hdus), output_filepath)
