import logging
from typing import Optional, Union
import xconf
import pandas as pd
from xpipeline.ref import clio
from xpipeline.tasks import characterization
from .. import utils, constants, types
from .base import InputCommand

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
    def main(self):
        from ..tasks import iofits
        output_filepath = self.get_output_paths("interpret.fits")
        self.quit_if_outputs_exist(output_filepath)
        
        grid_hdul = iofits.load_fits_from_path(self.input)
        grid_tbl = grid_hdul[self.ext].data

        import pandas as pd

        limits_df, detections_df = characterization.summarize_grid(
            pd.DataFrame(grid_tbl),
            r_px_colname=self.columns.r_px,
            pa_deg_colname=self.columns.pa_deg,
            snr_colname=self.columns.snr,
            injected_scale_colname=self.columns.injected_scale,
            hyperparameter_colnames=self.columns.hyperparameters,
        )

        hdus = [iofits.DaskHDU(None, kind="primary")]
        hdus.append(iofits.DaskHDU(limits_df.to_records(), kind="table", name="limits"))
        hdus.append(iofits.DaskHDU(detections_df.to_records(), kind="table", name="detections"))
        iofits.write_fits(iofits.DaskHDUList(hdus), output_filepath)

