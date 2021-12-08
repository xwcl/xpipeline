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
    r_px : str = xconf.field(default="r_px", help="")
    pa_deg : str = xconf.field(default="pa_deg", help="")
    detected_snr : str = xconf.field(default="detected_snr", help="Signal-to-noise ratio at a given location without injecting a signal")
    recovered_snr : str = xconf.field(default="recovered_snr", help="Signal-to-noise ratio at a given location with an injected companion")
    injected_scale : str = xconf.field(default="injected_scale", help="(companion)/(host) contrast of injection")

@xconf.config
class SummarizeGrid(InputCommand):
    ext : str = xconf.field(default="GRID", help="FITS binary table extension with calibration grid")
    columns : GridColumnsConfig = xconf.field(default=GridColumnsConfig(), help="Grid column names")
    arcsec_per_pixel : float = xconf.field(default=clio.CLIO2_PIXEL_SCALE.value)
    def main(self):
        from ..tasks import iofits
        dest_fs = self.get_dest_fs()
        output_filepath = self.get_output_paths("interpret.fits")
        
        grid_hdul = iofits.load_fits_from_path(self.input)
        grid_tbl = grid_hdul[self.ext].data



        summary_tbl = characterization.summarize_grid(
            grid_tbl,
            r_px_colname=self.columns.r_px,
            pa_deg_colname=self.columns.pa_deg,
            snr_colname=self.columns.recovered_snr,
            injected_scale_colname=self.columns.injected_scale,
        )

        hdus = [iofits.DaskHDU(None, kind="primary")]
        hdus.append(iofits.DaskHDU(summary_tbl, kind="table"))
        iofits.write_fits(iofits.DaskHDUList(hdus), output_filepath)

