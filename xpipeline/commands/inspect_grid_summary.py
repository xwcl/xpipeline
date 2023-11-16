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
class InspectGridSummary(InputCommand):
    ext : str = xconf.field(default="grid", help="FITS binary table extension with calibration grid")
    columns : GridColumnsConfig = xconf.field(default=GridColumnsConfig(), help="Grid column names")
    arcsec_per_pixel : float = xconf.field(default=clio.CLIO2_PIXEL_SCALE.value)
    wavelength_um : float = xconf.field(help="Wavelength in microns")
    primary_diameter_m : float = xconf.field(default=magellan.PRIMARY_MIRROR_DIAMETER.to(u.m).value)
    coverage_mask : FitsConfig = xconf.field(help="Mask image with 1s where pixels have observation coverage and 0 elsewhere")
    min_snr_for_injection: float = xconf.field(default=5, help="Minimum SNR to recover in order to trust the 5sigma contrast value")
    normalize_snrs: bool = xconf.field(default=False, help="Rescales the SNR values at a given radius by dividing by the stddev of all SNRs for that radius")
    filters : dict[str, FilterSpec] = xconf.field(default_factory=dict, help="Filter on column == value before grouping and summarizing")

    def main(self):
        pass