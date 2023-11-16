import logging
from typing import Optional
import xconf

from .base import MultiInputCommand

log = logging.getLogger(__name__)

@xconf.config
class ScaleTemplates(MultiInputCommand):
    """Compute a scale factor fitting the radial profile of
    a template PSF to frames from a data cube"""
    template_path : str = xconf.field(help="Path to FITS file with templates in extensions")
    saturation_threshold : Optional[float] = xconf.field(default=None, help="Value in counts above which pixels should be considered saturated and ignored for scaling purposes")

    def main(self):
        import dask
        from ..core import LazyPipelineCollection
        from ..tasks import iofits, improc
        from .. import pipelines, utils

        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)

        output_filepath = utils.join(destination, utils.basename("template_scale_factors.fits"))
        self.quit_if_outputs_exist([output_filepath])

        all_inputs = self.get_all_inputs(self.input)
        template_hdul = iofits.load_fits_from_path(self.template_path)
        if len(all_inputs) == 1:
            data_hdul = iofits.load_fits_from_path(all_inputs[0])
            factors_hdul = pipelines.compute_scale_factors(data_hdul, template_hdul, self.saturation_threshold)
        else:
            coll = LazyPipelineCollection(all_inputs).map(iofits.load_fits_from_path)
            factors_hdul = pipelines.compute_scale_factors(coll, template_hdul, self.saturation_threshold)
        factors_hdul, = dask.compute(factors_hdul)
        iofits.write_fits(factors_hdul, output_filepath)
