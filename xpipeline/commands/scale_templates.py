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
        from ..core import LazyPipelineCollection
        from ..tasks import iofits
        from .. import pipelines, utils

        destination = self.destination
        dest_fs = utils.get_fs(destination)
        dest_fs.makedirs(destination, exist_ok=True)

        output_filepath = utils.join(destination, utils.basename("scale_templates.txt"))
        self.quit_if_outputs_exist([output_filepath])

        all_inputs = self.get_all_inputs()
        coll = LazyPipelineCollection(all_inputs).map(iofits.load_fits_from_path)
        template_hdul = iofits.load_fits_from_path(self.template_path)
        res = pipelines.compute_scale_factors(coll, template_hdul, self.saturation_threshold)
        factors = res.compute()
        log.debug(f'{factors=}')
        with dest_fs.open(output_filepath, 'w') as fh:
            for factor in factors:
                fh.write(f'{factor}\n')
