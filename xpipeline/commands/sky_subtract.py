import sys
from typing import Optional
from xpipeline.core import LazyPipelineCollection
import logging
import xconf

from .base import MultiInputCommand, FileConfig

log = logging.getLogger(__name__)

@xconf.config
class SkySubtract(MultiInputCommand):
    """Subtract sky background with a PCA basis model file"""
    sky_model_path : str = xconf.field(help="Path to FITS file with sky model basis")
    mask_dilate_iters : int = xconf.field(default=2, help="Number of times to grow mask regions before selecting estimation pixels")
    n_sigma : float = xconf.field(default=3, help="Number of sigma (standard deviations of the background model input frames) beyond which pixel is considered illuminated and excluded from background estimation")
    excluded_regions : Optional[FileConfig] = xconf.field(default_factory=list, help="Regions presumed illuminated to be excluded from background estimation, stored as DS9 region file (reg format)")

    def main(self):
        import fsspec.spec
        from .. import utils
        from .. import pipelines
        from ..ref import clio
        from ..tasks import iofits, sky_model, regions

        destination = self.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        all_inputs = self.get_all_inputs()
        n_output_files = len(all_inputs)
        output_filepaths = [utils.join(destination, f"sky_subtract_{i:04}.fits") for i in range(n_output_files)]
        self.quit_if_outputs_exist(output_filepaths)

        if isinstance(self.excluded_regions, FileConfig):
            with self.excluded_regions.open() as fh:
                excluded_regions = regions.load_file(fh)
        coll = LazyPipelineCollection(all_inputs).map(iofits.load_fits_from_path)
        hdul = iofits.load_fits_from_path(self.sky_model_path)
        model_sky = sky_model.SkyModel.from_hdulist(hdul)
        excluded_pixels_mask = regions.make_mask(excluded_regions, model_sky.mean_sky.shape)

        output_coll = pipelines.sky_subtract(coll, model_sky, self.mask_dilate_iters, self.n_sigma, excluded_pixels_mask=excluded_pixels_mask)
        result = output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True).compute()
        log.info(result)

