import sys
import argparse
from xpipeline.core import LazyPipelineCollection
import fsspec.spec
import logging
import xconf

from .. import utils
from .. import pipelines
from ..ref import clio
from ..tasks import iofits, sky_model, improc


from .base import MultiInputCommand

log = logging.getLogger(__name__)

@xconf.config
class ExcludedRegion:
    origin_x : int = xconf.field(help="Origin X pixel")
    origin_y : int = xconf.field(help="Origin Y pixel")
    width : int = xconf.field(help="Width of region")
    height : int = xconf.field(help="Height of region")

@xconf.config
class SkySubtract(MultiInputCommand):
    """Subtract sky background with a PCA basis model file"""
    sky_model_path : str = xconf.field(help="Path to FITS file with sky model basis")
    mask_dilate_iters : int = xconf.field(default=6, help="Number of times to grow mask regions before selecting estimation pixels")
    n_sigma : float = xconf.field(default=3, help="Number of sigma (standard deviations of the background model input frames) beyond which pixel is considered illuminated and excluded from background estimation")
    excluded_regions : dict[str, ExcludedRegion] = xconf.field(default_factory=dict, help="Regions presumed illuminated to be excluded from background estimation")

    def main(self):
        destination = self.destination
        dest_fs = utils.get_fs(destination)
        assert isinstance(dest_fs, fsspec.spec.AbstractFileSystem)
        log.debug(f"calling makedirs on {dest_fs} at {destination}")
        dest_fs.makedirs(destination, exist_ok=True)

        all_inputs = self.get_all_inputs()
        n_output_files = len(all_inputs)
        output_filepaths = [utils.join(destination, f"sky_subtract_{i:04}.fits") for i in range(n_output_files)]
        for output_file in output_filepaths:
            if dest_fs.exists(output_file):
                log.error(f"Output exists: {output_file}")
                sys.exit(1)

        excluded_bboxes = []
        for _, er in self.excluded_regions.items():
            excluded_bboxes.append(improc.BBox(
                origin=improc.Pixel(y=er.origin_y, x=er.origin_x),
                extent=improc.PixelExtent(height=er.height, width=er.width)
            ))

        coll = LazyPipelineCollection(all_inputs).map(iofits.load_fits_from_path)
        hdul = iofits.load_fits_from_path(self.sky_model_path)
        model_sky = sky_model.SkyModel.from_hdulist(hdul)
        output_coll = pipelines.sky_subtract(coll, model_sky, self.mask_dilate_iters, self.n_sigma, excluded_bboxes)
        return output_coll.zip_map(iofits.write_fits, output_filepaths, overwrite=True).compute()

